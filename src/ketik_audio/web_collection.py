"""Import browser collection packages without changing the frozen terminal dataset."""

from __future__ import annotations

import io
import json
import math
import re
import stat
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import soundfile as sf

from . import web_quota
from .common import LABELS, ResearchError, file_hash, fresh_directory, now, write_json

SCHEMA = "ketik-web-session/v1"
FILES = {"audio.wav", "events.jsonl", "timing.json", "session.json"}
MAX_BYTES = 160 * 1024**2
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", re.I)
ROLES = ("calibration", "test1", "test2")
EVENTS = {
    "keydown",
    "keyup",
    "input",
    "beforeinput",
    "compositionstart",
    "compositionend",
    "paste",
    "drop",
}
PHASES = {"quiet", "sync_start", "typing", "sync_end"}


def require(condition, message):
    if not condition:
        raise ResearchError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def decode_json(data):
    def pairs(entries):
        result = {}
        for k, value in entries:
            require(k not in result, f"Properti JSON berulang: {k}")
            result[k] = value
        return result

    def constant(value):
        raise ResearchError(f"Konstanta JSON tidak finite: {value}")

    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError) as exc:
        raise ResearchError(f"JSON tidak valid: {exc}") from exc


def archive_payload(path):
    require(path.stat().st_size <= MAX_BYTES, "ZIP melebihi batas 160 MiB.")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            require(
                len(entries) == len(FILES) and {i.filename for i in entries} == FILES,
                "ZIP harus memuat tepat empat berkas format web v1, tanpa path atau duplikat.",
            )
            require(sum(i.file_size for i in entries) <= MAX_BYTES, "Isi ZIP melebihi 160 MiB.")
            require(
                all(
                    not stat.S_ISLNK(i.external_attr >> 16) and not i.flag_bits & 1 for i in entries
                ),
                "Symlink atau ZIP terenkripsi tidak didukung.",
            )
            require(
                all(i.file_size <= 1024**2 for i in entries if i.filename == "session.json"),
                "Metadata sesi terlalu besar.",
            )
            return {i.filename: archive.read(i) for i in entries}
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        raise ResearchError(f"ZIP rusak atau tidak didukung: {exc}") from exc


def validate_metadata(meta):
    require(
        isinstance(meta, dict) and meta.get("schema") in {SCHEMA, web_quota.SCHEMA},
        "Versi format sesi web tidak didukung.",
    )
    is_v2 = meta["schema"] == web_quota.SCHEMA
    require(
        meta.get("status") in {"complete", "interrupted"},
        "Sesi belum difinalisasi atau dipulihkan oleh web.",
    )
    require(type(meta.get("synthetic")) is bool, "Penanda data sintetis wajib ada.")
    require(meta.get("role") in ROLES, "Peran sesi tidak dikenal.")
    require(
        all(
            isinstance(meta.get(k), str) and UUID.fullmatch(meta[k])
            for k in ("session_id", "study_id", "participant_id")
        ),
        "ID sesi, studi, atau peserta tidak valid.",
    )
    require(
        meta.get("channels") == 1 and meta.get("format") == "IEEE_FLOAT_32_LE",
        "Audio harus PCM float32 mono.",
    )
    require(
        type(meta.get("sample_rate")) is int and 8000 <= meta["sample_rate"] <= 96000,
        "Sample rate tidak didukung.",
    )
    require(
        type(meta.get("frames")) is int
        and 0 < meta["frames"]
        and (
            meta["frames"] * 4 <= web_quota.MAX_AUDIO_BYTES
            if is_v2
            else meta["frames"] <= 301 * meta["sample_rate"]
        ),
        "Jumlah sampel di luar batas sesi.",
    )
    require(
        type(meta.get("first_frame")) is int and meta["first_frame"] >= 0,
        "Origin frame tidak valid.",
    )
    require(meta.get("timing_status") in {"checked", "needs_review"}, "Status timing tidak valid.")
    require(
        isinstance(meta.get("capture_issues"), list)
        and all(isinstance(x, str) for x in meta["capture_issues"]),
        "Daftar gangguan perekaman tidak valid.",
    )
    require(
        meta["status"] != "complete" or not meta["capture_issues"],
        "Sesi lengkap masih memuat gangguan perekaman.",
    )
    require(
        isinstance(meta.get("consent"), dict)
        and meta["consent"].get("version") == "local-audio-key-events-v1",
        "Versi persetujuan tidak tersedia.",
    )
    study = meta.get("study")
    require(
        isinstance(study, dict)
        and study.get("schema") == ("ketik-web-study/v2" if is_v2 else "ketik-web-study/v1"),
        "Profil studi tidak tersedia.",
    )
    if is_v2:
        require(study.get("collection") == web_quota.config(), "Protokol kuota studi berbeda.")
        web_quota.validate_preflight(meta)
    require(
        all(study.get(k) == meta[k] for k in ("study_id", "participant_id")),
        "Identitas sesi dan studi berbeda.",
    )
    require(study.get("timezone") == meta.get("timezone"), "Zona waktu studi berubah.")
    try:
        zone = ZoneInfo(meta["timezone"])
        start, end = (
            datetime.fromisoformat(meta[k].replace("Z", "+00:00"))
            for k in ("started_at", "finished_at")
        )
        require(
            start.utcoffset() is not None and end.utcoffset() is not None and end >= start,
            "Jam sesi tidak valid.",
        )
        require(
            start.astimezone(zone).date().isoformat() == meta.get("local_day"),
            "Tanggal sesi tidak cocok dengan zona waktu.",
        )
        days = study["days"]
        previous = None
        for role in ROLES:
            day = days[role]
            if day is not None:
                require(
                    isinstance(day, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", day),
                    "Format hari studi tidak valid.",
                )
                datetime.fromisoformat(day)
                require(
                    previous is None or day > previous, "Hari studi harus berbeda dan berurutan."
                )
                if ROLES.index(role):
                    require(previous is not None, "Hari studi sebelumnya tidak ada.")
            previous = day
        if meta["status"] == "complete":
            require(days[meta["role"]] == meta["local_day"], "Hari tahap studi berbeda dari sesi.")
        for previous_role in ROLES[: ROLES.index(meta["role"])]:
            require(
                days[previous_role] is not None and days[previous_role] < meta["local_day"],
                "Pengujian harus pada hari setelah tahap sebelumnya.",
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ResearchError(f"Metadata waktu tidak valid: {exc}") from exc
    require(
        isinstance(study.get("setup"), dict)
        and all(
            isinstance(study["setup"].get(k), str) and 0 < len(study["setup"][k]) <= 120
            for k in ("laptop", "keyboard")
        ),
        "Alias setup tidak lengkap.",
    )


def read_events(data, expected):
    lines = data.splitlines()
    require(
        len(lines) == expected and len(lines) <= 100_000, "Jumlah event berbeda atau terlalu besar."
    )
    events = []
    for i, line in enumerate(lines):
        require(len(line) <= 100_000, "Satu event terlalu besar.")
        row = decode_json(line)
        require(
            isinstance(row, dict)
            and row.get("seq") == i
            and row.get("type") in EVENTS
            and row.get("phase") in PHASES,
            "Urutan atau jenis event tidak valid.",
        )
        require(
            all(finite(row.get(k)) and row[k] >= 0 for k in ("time_ms", "handler_ms")),
            "Jam event tidak finite.",
        )
        require(
            not events or row["handler_ms"] >= events[-1]["handler_ms"],
            "Jam penanganan event mundur.",
        )
        if row["type"] in {"keydown", "keyup"}:
            require(
                all(isinstance(row.get(k), str) and len(row[k]) <= 100 for k in ("key", "code")),
                "Nama tombol tidak valid.",
            )
            require(
                all(
                    type(row.get(k)) is bool
                    for k in ("repeat", "is_composing", "shift", "ctrl", "alt", "meta")
                ),
                "Flag event keyboard tidak valid.",
            )
        require(
            "calibration" not in row
            or row["calibration"] is True
            and row["type"] == "keydown"
            and row["code"] == "Space"
            and row["phase"] in {"sync_start", "sync_end"}
            and not row["repeat"],
            "Event pemeriksaan timing tidak valid.",
        )
        events.append(row)
    return events


def replay_timing(audio, rate, first_frame, events, timing, capture_issues):
    require(
        isinstance(timing, dict) and timing.get("method") == "calibration-affine-energy-v1",
        "Metode timing tidak didukung.",
    )
    require(
        timing.get("first_frame") == first_frame and timing.get("sample_rate") == rate,
        "Origin timeline berbeda.",
    )
    anchors = timing.get("anchors")
    require(isinstance(anchors, list) and len(anchors) < 200_000, "Anchor timing tidak valid.")
    pairs = []
    previous = -math.inf
    for a in anchors:
        require(
            isinstance(a, dict)
            and all(finite(a.get(k)) for k in ("before_ms", "after_ms", "context_s")),
            "Anchor tidak finite.",
        )
        require(
            a["before_ms"] >= 0 and a["after_ms"] >= a["before_ms"] and a["context_s"] >= 0,
            "Rentang anchor tidak valid.",
        )
        mid = (a["before_ms"] + a["after_ms"]) / 2
        require(mid >= previous, "Jam anchor mundur.")
        previous = mid
        if a["after_ms"] - a["before_ms"] <= 5 and (not pairs or mid > pairs[-1][0]):
            pairs.append((mid, a["context_s"] * rate - first_frame))
    points = np.asarray(pairs, dtype=float).reshape(-1, 2)
    size = round(rate * 0.005)
    cut = len(audio) // size * size
    energy = np.sqrt(np.mean(audio[:cut].reshape(-1, size).astype(float) ** 2, axis=1))
    quiet = energy[round(0.5 * rate / size) : round(2.5 * rate / size)]
    floor = float(np.sqrt(np.mean(quiet**2))) if len(quiet) else 0.0

    def frame(time):
        if len(points) < 2 or not points[0, 0] <= time <= points[-1, 0]:
            return None
        return float(np.interp(time, points[:, 0], points[:, 1]))

    markers = []
    for event in events:
        if event.get("calibration") is not True:
            continue
        marker = {"event_seq": event["seq"], "phase": event["phase"]}
        at = frame(event["time_ms"])
        if at is None:
            markers.append({**marker, "reason": "outside_clock"})
            continue
        start = max(0, math.ceil((at - 0.25 * rate) / size))
        end = min(len(energy), math.floor((at + 0.25 * rate) / size))
        best = start + int(np.argmax(energy[start:end])) if end > start else start
        if end <= start or energy[best] <= max(1e-4, 4 * floor):
            markers.append({**marker, "reason": "weak_signal"})
            continue
        indices = np.arange(start, end)
        competing = np.any(
            (np.abs(indices - best) * size > 0.025 * rate)
            & (energy[start:end] >= 0.9 * energy[best])
        )
        markers.append(
            {
                **marker,
                "clock_frame": at,
                "peak_frame": best * size + size / 2,
                "reason": "ambiguous_peak" if competing else None,
            }
        )
    train = [m for i, m in enumerate(markers) if i % 2 == 0 and m["reason"] is None]
    validation = [m for i, m in enumerate(markers) if i % 2 == 1 and m["reason"] is None]
    affine, residuals = None, []
    if len(train) >= 2:
        x, y = np.asarray([(m["clock_frame"], m["peak_frame"]) for m in train]).T
        xx, yy = x - x.mean(), y - y.mean()
        if xx @ xx > 0:
            slope = float(xx @ yy / (xx @ xx))
            affine = {"slope": slope, "intercept": float(y.mean() - slope * x.mean())}
            residuals = [
                (m["peak_frame"] - (affine["intercept"] + affine["slope"] * m["clock_frame"]))
                / rate
                * 1000
                for m in validation
            ]
    issues = list(capture_issues)
    if len(markers) != 16 or any(
        sum(m["phase"] == p for m in markers) != 8 for p in ("sync_start", "sync_end")
    ):
        issues.append("incomplete_calibration")
    if any(m["reason"] for m in markers) or len(train) != 8 or len(validation) != 8:
        issues.append("uncertain_calibration")
    if affine is None or abs(affine["slope"] - 1) > 0.005 or any(abs(r) > 20 for r in residuals):
        issues.append("timing_residual")
    if len(points) < 2 or np.any(np.diff(points[:, 1]) < 0):
        issues.append("non_monotonic_clock")
    if any(
        e["handler_ms"] - e["time_ms"] > 20 or e["handler_ms"] < e["time_ms"] - 1 for e in events
    ):
        issues.append("event_dispatch_lag")
    report = {
        "method": timing["method"],
        "status": "needs_review" if issues else "checked",
        "affine": affine,
        "residuals_ms": residuals,
        "quiet_rms": floor,
        "issues": sorted(set(issues)),
        "calibration": markers,
    }
    same = (
        report["status"] == timing.get("status")
        and sorted(set(issues)) == sorted(timing.get("issues", []))
        and len(markers) == len(timing.get("calibration", []))
    )
    if same:
        for a, b in zip(markers, timing["calibration"], strict=True):
            same &= a["reason"] == b.get("reason") and a["event_seq"] == b.get("event_seq")
            for k in ("clock_frame", "peak_frame"):
                same &= k not in a or finite(b.get(k)) and math.isclose(a[k], b[k], abs_tol=1e-5)
        if affine is not None:
            same &= isinstance(timing.get("affine"), dict) and all(
                finite(timing["affine"].get(k))
                and math.isclose(v, timing["affine"][k], abs_tol=1e-5)
                for k, v in affine.items()
            )
    report["matches_browser_report"] = bool(same)
    if not same:
        report["status"] = "needs_review"
        report["issues"].append("browser_timing_mismatch")
    return report, frame


def letter_event(e, labels=LABELS):
    if "space" in labels:
        return web_quota.event_class(e) in labels
    return (
        e["type"] == "keydown"
        and e["phase"] == "typing"
        and not any(e[k] for k in ("repeat", "is_composing", "ctrl", "alt", "meta"))
        and len(e["key"]) == 1
        and e["key"].lower() in LABELS
    )


def coverage(audio, rate, events, timing, frame, labels=LABELS, selected_sequences=None):
    def label(e):
        return web_quota.event_class(e) if "space" in labels else e["key"].lower()

    raw = Counter(label(e) for e in events if letter_event(e, labels))
    selected = set(selected_sequences) if selected_sequences is not None else None
    candidates, rejected, counts = [], Counter(), Counter()
    if timing["status"] == "checked":
        affine = timing["affine"]

        def mapped(e):
            position = frame(e["time_ms"])
            return None if position is None else affine["intercept"] + affine["slope"] * position

        neighbors = [
            (e["seq"], mapped(e))
            for e in events
            if e["type"] in {"keydown", "keyup"} and not e.get("repeat", False)
        ]
        for e in events:
            if not letter_event(e, labels) or selected is not None and e["seq"] not in selected:
                continue
            at = mapped(e)
            if at is None:
                rejected["outside_clock"] += 1
                continue
            start, end = round(at - 0.015 * rate), round(at + 0.065 * rate)
            if start < 0 or end > len(audio):
                rejected["boundary"] += 1
            elif any(i != e["seq"] and t is not None and start <= t < end for i, t in neighbors):
                rejected["neighbor_press_or_release"] += 1
            elif np.sqrt(np.mean(audio[start:end].astype(float) ** 2)) <= 1e-4:
                rejected["weak_signal"] += 1
            elif np.max(np.abs(audio[start:end])) >= 0.999:
                rejected["clipping"] += 1
            else:
                letter = label(e)
                counts[letter] += 1
                candidates.append({"event_seq": e["seq"], "letter": letter, "frame_estimate": at})
    else:
        rejected["unverified_timing"] = len(selected) if selected is not None else sum(raw.values())
    return {
        "recorded_counts": {k: raw[k] for k in labels},
        "candidate_counts": {k: counts[k] for k in labels},
        **({"candidate_scope": "quota_selected"} if selected is not None else {}),
        "candidate_events": candidates,
        "rejected": dict(rejected),
        "interpretation": (
            "Potongan 80 ms menurut estimasi penanda, tanpa press/release tetangga; "
            "belum sertifikasi onset atau evaluasi classifier."
        ),
    }


def import_session(package: Path, output: Path):
    require(not output.exists(), f"Tujuan sudah ada: {output}. Pilih direktori baru.")
    files = archive_payload(package)
    meta = decode_json(files["session.json"])
    validate_metadata(meta)
    require(
        isinstance(meta.get("files"), dict) and set(meta["files"]) == FILES - {"session.json"},
        "Manifest hash tidak lengkap.",
    )
    import hashlib

    for name, digest in meta["files"].items():
        require(hashlib.sha256(files[name]).hexdigest() == digest, f"Hash berbeda: {name}")
    try:
        info = sf.info(io.BytesIO(files["audio.wav"]))
        require(
            info.channels == 1
            and info.subtype == "FLOAT"
            and info.samplerate == meta["sample_rate"]
            and info.frames == meta["frames"],
            "WAV berbeda dari metadata.",
        )
        audio, rate = sf.read(io.BytesIO(files["audio.wav"]), dtype="float32")
    except (RuntimeError, ValueError) as exc:
        raise ResearchError(f"WAV tidak valid: {exc}") from exc
    require(np.isfinite(audio).all(), "Sampel audio tidak finite.")
    cursor = meta["first_frame"]
    blocks = meta.get("audio_blocks")
    require(isinstance(blocks, list) and bool(blocks), "Daftar blok audio kosong.")
    for i, b in enumerate(blocks):
        require(
            isinstance(b, dict)
            and b.get("seq") == i
            and b.get("first_frame") == cursor
            and type(b.get("frames")) is int
            and b["frames"] > 0,
            "Daftar blok tidak kontinu.",
        )
        cursor += b["frames"]
    require(cursor - meta["first_frame"] == len(audio), "Jumlah sampel blok berbeda.")
    require(
        type(meta.get("event_count")) is int and meta["event_count"] >= 0,
        "Jumlah event tidak valid.",
    )
    events = read_events(files["events.jsonl"], meta["event_count"])
    collection = web_quota.selection(events) if meta["schema"] == web_quota.SCHEMA else None
    if collection is not None:
        web_quota.validate_stored_check(meta["preflight"], events)
        require(meta.get("collection") == collection, "Manifest kuota berbeda dari event aktual.")
        require(
            meta["status"] != "complete" or collection["complete"],
            "Sesi lengkap belum memenuhi kuota.",
        )
        if "additional_keys" in meta:
            require(
                meta["additional_keys"] == web_quota.additional_counts(events),
                "Hitungan tombol tambahan berbeda dari event aktual.",
            )
    timing, frame = replay_timing(
        audio,
        rate,
        meta["first_frame"],
        events,
        decode_json(files["timing.json"]),
        meta["capture_issues"],
    )
    cover = coverage(
        audio,
        rate,
        events,
        timing,
        frame,
        web_quota.CLASSES if collection is not None else LABELS,
        collection["selected_sequences"] if collection is not None else None,
    )
    require(
        meta.get("recorded_counts") == cover["recorded_counts"],
        "Hitungan huruf berbeda dari event aktual.",
    )
    clipped = int(np.count_nonzero(np.abs(audio) >= 0.999))
    report = {
        "schema": "ketik-web-import/v2" if collection is not None else "ketik-web-import/v1",
        "created_at": now(),
        "source_sha256": file_hash(package),
        "session_id": meta["session_id"],
        "participant_id": meta["participant_id"],
        "study_id": meta["study_id"],
        "status": "checked"
        if meta["status"] == "complete" and timing["status"] == "checked"
        else "needs_review",
        "capture_status": meta["status"],
        "synthetic": meta["synthetic"],
        "role": meta["role"],
        "local_day": meta["local_day"],
        "duration_seconds": len(audio) / rate,
        "timing": timing,
        "coverage": cover,
        **(
            {
                "collection": collection,
                "preflight": {
                    "method": meta["preflight"]["method"],
                    "status": meta["preflight"]["status"],
                    "evidence": (
                        "browser-reported; waveform included at the start of audio.wav"
                        if meta["preflight"].get("source") == "session-sync-start"
                        else "browser-reported; preflight waveform is ephemeral and not in this package"
                    ),
                },
                **(
                    {"additional_keys": meta["additional_keys"]}
                    if "additional_keys" in meta else {}
                ),
            }
            if collection is not None
            else {}
        ),
        "quality": {
            "peak_abs": float(np.max(np.abs(audio))),
            "clipped_samples": clipped,
            "clipped_fraction": clipped / len(audio),
        },
        "physical_identity_and_calendar": "participant-reported; not independently verified",
        "compatible_with_legacy_terminal_dataset": False,
        "classifier_fit_performed": False,
    }
    fresh_directory(output)
    for name, data in files.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
    write_json(output / "review.json", report)
    return report
