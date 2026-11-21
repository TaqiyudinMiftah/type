"""Fixed collection, clock and paired crop rules for a two-microphone study."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime
from math import gcd
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .common import LABELS, ResearchError, file_hash, fresh_directory, now, read_json, write_json
from .public_recordings import MIN_RMS, center_crop, rms
from .recording import make_prompts
from .skaid import pcm_fingerprint

SCHEMA = "ketik-paired-mic-v1"
MICS = ("a", "b")
SESSIONS = ("pilot", "calibration", "test1", "test2")
PER_KEY = {"pilot": 2, "calibration": 16, "test1": 12, "test2": 12}
PILOT_KEYS = tuple("adhtu")
FILES = {"mic-a.wav", "mic-b.wav", "clock-a.npz", "clock-b.npz", "events.json"}
START_STEPS = ("calibration", "test1", "test2")
MIN_GAP = 0.8


def setup(root, participant, keyboard, mic_a, mic_b, geometry, timezone, synthetic=False):
    names = [v.strip() for v in (participant, keyboard, mic_a, mic_b, geometry)]
    if any(not n or len(n) > 500 for n in names) or names[2].casefold() == names[3].casefold():
        raise ResearchError("Isi alias peserta, setup, dan dua label mikrofon berbeda.")
    validate_timezone(timezone)
    fresh_directory(root)
    record = {
        "schema": SCHEMA,
        "created_at": now(),
        "participant": names[0],
        "keyboard": names[1],
        "microphones": {"a": names[2], "b": names[3]},
        "geometry_and_channel": names[4],
        "timezone": timezone,
        "synthetic": bool(synthetic),
    }
    write_json(root / "study.json", record)
    return record


def validate_timezone(name):
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ResearchError(f"Zona waktu tidak dikenal: {name}.") from exc


def study(root):
    value = read_json(root / "study.json")
    if value.get("schema") != SCHEMA or set(value.get("microphones", {})) != set(MICS):
        raise ResearchError("Metadata studi dua mikrofon tidak valid.")
    return value


def local_day(iso, zone):
    timestamp = datetime.fromisoformat(iso)
    if timestamp.utcoffset() is None:
        raise ResearchError("Timestamp sesi harus menyimpan zona waktu.")
    return timestamp.astimezone(validate_timezone(zone)).date().isoformat()


def vocabulary(session):
    return PILOT_KEYS if session == "pilot" else LABELS


def prompts(session):
    keys = vocabulary(session)
    return [
        c for c in make_prompts(PER_KEY[session], 20261115 + SESSIONS.index(session)) if c in keys
    ]


def session_path(root, name):
    if name not in SESSIONS:
        raise ResearchError("Nama sesi tidak dikenal.")
    return root / "raw" / name


def verify_session(root, name):
    path = session_path(root, name)
    meta = read_json(path / "session.json")
    s = study(root)
    if (
        meta.get("status") != "complete"
        or meta.get("schema") != SCHEMA
        or meta.get("session") != name
        or meta.get("study_sha256") != file_hash(root / "study.json")
        or meta.get("synthetic") != s["synthetic"]
        or meta.get("day") != local_day(meta["started_at"], s["timezone"])
        or meta["finished_at"] <= meta["started_at"]
        or meta.get("accepted") != len(prompts(name))
        or meta.get("capture_issues")
        or meta.get("mode") not in ("shared", "dual")
        or set(meta.get("files", {})) != FILES
        or set(meta.get("channels", {})) != set(MICS)
    ):
        raise ResearchError(
            f"Sesi {name} tidak lengkap, tidak valid, atau berasal dari studi lain."
        )
    for f, expected in meta["files"].items():
        if file_hash(path / f) != expected:
            raise ResearchError(f"File sesi {name} berubah: {f}.")
    return meta


def compare_sessions(earlier, later):
    if earlier["day"] >= later["day"] or earlier["finished_at"] >= later["started_at"]:
        raise ResearchError(
            "Kalibrasi, test1, dan test2 harus hari lokal berbeda secara berurutan."
        )
    if earlier["mode"] != later["mode"]:
        raise ResearchError("Mode clock berubah antarsesi.")
    for mic in MICS:
        for attr in ("device_name", "host_api", "input_channel", "sample_rate"):
            if earlier["channels"][mic][attr] != later["channels"][mic][attr]:
                raise ResearchError(f"Rute mikrofon {mic.upper()} berubah antarsesi: {attr}.")
    hashes = {earlier["files"][f"mic-{m}.wav"] for m in MICS}
    if hashes & {later["files"][f"mic-{m}.wav"] for m in MICS}:
        raise ResearchError("Audio salinan ditemukan lintas hari.")


def write_session(path, metadata, events):
    write_json(path / "events.json", {"events": events})
    files = {f: file_hash(path / f) for f in sorted(FILES) if (path / f).is_file()}
    if metadata["status"] == "complete" and set(files) != FILES:
        raise ResearchError("Sesi lengkap tidak memiliki seluruh file audio/clock.")
    write_json(path / "session.json", {**metadata, "files": files})


def validate_clock(frame, adc, total, rate):
    if (
        frame.ndim != 1
        or adc.shape != frame.shape
        or len(frame) < 2
        or frame[0] != 0
        or not np.issubdtype(frame.dtype, np.integer)
        or not np.isfinite(adc).all()
        or np.any(np.diff(frame) <= 0)
        or np.any(np.diff(adc) <= 0)
        or frame[-1] >= total
    ):
        raise ResearchError("Anchor ADC kosong, clock tidak naik, atau saat rekaman tidak valid.")
    jitter = float(np.max(np.abs(np.diff(adc) - np.diff(frame) / rate)))
    if jitter > 0.010:
        raise ResearchError(f"Lompatan timing ADC {1000 * jitter:.2f} ms melebihi 10 ms.")
    measured_rate = float((frame[-1] - frame[0]) / (adc[-1] - adc[0]))
    if abs(measured_rate / rate - 1) > 0.01:
        raise ResearchError("Rentang clock ADC berbeda lebih dari 1% dari sample rate.")
    return {
        "anchors": len(frame),
        "max_block_jitter_ms": jitter * 1000,
        "effective_rate": measured_rate,
    }


def clock_index(value, frame, adc):
    if not math.isfinite(value) or not adc[0] <= value <= adc[-1]:
        return None
    return round(float(np.interp(value, adc, frame)))


def map_reading(event_ns, before_ns, after_ns, stream_time):
    midpoint = (before_ns + after_ns) / 2
    return {
        "adc_time": stream_time - (midpoint - event_ns) / 1e9,
        "span_ms": (after_ns - before_ns) / 1e6,
        "lag_ms": (midpoint - event_ns) / 1e6,
    }


def resample80(clip, rate):
    divisor = gcd(16000, rate)
    output = resample_poly(
        clip.astype(np.float32),
        16000 // divisor,
        rate // divisor,
        window=("kaiser", 5.0),
        padtype="constant",
    )
    if output.shape != (1280,) or not np.isfinite(output).all():
        raise ResearchError("Crop 80 ms tidak menjadi 1.280 sampel finite.")
    return output


def press_crop(audio, rate, marker, floor):
    start, end = marker - round(rate * 0.100), marker + round(rate * 0.050)
    n = round(0.005 * rate)
    if start < 0 or end > len(audio):
        return None, "marker_boundary", None
    count = (end - start) // n
    energy = np.mean(audio[start : start + count * n].reshape(count, n).astype(float) ** 2, axis=1)
    best = int(np.argmax(energy))
    center = int(start + best * n + n // 2)
    clip = center_crop(audio, rate, center)
    peak_offset = (center - marker) / rate * 1000
    if clip is None or len(clip) != round(0.08 * rate):
        return None, "crop_boundary", peak_offset
    if rms(clip) <= MIN_RMS or np.sqrt(energy[best]) < 4 * floor:
        return None, "weak_signal", peak_offset
    if np.max(np.abs(clip)) >= 0.999:
        return None, "clipping", peak_offset
    return clip, None, peak_offset


def load_channel(path, mic, metadata):
    audio, rate = sf.read(path / f"mic-{mic}.wav", dtype="float32")
    conf = metadata["channels"][mic]
    if (
        audio.ndim != 1
        or rate != conf["sample_rate"]
        or len(audio) != conf["frames"]
        or not np.isfinite(audio).all()
        or rate < 16000
        or rate > 192000
        or rate % 100
        or len(audio) < 3 * rate
        or sf.info(path / f"mic-{mic}.wav").subtype != "FLOAT"
    ):
        raise ResearchError(f"Formato WAV/percorso mikrofon {mic.upper()} tidak cocok.")
    with np.load(path / f"clock-{mic}.npz", allow_pickle=False) as z:
        frame, adc = z["frame"], z["adc"]
    quality = validate_clock(frame, adc, len(audio), rate)
    quality["quiet_rms"] = rms(audio[rate // 2 : 5 * rate // 2])
    quality["pcm_sha256"] = pcm_fingerprint(audio[:, None], rate)
    return audio, rate, frame, adc, quality


def check_events(events, name):
    expected = prompts(name)
    current = 0
    elapsed = np.array([e["elapsed"] for e in events])
    if (
        not len(elapsed)
        or not np.isfinite(elapsed).all()
        or np.any(elapsed < 0)
        or np.any(np.diff(elapsed) <= 0)
    ):
        raise ResearchError("Waktu event terminal tidak berurutan.")
    intervals, paused, previous = [], None, -float("inf")
    for e in events:
        when, kind = e["elapsed"], e["kind"]
        if kind == "pause" and paused is None:
            paused = when
        elif kind == "resume" and paused is not None:
            intervals.append((paused - 0.1, when + 0.35))
            paused = None
        elif kind == "letter" and paused is None:
            if (
                current >= len(expected)
                or e["prompt"] != expected[current]
                or e["key"] not in vocabulary(name)
            ):
                raise ResearchError("Log label berbeda dari urutan prompt terkunci.")
            accepted = e["key"] == e["prompt"] and when - previous >= MIN_GAP
            if e["accepted"] != accepted or set(e.get("clocks", {})) != set(MICS):
                raise ResearchError("Penerimaan label tidak cocok dengan aturan.")
            current += int(accepted)
        elif kind != "other" or paused is not None:
            raise ResearchError("Urutan event kontrol tidak valid.")
        previous = when
    if current != len(expected) or paused is not None:
        raise ResearchError("Log tidak melengkapi sesi.")
    return elapsed, intervals


def extract_session(root, name, metadata=None):
    meta = metadata or verify_session(root, name)
    path = session_path(root, name)
    channels = {m: load_channel(path, m, meta) for m in MICS}
    quality = {m: ch[-1] for m, ch in channels.items()}
    if quality["a"]["pcm_sha256"] == quality["b"]["pcm_sha256"]:
        raise ResearchError("Kedua rute audio berisi PCM identik; periksa sambungan mikrofon.")
    events = read_json(path / "events.json")["events"]
    times, intervals = check_events(events, name)
    records, examples, issues = [], [], Counter()
    for i, e in enumerate(events):
        if e["kind"] != "letter" or not e["accepted"]:
            issues["not_an_accepted_prompt"] += 1
            continue
        start, end = times[i] - 0.115, times[i] + 0.115
        neighbors = np.searchsorted(times, [start, end], side="left")
        if neighbors[1] - neighbors[0] != 1 or any(a < end and b > start for a, b in intervals):
            issues["control_or_neighbor"] += 1
            continue
        views, offsets, frames = [], [], []
        valid = True
        for mic in MICS:
            a, rate, frame, adc, q = channels[mic]
            clock = e["clocks"][mic]
            timing = clock_index(clock["adc_time"], frame, adc)
            if timing is None or not 0 <= clock["span_ms"] <= 5 or not 0 <= clock["lag_ms"] <= 20:
                issues[f"{mic}_clock"] += 1
                valid = False
                continue
            wave, error, offset = press_crop(a, rate, timing, q["quiet_rms"])
            if error:
                issues[f"{mic}_{error}"] += 1
                valid = False
                continue
            views.append(resample80(wave, rate))
            offsets.append(offset)
            frames.append(timing)
        if not valid:
            continue
        if abs(offsets[0] - offsets[1]) > 50:
            issues["peak_pair_offset"] += 1
            continue
        records.append(
            {
                "id": f"{name}:{i:04d}",
                "session": name,
                "letter": e["key"],
                "marker_frames": frames,
                "peak_offsets_ms": offsets,
                "pcm_sha256": [pcm_fingerprint(v[:, None], 16000) for v in views],
            }
        )
        examples.append(views)
    if not records:
        raise ResearchError(f"Tidak ada klip berpasangan lolos pada {name}.")
    identical = sum(r["pcm_sha256"][0] == r["pcm_sha256"][1] for r in records)
    if identical:
        raise ResearchError(f"Terdapat {identical} crop A/B identik; periksa rute mikrofon.")
    counts = Counter(r["letter"] for r in records)
    result = {
        "session": name,
        "day": meta["day"],
        "audio_quality": quality,
        "events": len(events),
        "accepted_prompts": meta["accepted"],
        "eligible_pairs": len(records),
        "eligible_per_letter": {c: counts[c] for c in vocabulary(name)},
        "issues": dict(issues),
    }
    return records, np.asarray(examples, dtype=np.float32), result


def require_coverage(result, minimum, keys):
    short = {
        c: result["eligible_per_letter"].get(c, 0)
        for c in keys
        if result["eligible_per_letter"].get(c, 0) < minimum
    }
    if short:
        raise ResearchError(f"Coverage {result['session']} <{minimum}/huruf: {short}.")
