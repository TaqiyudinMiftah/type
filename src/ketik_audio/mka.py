"""MKA corpus provenance, exact duplicate checks, and excerpt tracing on CPU."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import PurePosixPath

import numpy as np
import soundfile as sf
from scipy.signal import correlate

from .common import ResearchError, now, write_json

PLATFORMS = ("hp", "Lenovo", "MSI", "Mac", "Messenger", "Zoom")
KEYS = "1234567890qwertyuiopasdfghjklzxcvbnm"
ROLES = {
    "Raw data": "raw",
    "Raw Data": "raw",
    "Sound Segment(wav)": "segment",
    "Sound Segment Metadata(.txt)": "metadata",
    "Sound Segment(.matrix)": "feature",
}
MATCH_CORRELATION = 0.999


def parse_member(name):
    parts = PurePosixPath(name).parts
    if ".." in parts or PurePosixPath(name).is_absolute():
        raise ResearchError(f"Path anggota ZIP tidak valid: {name}")
    if len(parts) < 4 or parts[0] != "MKA datasets":
        return {"scope": "other", "platform": None, "role": "other", "label": None}
    platform, section = parts[1:3]
    scope = (
        "canonical"
        if platform in PLATFORMS
        else "aggregate"
        if platform == "All Dataset"
        else "other"
    )
    role = ROLES.get(section, "other")
    label = (
        PurePosixPath(parts[3]).stem
        if role == "raw" and scope == "canonical"
        else parts[3]
        if len(parts) >= 5
        else None
    )
    return {"scope": scope, "platform": platform, "role": role, "label": label}


def pcm_digest(audio, rate):
    """Keep rate, channel layout and all decoded precision in the fingerprint."""
    values = np.asarray(audio, dtype="<f8", order="C")
    digest = hashlib.sha256(json.dumps([int(rate), *values.shape]).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def audio_details(blob):
    with sf.SoundFile(io.BytesIO(blob)) as source:
        rate, frames, channels = source.samplerate, source.frames, source.channels
        subtype, format_name = source.subtype, source.format
        values = source.read(dtype="float64", always_2d=True)
    if values.shape != (frames, channels):
        raise ResearchError("Jumlah frame decode berbeda dari header.")
    finite = bool(np.isfinite(values).all())
    peak = float(np.abs(values).max(initial=0)) if finite else None
    return {
        "sample_rate": rate,
        "frames": frames,
        "channels": channels,
        "duration_seconds": frames / rate,
        "subtype": subtype,
        "format": format_name,
        "all_finite": finite,
        "peak": peak,
        "rms": float(np.sqrt(np.mean(values**2))) if finite and frames else None,
        "samples_at_or_above_one": int((np.abs(values) >= 1).sum()) if finite else None,
        "usable_waveform": finite and frames > 0 and peak > 0,
        "pcm_sha256": pcm_digest(values, rate),
    }


def text_profile(blob):
    text = blob.decode("utf-8-sig", errors="replace")
    object_class = re.search(r'Object class\s*=\s*"([^"]+)"', text)
    if object_class:
        kind = "praat_object"
    elif text.strip() and set(text) <= set("0123456789eE+-. \t\r\n,;[]"):
        kind = "numeric_table"
    elif not text.strip():
        kind = "empty"
    else:
        kind = "other_text"
    words = ("participant", "session", "microphone", "recorder", "device", "timestamp", "keyboard")
    return {
        "kind": kind,
        "object_class": object_class[1] if object_class else None,
        "context_keywords": [w for w in words if re.search(rf"\b{w}\b", text, re.IGNORECASE)],
        "preview": text[:220],
    }


def scan_archive(archive, output):
    rows, decoded, total_bytes, folders = [], {}, 0, 0
    with zipfile.ZipFile(archive) as source:
        infos = source.infolist()
        if len({i.filename for i in infos}) != len(infos):
            raise ResearchError("ZIP memiliki nama anggota berulang.")
        for number, info in enumerate(infos, 1):
            if info.is_dir():
                folders += 1
                continue
            blob = source.read(info)  # zipfile validates CRC as each complete member is read.
            if len(blob) != info.file_size:
                raise ResearchError("Ukuran anggota ZIP berbeda.")
            total_bytes += len(blob)
            digest = hashlib.sha256(blob).hexdigest()
            row = {
                "path": info.filename,
                "bytes": len(blob),
                "compressed_bytes": info.compress_size,
                "crc32": f"{info.CRC:08x}",
                "sha256": digest,
                **parse_member(info.filename),
            }
            looks_audio = blob[:4] in (b"RIFF", b"RF64", b"fLaC", b"FORM")
            if looks_audio or PurePosixPath(info.filename).suffix.lower() == ".wav":
                if digest not in decoded:
                    try:
                        decoded[digest] = {"audio": audio_details(blob)}
                    except (RuntimeError, ValueError, ResearchError) as exc:
                        decoded[digest] = {"audio_error": str(exc)}
                row.update(decoded[digest])
            elif row["role"] in ("metadata", "feature"):
                row["text"] = text_profile(blob)
            rows.append(row)
            if number % 500 == 0:
                print(
                    f"Inventory {number}/{len(infos)} entries, {total_bytes / 1e9:.2f} GB decoded",
                    flush=True,
                )
    result = {
        "created_at": now(),
        "zip_entries": len(infos),
        "directories": folders,
        "file_count": len(rows),
        "uncompressed_bytes_read": total_bytes,
        "all_member_crcs_verified": True,
        "files": rows,
    }
    write_json(output / "inventory.json", result)
    return result


def duplicate_groups(rows, field):
    groups = defaultdict(list)
    for row in rows:
        value = row.get("audio", {}).get(field) if field == "pcm_sha256" else row.get(field)
        if value:
            groups[value].append(row["path"])
    return [{field: key, "paths": paths} for key, paths in groups.items() if len(paths) > 1]


def count_budget(rows, keys=KEYS):
    counts = {}
    for platform in PLATFORMS:
        per_key = {}
        for key in keys:
            per_key[key] = len(
                {
                    row["audio"]["pcm_sha256"]
                    for row in rows
                    if row["scope"] == "canonical"
                    and row["platform"] == platform
                    and row["role"] == "segment"
                    and row["label"] == key
                    and row.get("audio", {}).get("usable_waveform")
                }
            )
        counts[platform] = {
            "unique_supplied_clips_by_key": per_key,
            "minimum_unique_supplied_clips_per_key": min(per_key.values()),
            "count_only_feasible_K": {
                str(test): [k for k in (1, 2, 3, 5, 10) if min(per_key.values()) >= k + test]
                for test in (1, 5, 10)
            },
        }
    return counts


def locate_excerpt(raw, clip):
    """Find an affine/inverted excerpt using zero-mean normalized correlation."""
    raw, clip = np.asarray(raw, dtype=np.float64), np.asarray(clip, dtype=np.float64)
    if raw.ndim != 1 or clip.ndim != 1 or len(clip) < 2 or len(raw) < len(clip):
        raise ResearchError("Dimensi/panjang waveform tidak mendukung penelusuran.")
    if not np.isfinite(raw).all() or not np.isfinite(clip).all():
        raise ResearchError("Waveform penelusuran non-finite.")
    centered = clip - clip.mean()
    clip_energy = float(centered @ centered)
    if clip_energy <= 0:
        raise ResearchError("Potongan konstan tidak memiliki lokasi yang teridentifikasi.")
    length = len(clip)
    sums = np.concatenate(([0.0], np.cumsum(raw, dtype=np.float64)))
    squares = np.concatenate(([0.0], np.cumsum(raw * raw, dtype=np.float64)))
    window_sum = sums[length:] - sums[:-length]
    energy = np.maximum(squares[length:] - squares[:-length] - window_sum**2 / length, 0)
    numerator = correlate(raw, centered, mode="valid", method="fft")
    denominator = np.sqrt(energy * clip_energy)
    energy_floor = np.finfo(np.float64).eps * max(float(squares[-1]), 1) * 16
    scores = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=energy > energy_floor,
    )
    lag = int(np.abs(scores).argmax())
    patch = raw[lag : lag + length]
    x = patch - patch.mean()
    if float(x @ x) <= 0:
        raise ResearchError("Tidak menemukan bagian sumber yang bervariasi.")
    gain = float((x @ centered) / (x @ x))
    offset = float(clip.mean() - gain * patch.mean())
    residual = float(np.sqrt(np.mean((clip - (gain * patch + offset)) ** 2)))
    correlation = float(np.corrcoef(patch, clip)[0, 1])
    return {
        "start_frame": lag,
        "stop_frame": lag + length,
        "correlation": correlation,
        "affine_gain": gain,
        "affine_offset": offset,
        "normalized_residual_rms": residual / float(clip.std()),
        "matched": abs(correlation) >= MATCH_CORRELATION,
    }


def locate_multichannel_excerpt(raw, clip):
    clip_channel = int(np.var(clip, axis=0).argmax())
    raw_channel = (
        clip_channel if raw.shape[1] == clip.shape[1] else int(np.var(raw, axis=0).argmax())
    )
    match = locate_excerpt(raw[:, raw_channel], clip[:, clip_channel])
    start, stop = match["start_frame"], match["stop_frame"]
    return {
        **match,
        "raw_channel": raw_channel,
        "clip_channel": clip_channel,
        "all_channels_pcm_equal": np.array_equal(raw[start:stop], clip),
    }


def trace_segments(archive, rows, output):
    raw_index = defaultdict(list)
    for row in rows:
        if row["scope"] == "canonical" and row["role"] == "raw":
            raw_index[row["platform"], row["label"]].append(row)
    traces = []
    with zipfile.ZipFile(archive) as source:
        for platform in PLATFORMS:
            for key in KEYS:
                parents = raw_index[platform, key]
                if len(parents) != 1:
                    raise ResearchError(
                        "Penelusuran memerlukan satu induk nominal yang tidak ambigu."
                    )
                parent = parents[0]
                raw, rate = sf.read(io.BytesIO(source.read(parent["path"])), always_2d=True)
                for row in rows:
                    if not (
                        row["scope"] == "canonical"
                        and row["platform"] == platform
                        and row["role"] == "segment"
                        and row["label"] == key
                        and "audio" in row
                    ):
                        continue
                    base = {
                        "platform": platform,
                        "key": key,
                        "path": row["path"],
                        "parent": parent["path"],
                    }
                    clip, clip_rate = sf.read(io.BytesIO(source.read(row["path"])), always_2d=True)
                    if clip_rate != rate:
                        traces.append({**base, "matched": False, "reason": "sample_rate_mismatch"})
                        continue
                    try:
                        match = locate_multichannel_excerpt(raw, clip)
                    except ResearchError as exc:
                        traces.append({**base, "matched": False, "reason": str(exc)})
                        continue
                    start, stop = match["start_frame"], match["stop_frame"]
                    traces.append(
                        {
                            **base,
                            **match,
                            "sample_rate": rate,
                            "start_seconds": start / rate,
                            "stop_seconds": stop / rate,
                        }
                    )
            print(f"Traced 36 classes: {platform}", flush=True)
    overlaps = []
    grouped = defaultdict(list)
    for trace in traces:
        if trace["matched"]:
            grouped[trace["parent"]].append(trace)
    for parent, excerpts in grouped.items():
        ordered = sorted(excerpts, key=lambda r: r["start_frame"])
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                overlap = min(left["stop_frame"], right["stop_frame"]) - right["start_frame"]
                if overlap > 0:
                    overlaps.append(
                        {
                            "parent": parent,
                            "left": left["path"],
                            "right": right["path"],
                            "overlap_frames": overlap,
                        }
                    )
    result = {
        "threshold_absolute_correlation": MATCH_CORRELATION,
        "traces": traces,
        "matched": sum(t["matched"] for t in traces),
        "all_channels_exact": sum(t.get("all_channels_pcm_equal", False) for t in traces),
        "overlapping_pairs": overlaps,
    }
    write_json(output / "excerpt-provenance.json", result)
    return result


def compare_harrison(rows, reference, output):
    by_bytes, by_pcm = defaultdict(list), defaultdict(list)
    for row in rows:
        if "audio" in row:
            by_bytes[row["sha256"]].append(row["path"])
            by_pcm[row["audio"]["pcm_sha256"]].append(row["path"])
    verified, matches = [], []
    for path in sorted(reference.rglob("*.wav")):
        blob = path.read_bytes()
        digest, audio = hashlib.sha256(blob).hexdigest(), audio_details(blob)
        verified.append({"path": str(path), "sha256": digest, "pcm_sha256": audio["pcm_sha256"]})
        if by_bytes.get(digest) or by_pcm.get(audio["pcm_sha256"]):
            matches.append(
                {
                    "harrison": str(path),
                    "byte_matches": by_bytes.get(digest, []),
                    "pcm_matches": by_pcm.get(audio["pcm_sha256"], []),
                }
            )
    if not verified:
        raise ResearchError("Tidak ada WAV Harrison pembanding.")
    result = {
        "reference_files": verified,
        "exact_matches": matches,
        "transformed_copy_search_performed": False,
    }
    write_json(output / "harrison-comparison.json", result)
    return result


def summarise_inventory(rows):
    result = {"platforms": {}, "clip_budgets": count_budget(rows)}
    for platform in PLATFORMS:
        selected = [r for r in rows if r["scope"] == "canonical" and r["platform"] == platform]
        raw = [r for r in selected if r["role"] == "raw" and "audio" in r]
        segments = [r for r in selected if r["role"] == "segment" and "audio" in r]
        raw_counts = Counter(r["label"] for r in raw if r["label"] in KEYS)
        metadata = [r for r in selected if r["role"] == "metadata"]
        result["platforms"][platform] = {
            "raw_files": len(raw),
            "raw_classes": len({r["label"] for r in raw}),
            "common_36_keys_present": [k for k in KEYS if raw_counts[k]],
            "raw_files_per_common_key": {k: raw_counts[k] for k in KEYS},
            "provided_segment_files": len(segments),
            "raw_rates_channels_subtypes": dict(
                Counter(
                    f"{r['audio']['sample_rate']}Hz/{r['audio']['channels']}ch/{r['audio']['subtype']}"
                    for r in raw
                )
            ),
            "segment_rates_channels_subtypes": dict(
                Counter(
                    f"{r['audio']['sample_rate']}Hz/{r['audio']['channels']}ch/{r['audio']['subtype']}"
                    for r in segments
                )
            ),
            "raw_duration_seconds_range": [
                min(r["audio"]["duration_seconds"] for r in raw),
                max(r["audio"]["duration_seconds"] for r in raw),
            ],
            "segment_duration_seconds_range": [
                min(r["audio"]["duration_seconds"] for r in segments),
                max(r["audio"]["duration_seconds"] for r in segments),
            ],
            "metadata_files": len(metadata),
            "metadata_types": dict(
                Counter(r.get("text", {}).get("kind", "not_text") for r in metadata)
            ),
            "metadata_context_keyword_hits": [
                r["path"] for r in metadata if r.get("text", {}).get("context_keywords")
            ],
        }
    result["audio_decode_errors"] = [
        {"path": r["path"], "error": r["audio_error"]} for r in rows if "audio_error" in r
    ]
    result["silent_or_nonfinite_audio"] = [
        r["path"] for r in rows if "audio" in r and not r["audio"]["usable_waveform"]
    ]
    result["file_hash_duplicate_groups"] = duplicate_groups(rows, "sha256")
    result["pcm_duplicate_groups"] = duplicate_groups(rows, "pcm_sha256")
    return result
