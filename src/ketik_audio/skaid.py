"""SKAID event-clock diagnostics and native audio inspection without label inference."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import subprocess
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath

import numpy as np

from .common import ResearchError

FIELDS = {"Session ID", "Timestamp (ms)", "Event", "Key", "Phase"}
PHASES = ("Phase 1", "Phase 2")


def session_token(value):
    match = re.fullmatch(r"qt_(?:session_)?(\d{13})", value)
    return f"qt_{match[1]}" if match else None


def member_tokens(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ResearchError("Path ZIP tidak valid.")
    return sorted({f"qt_{token}" for token in re.findall(r"qt_(?:session_)?(\d{13})(?!\d)", name)})


def recording_identity(name):
    member_tokens(name)  # Reject unsafe paths even though no archive path is extracted.
    found = re.fullmatch(
        r"Participant Recordings/(qt_\d{13})/(qt_\d{13})_(email_1|free_form)\.(m4a|wav)", name
    )
    if not found:
        return None
    if found[1] != found[2]:
        raise ResearchError("ID folder dan nama rekaman berbeda.")
    return {
        "session_token": found[1],
        "phase": "Phase 1" if found[3] == "email_1" else "Phase 2",
        "task": found[3],
        "extension": found[4],
    }


def integer_timestamp(value):
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ResearchError("Timestamp tidak numerik.") from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise ResearchError("Timestamp harus milidetik integer finite.")
    return int(number), int(number.as_tuple().exponent)


def physical_key(value):
    return value.lower() if len(value) == 1 and value.isascii() and value.isalpha() else value


def audit_log(blob, name):
    reader = csv.DictReader(io.StringIO(blob.decode("utf-8-sig")))
    if set(reader.fieldnames or ()) != FIELDS:
        raise ResearchError(f"Kolom log tidak sesuai: {name}")
    events, sessions, active, repeats, unmatched, matches = [], set(), {}, [], [], []
    seen, duplicate_rows, exponents = set(), [], Counter()
    for row_number, row in enumerate(reader, start=2):
        stamp, exponent = integer_timestamp(row["Timestamp (ms)"])
        if row["Event"] not in ("press", "release") or row["Phase"] not in PHASES:
            raise ResearchError(f"Event/fase tidak dikenal pada {name}:{row_number}")
        sessions.add(row["Session ID"])
        exponents[str(exponent)] += 1
        key = physical_key(row["Key"])
        event = {
            "row": row_number,
            "timestamp_ms": stamp,
            "phase": row["Phase"],
            "event": row["Event"],
            "raw_key": row["Key"],
            "physical_key": key,
        }
        identity = (stamp, row["Event"], row["Key"], row["Phase"])
        if identity in seen:
            duplicate_rows.append(row_number)
        seen.add(identity)
        if row["Event"] == "press":
            if key in active:
                repeats.append({"row": row_number, "initial_press_row": active[key]["row"]})
            else:
                active[key] = event
        elif key in active:
            press = active.pop(key)
            matches.append(
                {
                    "press_row": press["row"],
                    "release_row": row_number,
                    "duration_ms": stamp - press["timestamp_ms"],
                    "cross_phase": press["phase"] != row["Phase"],
                }
            )
        else:
            unmatched.append(row_number)
        events.append(event)
    if not events:
        raise ResearchError(f"Log kosong: {name}")
    stamps = np.asarray([e["timestamp_ms"] for e in events], dtype=np.int64)
    unique = np.unique(stamps)
    gaps = np.diff(unique)
    phases = {}
    for phase in PHASES:
        subset = [e for e in events if e["phase"] == phase]
        presses = [e for e in subset if e["event"] == "press"]
        times = [e["timestamp_ms"] for e in subset]
        counts = Counter(e["physical_key"] for e in presses)
        phases[phase] = {
            "events": len(subset),
            "presses": len(presses),
            "releases": len(subset) - len(presses),
            "min_timestamp_ms": min(times) if times else None,
            "max_timestamp_ms": max(times) if times else None,
            "span_ms": max(times) - min(times) if times else 0,
            "unique_timestamps": len(set(times)),
            "press_counts": dict(sorted(counts.items())),
        }
    tokens = sorted({token for value in sessions if (token := session_token(value))})
    return {
        "file": name,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "events": len(events),
        "session_ids": sorted(sessions),
        "session_tokens": tokens,
        "filename_tokens": member_tokens(name),
        "timestamp_unique_count": len(unique),
        "timestamp_span_ms": int(unique[-1] - unique[0]),
        "backwards_timestamp_rows": [
            events[i + 1]["row"] for i in np.flatnonzero(np.diff(stamps) < 0)
        ],
        "timestamp_decimal_exponents": dict(exponents),
        "minimum_positive_timestamp_step_ms": int(gaps.min()) if len(gaps) else None,
        "positive_timestamp_step_gcd_ms": int(np.gcd.reduce(gaps)) if len(gaps) else None,
        "duplicate_event_tuple_rows": duplicate_rows,
        "repeated_press_while_held": repeats,
        "unmatched_release_rows": unmatched,
        "unclosed_initial_press_rows": sorted(e["row"] for e in active.values()),
        "matched_initial_press_release": len(matches),
        "nonpositive_hold_durations": sum(m["duration_ms"] <= 0 for m in matches),
        "cross_phase_hold_pairs": sum(m["cross_phase"] for m in matches),
        "phases": phases,
    }, events


def pcm_fingerprint(audio, rate):
    values = np.asarray(audio, dtype="<f4", order="C")
    digest = hashlib.sha256(json.dumps([int(rate), *values.shape]).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def envelopes(audio, rate):
    if audio.ndim != 2 or len(audio) == 0 or rate < 100 or not np.isfinite(audio).all():
        raise ResearchError("Waveform/rate envelope tidak valid.")
    count = (len(audio) * 100 + rate - 1) // rate
    edges = np.minimum(np.arange(count + 1, dtype=np.int64) * rate // 100, len(audio))
    starts, lengths = edges[:-1], np.diff(edges)
    power = np.mean(audio.astype(np.float64) ** 2, axis=1)
    rms = np.sqrt(np.add.reduceat(power, starts) / lengths).astype(np.float32)
    peak = np.maximum.reduceat(np.max(np.abs(audio), axis=1), starts).astype(np.float32)
    return edges, rms, peak


def probe(path):
    process = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        timeout=60,
    )
    if process.returncode:
        raise ResearchError(process.stderr.decode(errors="replace")[-1500:])
    result = json.loads(process.stdout)
    audio = [s for s in result.get("streams", []) if s.get("codec_type") == "audio"]
    if len(audio) != 1:
        raise ResearchError(
            "Audit decoder memerlukan tepat satu stream audio; simpan probe untuk inspeksi."
        )
    return result, audio[0]


def decode(path, stream):
    rate, channels = int(stream["sample_rate"]), int(stream["channels"])
    process = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-acodec",
            "pcm_f32le",
            "-f",
            "f32le",
            "pipe:1",
        ],
        capture_output=True,
        timeout=120,
    )
    if process.returncode:
        raise ResearchError(process.stderr.decode(errors="replace")[-1500:])
    values = np.frombuffer(process.stdout, dtype="<f4")
    if len(values) % channels or not len(values) or not np.isfinite(values).all():
        raise ResearchError("Jumlah/frame/finite decoder tidak valid.")
    return values.reshape(-1, channels), rate


def audio_details(audio, rate):
    return {
        "sample_rate": rate,
        "channels": audio.shape[1],
        "frames": len(audio),
        "duration_seconds": len(audio) / rate,
        "all_finite": bool(np.isfinite(audio).all()),
        "peak": float(np.abs(audio).max()),
        "rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
        "samples_at_or_above_one": int((np.abs(audio) >= 1).sum()),
        "pcm_sha256": pcm_fingerprint(audio, rate),
    }
