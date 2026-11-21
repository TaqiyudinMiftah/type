"""Geometry, provenance, and audio checks for released keystroke clips."""

from __future__ import annotations

import hashlib
import io
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import PurePosixPath

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .common import LABELS, ResearchError
from .skaid_pilot_common import KEYS, KS

VOCABULARIES = {"skaid12": KEYS, "latin26": LABELS}
TIERS = ("linked", "isolated80", "verified80")
MIN_RMS = 1e-4


def archive_members(z):
    rows = z.infolist()
    names = [r.filename for r in rows]
    if len(set(names)) != len(names):
        raise ResearchError("Nama member ZIP duplikat.")
    for row in rows:
        name = row.filename
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or not name
            or (row.external_attr >> 16) & 0o170000 == 0o120000
        ):
            raise ResearchError("Member ZIP tidak aman.")
    return sorted(
        (
            r
            for r in rows
            if not r.is_dir()
            and "__MACOSX" not in PurePosixPath(r.filename).parts
            and not PurePosixPath(r.filename).name.startswith("._")
        ),
        key=lambda r: r.filename,
    )


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def wav(blob):
    try:
        with sf.SoundFile(io.BytesIO(blob)) as f:
            rate, subtype = int(f.samplerate), f.subtype
            if f.format not in ("WAV", "WAVEX") or f.frames <= 0 or f.channels <= 0:
                raise ResearchError("Format/ukuran WAV tidak valid.")
            frames = f.frames
            values = f.read(dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError, ValueError) as exc:
        raise ResearchError("Dekode WAV gagal.") from exc
    if len(values) != frames or not np.isfinite(values).all():
        raise ResearchError("WAV terpotong atau tidak finite.")
    return values, rate, subtype


def letter(key):
    return (
        key.lower()
        if isinstance(key, str) and len(key) == 1 and key.isascii() and key.isalpha()
        else None
    )


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def isolated_window(t, other_events, ordinal, duration, pre=15, post=65):
    start, end = t - pre, t + post
    if start < 0 or end > duration:
        return False
    lo, hi = np.searchsorted(other_events, [start, end], side="left")
    return lo <= ordinal < hi and hi - lo == 1


def hear_crop(values, rate, t, duration, export_pre, export_post):
    """Return the nominal 80 ms around a log event in its original export."""
    start_ms, end_ms = t - export_pre, t + export_post
    if start_ms < 0 or end_ms > duration:
        return None, None
    start = int(np.floor(start_ms * rate / 1000))
    nominal_end = int(np.ceil(end_ms * rate / 1000))
    if abs(len(values) - (nominal_end - start)) > 2:
        return None, None
    center = int(np.floor(t * rate / 1000)) - start
    clip = center_crop(values, rate, center)
    return (clip, center) if clip is not None else (None, None)


def center_crop(values, rate, center):
    before, after = round(0.015 * rate), round(0.065 * rate)
    start, end = int(center) - before, int(center) + after
    if start < 0 or end > len(values):
        return None
    return values[start:end]


def peak_clip80(values, rate):
    size = round(0.005 * rate)
    if size <= 0:
        raise ResearchError("Sample rate tidak valid untuk blok 5 ms.")
    n = len(values) // size
    if n <= 0:
        return None, None
    powers = (values[: n * size].astype(np.float64) ** 2).reshape(n, size, -1).mean(axis=(1, 2))
    center = int(np.argmax(powers)) * size + size // 2
    return center_crop(values, rate, center), center


def downsample80(clip, rate):
    if rate != 48000 or clip.ndim != 2 or len(clip) != 3840:
        raise ResearchError("Baseline NC1 mengharapkan crop 80 ms native 48 kHz.")
    mono = clip.mean(axis=1, dtype=np.float32)
    result = resample_poly(mono, 1, 3, window=("kaiser", 5.0), padtype="constant")
    if len(result) != 1280 or not np.isfinite(result).all():
        raise ResearchError("Resampling crop gagal.")
    return result


def parent_error(clip, parent, rate, t):
    reference, clip64 = rms(clip), clip.astype(np.float64)
    if reference == 0 or clip.shape[1] != parent.shape[1]:
        return None, None
    start = int(np.floor((t - 15) * rate / 1000))
    count = len(clip)
    candidates = []
    for delta in (-2, -1, 0, 1, 2):
        a = start + delta
        if 0 <= a and a + count <= len(parent):
            error = rms(clip64 - parent[a : a + count]) / reference
            candidates.append((error, abs(delta), delta))
    if not candidates:
        return None, None
    error, _, delta = min(candidates)
    return error, delta


def cross_record_duplicates(records, hash_key, group_key):
    groups = defaultdict(set)
    for r in records:
        if r[hash_key] is not None:
            groups[r[hash_key]].add(r[group_key])
    return {k for k, values in groups.items() if len(values) > 1}


def hear_pairs(packs, records):
    counts = {tier: defaultdict(Counter) for tier in TIERS}
    for r in records:
        for tier in TIERS:
            if r[tier] and r["letter"]:
                counts[tier][r["pack"]][r["letter"]] += 1
    pairs = []
    for a in packs:
        for b in packs:
            if (
                a["participant"] != b["participant"]
                or a["device"] != b["device"]
                or a["session"] >= b["session"]
                or not a["track_unique"]
                or not b["track_unique"]
                or not a["parent_pcm_sha256"]
                or not b["parent_pcm_sha256"]
                or a["parent_pcm_sha256"] == b["parent_pcm_sha256"]
            ):
                continue
            gap = b["start_relative_seconds"] - a["start_relative_seconds"]
            if gap < a["native_seconds"]:
                continue
            pairs.append(
                {
                    "participant": a["participant"],
                    "calibration": a["index"],
                    "query": b["index"],
                    "gap_hours": gap / 3600,
                }
            )
    rows = []
    for tier in TIERS:
        for vocab, keys in VOCABULARIES.items():
            for k in KS:
                qualified = [
                    p
                    for p in pairs
                    if all(
                        counts[tier][p["calibration"]][c] >= k and counts[tier][p["query"]][c] >= 5
                        for c in keys
                    )
                ]
                rows.append(
                    {
                        "tier": tier,
                        "vocabulary": vocab,
                        "k": k,
                        "participants": len({p["participant"] for p in qualified}),
                        "pairs": len(qualified),
                        "pairs_at_least_12h": sum(p["gap_hours"] >= 12 for p in qualified),
                        "pair_ids": [[p["calibration"], p["query"]] for p in qualified],
                    }
                )
    return pairs, rows, counts


def nc1_identity(path):
    parts = PurePosixPath(path).parts
    if len(parts) != 4 or parts[0] != "dataset" or not parts[3].endswith(".wav"):
        raise ResearchError("Path NC1 tidak mengikuti struktur rilis.")
    _, condition, key, name = parts
    if key not in LABELS or not name.startswith(f"{key}_"):
        raise ResearchError("Label nama WAV dan folder berbeda.")
    stamp = datetime.strptime(PurePosixPath(name).stem[2:], "%Y%m%d_%H%M%S_%f")
    return condition, key, stamp
