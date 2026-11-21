"""Fixed MFCC/Ridge building blocks for held-recording public-data evaluations."""

from __future__ import annotations

import subprocess
from collections import Counter

import numpy as np
from scipy.signal import resample_poly
from sklearn.metrics import confusion_matrix

from .common import ResearchError
from .skaid_pilot_common import KS

ALPHA = 0.001
DRAWS = (101, 102, 103)
SEED = 20261115


def ranks(rows, keys, n, group_index, draw, role):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, role, group_index, draw]))
    result = {}
    for key in keys:
        ids = sorted(r["id"] for r in rows if r["letter"] == key)
        if len(ids) != len(set(ids)) or len(ids) < n:
            raise ResearchError(f"Contoh unik tidak cukup untuk kelas {key}.")
        result[key] = [ids[i] for i in rng.permutation(len(ids))[:n]]
    return result


def prefix(ranked, keys, k):
    if k not in KS or set(ranked) != set(keys) or any(len(ranked[c]) < k for c in keys):
        raise ResearchError("Ranking dukungan tidak lengkap.")
    return [i for c in keys for i in ranked[c][:k]]


def permute(labels, group_index, draw):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 3, group_index, draw]))
    return rng.permutation(labels)


def op(args, payload):
    process = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", *args, "pipe:1"],
        input=payload,
        capture_output=True,
        timeout=120,
    )
    if process.returncode:
        raise ResearchError(process.stderr.decode(errors="replace")[-1500:])
    return process.stdout


def mulaw8k(waves):
    """Offline 16→8 kHz/µ-law/16 kHz; no filtering across example boundaries."""
    if waves.ndim != 2 or waves.shape[1] != 1280 or not len(waves) or not np.isfinite(waves).all():
        raise ResearchError("Codec membutuhkan matriks crop 80 ms/16 kHz.")
    filt = {"axis": 1, "window": ("kaiser", 5.0), "padtype": "constant"}
    low = resample_poly(waves.astype(np.float32), 1, 2, **filt)
    encoded = op(
        [
            "-f",
            "f32le",
            "-ar",
            "8000",
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "pcm_mulaw",
            "-f",
            "mulaw",
        ],
        low.astype("<f4", copy=False).tobytes(),
    )
    if len(encoded) != low.size:
        raise ResearchError("Panjang bitstream µ-law berbeda.")
    decoded = op(
        [
            "-f",
            "mulaw",
            "-ar",
            "8000",
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
        ],
        encoded,
    )
    if len(decoded) != 4 * low.size:
        raise ResearchError("Panjang decoder µ-law berbeda.")
    result = resample_poly(np.frombuffer(decoded, dtype="<f4").reshape(low.shape), 2, 1, **filt)
    if result.shape != waves.shape or not np.isfinite(result).all():
        raise ResearchError("Ukuran/sinyal sesudah codec tidak valid.")
    return result


def scores_metrics(labels, scores, keys, subset=None):
    if scores.shape != (len(labels), len(keys)) or not np.isfinite(scores).all():
        raise ResearchError("Ukuran skor/label tidak valid.")
    if subset is not None:
        ids = [keys.index(c) for c in subset]
        mask = np.isin(labels, ids)
        labels, scores = labels[mask], scores[mask]
    if not len(labels) or not np.all(np.isin(labels, np.arange(len(keys)))):
        raise ResearchError("Query kosong atau kelas di luar kosakata.")
    pred = scores.argmax(1)
    matrix = confusion_matrix(labels, pred, labels=np.arange(len(keys)))
    counts = matrix.sum(1)
    correct = matrix.diagonal()
    top3 = np.argsort(-scores, axis=1, kind="stable")[:, :3]
    return {
        "examples": len(labels),
        "correct": int(correct.sum()),
        "accuracy": float(correct.sum() / len(labels)),
        "balanced_accuracy_present": float(np.mean(correct[counts > 0] / counts[counts > 0])),
        "top3": float(np.mean(np.any(top3 == labels[:, None], axis=1))),
        "classes_present": [c for c, n in zip(keys, counts, strict=True) if n],
        "support_per_class": counts.tolist(),
        "correct_per_class": correct.tolist(),
        "confusion": matrix.tolist(),
    }


def check_roles(records, paths, rankings, selections, keys_by_dataset):
    by_id = {r["id"]: r for r in records}
    if len(by_id) != len(records):
        raise ResearchError("ID klip duplikat.")
    used = set()
    for p in paths:
        a, b = p["calibration"], p["query_group"]
        keys = keys_by_dataset[p["dataset"]]
        ar, qr = rankings[a], selections[b]
        qrows = [by_id[i] for i in qr]
        qcounts = Counter(r["letter"] for r in qrows)
        if (
            a == b
            or {r["group"] for r in qrows} != {b}
            or len(qr) != len(set(qr))
            or not qcounts
            or set(qcounts) - set(keys)
            or min(qcounts.values()) < 5
            or (p["dataset"] == "hear" and set(qcounts.values()) != {5})
            or (p["dataset"] == "hear" and set(qcounts) != set(keys))
        ):
            raise ResearchError("Grup atau cakupan query tidak sesuai.")
        used.update(qr)
        query_sha = {r["crop_pcm_sha256"] for r in qrows}
        for draw in DRAWS:
            ids = prefix(ar[str(draw)], keys, 10)
            rows = [by_id[i] for i in ids]
            if (
                len(set(ids)) != 10 * len(keys)
                or set(ids) & set(qr)
                or {r["group"] for r in rows} != {a}
                or {r["crop_pcm_sha256"] for r in rows} & query_sha
                or [r["letter"] for r in rows] != [c for c in keys for _ in range(10)]
            ):
                raise ResearchError("Dukungan lintas rekaman/kelas tidak konsisten.")
            used.update(ids)
    if used != set(by_id):
        raise ResearchError("Ada klip tercatat tanpa peran di plan.")
