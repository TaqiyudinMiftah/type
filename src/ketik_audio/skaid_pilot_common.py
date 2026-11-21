"""Pinned constants and provenance shared by the external SKAID classification pilot."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .common import ResearchError, file_hash, read_json

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "research/SKAID_PILOT_PROTOCOL.md"
NATIVE = ROOT / "reports/skaid-audit"
DRIFT = ROOT / "runs/skaid-drift"
DATA = ROOT / "data/external/skaid-pilot"
CACHE = ROOT / "data/external/skaid-pilot-features"
RUN = ROOT / "runs/skaid-pilot"
SOURCE = ROOT / "runs/resnet-fewshot"
CHECKPOINT = ROOT / "data/external/beats/BEATs_iter3_plus_AS2M.pt"
KEYS = tuple(sorted("etaoinshrdlu"))
KS = (1, 2, 3, 5, 10)
DRAWS = (101, 102, 103)
SEEDS = (42, 43, 44)
ENCODERS = ("mfcc", "beats", *(f"pretrained_s{s}" for s in SEEDS), *(f"random_s{s}" for s in SEEDS))


def array_hash(values):
    values = np.ascontiguousarray(values)
    h = hashlib.sha256(str((values.shape, str(values.dtype))).encode())
    h.update(values.tobytes())
    return h.hexdigest()


def verify(base, record):
    if record.get("status") != "complete":
        raise ResearchError(f"Tahap belum lengkap: {base}")
    for name, digest in record["input_sha256"].items():
        if file_hash(ROOT / name) != digest:
            raise ResearchError(f"Masukan berubah: {name}")
    for name, digest in record["artifact_sha256"].items():
        if file_hash(base / name) != digest:
            raise ResearchError(f"Artefak berubah: {base / name}")


def source_hashes():
    lock = read_json(SOURCE / "sources-locked.json")
    mapping = {
        "model_implementation_sha256": "src/ketik_audio/cnn_repair.py",
        "fewshot_implementation_sha256": "src/ketik_audio/fewshot.py",
        "harrison_implementation_sha256": "src/ketik_audio/harrison.py",
        "implementation_sha256": "src/ketik_audio/resnet_fewshot.py",
        "protocol_sha256": "research/RESNET_FEWSHOT_PROTOCOL.md",
        "segments_sha256": "data/external/harrison-fewshot/segments.npz",
        "features_sha256": "data/external/harrison-fewshot-features/features.npz",
    }
    result = {}
    for field, name in mapping.items():
        digest = file_hash(ROOT / name)
        if digest != lock["signature"][field]:
            raise ResearchError(f"Sumber Harrison berubah: {name}")
        result[name] = digest
    paths = [
        SOURCE / "sources-locked.json",
        SOURCE / "results.json",
        ROOT / "src/ketik_audio/frozen.py",
        ROOT / "src/ketik_audio/devices.py",
        ROOT / "research/vendor/beats/provenance.json",
        CHECKPOINT,
        ROOT / "data/external/beats/provenance.json",
    ]
    for seed in SEEDS:
        folder = SOURCE / f"sources/s{seed}"
        if file_hash(folder / "best.pt") != lock["source_checkpoints"][str(seed)]:
            raise ResearchError("Checkpoint sumber berbeda.")
        if (
            file_hash(folder / "normalization.npz")
            != read_json(folder / "result.json")["normalization_sha256"]
        ):
            raise ResearchError("Normalisasi sumber berbeda.")
        paths.extend(
            folder / n for n in ("best.pt", "normalization.npz", "config.json", "scores.npz")
        )
    vendor = ROOT / "research/vendor/beats"
    for name, entry in read_json(vendor / "provenance.json")["files"].items():
        path = vendor / name
        if file_hash(path) != entry["sha256"]:
            raise ResearchError("Vendor BEATs berubah.")
        paths.append(path)
    result.update({str(p.relative_to(ROOT)): file_hash(p) for p in paths})
    if (
        result[str(CHECKPOINT.relative_to(ROOT))]
        != "d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34"
    ):
        raise ResearchError("Checkpoint BEATs berbeda.")
    return result
