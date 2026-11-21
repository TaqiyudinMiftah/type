"""Shared configuration, portable metadata, and artifact validation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LABELS = tuple("abcdefghijklmnopqrstuvwxyz")
SCHEMA_VERSION = 1


class ResearchError(Exception):
    """An actionable input, environment, or experiment error."""


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 44_100
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64
    n_mfcc: int = 20
    segment_ms: int = 300
    pre_ms: int = 50
    refractory_ms: int = 250
    match_ms: int = 75
    detector_k: float = 5.0

    def __post_init__(self) -> None:
        if self.sample_rate < 8000 or self.n_fft <= 0 or self.hop_length <= 0:
            raise ResearchError("Konfigurasi sample rate/FFT/hop tidak valid.")
        if not 0 < self.pre_ms < self.segment_ms:
            raise ResearchError("pre_ms harus lebih kecil daripada panjang segmen.")
        if self.n_mfcc > self.n_mels or self.n_mfcc < 1 or self.detector_k <= 0:
            raise ResearchError("Konfigurasi MFCC/mel/detektor tidak valid.")
        samples = round(self.segment_ms * self.sample_rate / 1000)
        if (
            self.n_mels < 8
            or samples < self.n_fft + 8 * self.hop_length
            or self.match_ms <= 0
            or self.refractory_ms <= 0
        ):
            raise ResearchError("Segmen terlalu pendek atau konfigurasi waktu/mel tidak valid.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FeatureConfig:
        return cls(**value)


def now() -> str:
    return datetime.now(UTC).isoformat()


def role_for_session(number: int) -> str:
    if not 1 <= number <= 6:
        raise ResearchError("Nomor sesi harus 1–6: 1–4 latih, 5 validasi, 6 uji.")
    return "train" if number <= 4 else "validation" if number == 5 else "test"


def read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ResearchError(f"Tidak dapat membaca JSON {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ResearchError(f"JSON harus berupa objek: {path}")
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def fresh_directory(path: Path) -> None:
    """Never silently replace an existing experiment or raw session."""
    if path.exists():
        raise ResearchError(f"Tujuan sudah ada: {path}. Pilih direktori keluaran baru.")
    path.mkdir(parents=True)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment() -> dict[str, Any]:
    packages = {}
    for name in ["ketik-audio", "numpy", "scipy", "librosa", "scikit-learn", "torch"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "gpu_visibility": {
            key: os.environ[key]
            for key in [
                "HIP_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "ROCM_SDK_TARGET_FAMILY",
            ]
            if key in os.environ
        },
    }
