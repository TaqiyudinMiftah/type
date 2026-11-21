"""Raw-session validation and a fixed, auditable train/validation split."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import (
    detect,
    features_at,
    load_audio,
    mask_protocol,
    protocol_intervals,
    without_controls,
)
from .common import (
    LABELS,
    SCHEMA_VERSION,
    FeatureConfig,
    ResearchError,
    environment,
    file_hash,
    fresh_directory,
    now,
    read_json,
    role_for_session,
    write_json,
)
from .metrics import detection


@dataclass
class Session:
    path: Path
    metadata: dict
    times: np.ndarray
    labels: np.ndarray
    hashes: dict
    truth_times: np.ndarray
    truth_labels: np.ndarray
    ignored_intervals: list[tuple[float, float]]


def clean_markers(session: Session, duration: float, config: FeatureConfig) -> np.ndarray:
    """Same sample-quality policy for train, validation, and held-out classification."""
    valid = (session.times >= config.pre_ms / 1000) & (
        session.times + (config.segment_ms - config.pre_ms) / 1000 <= duration
    )
    for i, seconds in enumerate(session.times):
        start = seconds - config.pre_ms / 1000
        end = seconds + (config.segment_ms - config.pre_ms) / 1000
        controls = any(left < end and right > start for left, right in session.ignored_intervals)
        neighbors = (
            (session.truth_times > start)
            & (session.truth_times < end)
            & (np.abs(session.truth_times - seconds) > 1e-6)
        ).any()
        if controls or neighbors:
            valid[i] = False
    return valid


def load_session(root: Path, number: int) -> Session:
    path = root / "raw" / f"session-{number:02d}"
    metadata = read_json(path / "session.json")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ResearchError(f"Versi format sesi tidak didukung: {path}")
    if metadata.get("session") != number or metadata.get("role") != role_for_session(number):
        raise ResearchError(f"Identitas/peran sesi tidak cocok: {path}")
    if metadata.get("status") != "complete" or metadata.get("input_overflows", 0):
        raise ResearchError(f"Sesi belum lengkap atau audio hilang: {path}. Rekam ulang sesi ini.")
    try:
        info = sf.info(path / "audio.wav")
        with (path / "events.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        accepted = [row for row in rows if row["accepted"] == "1"]
        samples = np.asarray([int(row["sample_index"]) for row in accepted], dtype=np.int64)
        labels = np.asarray([LABELS.index(row["key"]) for row in accepted], dtype=np.int64)
        all_samples = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
        all_labels = np.asarray([LABELS.index(row["key"]) for row in rows], dtype=np.int64)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise ResearchError(f"Data sesi rusak: {path}: {exc}") from exc
    if info.samplerate != metadata.get("sample_rate") or info.frames != metadata.get("frames"):
        raise ResearchError(f"Metadata dan WAV tidak cocok: {path}")
    if not len(samples) or np.any(samples < 0) or np.any(samples >= info.frames):
        raise ResearchError(f"Penanda waktu kosong/di luar audio: {path}")
    if np.any(np.diff(all_samples) <= 0):
        raise ResearchError(f"Urutan penanda waktu tidak valid: {path}")
    if np.any(all_samples < 0) or np.any(all_samples >= info.frames):
        raise ResearchError(f"Penanda waktu salah tekan berada di luar audio: {path}")
    expected = metadata.get("per_key", 25)
    if any(np.count_nonzero(labels == i) != expected for i in range(len(LABELS))):
        raise ResearchError(f"Jumlah contoh harus seimbang untuk seluruh a–z: {path}")
    if metadata.get("accepted_events") != len(samples):
        raise ResearchError(f"Jumlah penanda waktu tidak cocok dengan metadata: {path}")
    if any(row["key"] != row["expected"] for row in accepted):
        raise ResearchError(f"Label salah tekan diterima sebagai data latih: {path}")
    intervals = protocol_intervals(metadata)
    keep = without_controls(all_samples / info.samplerate, intervals)
    return Session(
        path,
        metadata,
        samples / info.samplerate,
        labels,
        {name: file_hash(path / name) for name in ["audio.wav", "events.csv", "session.json"]},
        all_samples[keep] / info.samplerate,
        all_labels[keep],
        intervals,
    )


def prepare(root: Path, output: Path) -> dict:
    if output.exists():
        raise ResearchError(f"Tujuan sudah ada: {output}. Gunakan --output baru.")
    sessions = [load_session(root, i) for i in range(1, 6)]
    synthetic = {bool(s.metadata.get("synthetic", False)) for s in sessions}
    if len(synthetic) != 1:
        raise ResearchError("Data sintetis dan rekaman nyata tidak boleh dicampur.")
    hashes = [s.hashes["audio.wav"] for s in sessions]
    if len(set(hashes)) != len(hashes):
        raise ResearchError("Audio yang sama ditemukan pada beberapa sesi; split tidak independen.")
    config = FeatureConfig()
    validation = sessions[-1]
    validation_audio = mask_protocol(
        load_audio(validation.path / "audio.wav", config), validation.ignored_intervals, config
    )
    choices = []
    for k in [3.0, 5.0, 8.0]:
        candidate = replace(config, detector_k=k)
        measured = detection(
            validation.truth_times, detect(validation_audio, candidate), config.match_ms / 1000
        )
        choices.append({"k": k, **measured})
    best = max(choices, key=lambda item: (item["f1"], -abs(item["k"] - 5.0)))
    config = replace(config, detector_k=best["k"])
    mels, mfccs, targets, numbers, identifiers = [], [], [], [], []
    rejected = []
    for session in sessions:
        print(
            f"Memproses sesi {session.metadata['session']} ({len(session.times)} ketukan)…",
            flush=True,
        )
        audio = load_audio(session.path / "audio.wav", config)
        # Classifier crops use keyboard markers; detector quality is evaluated separately.
        valid = clean_markers(session, len(audio) / config.sample_rate, config)
        for index in np.flatnonzero(~valid):
            rejected.append(
                {
                    "session": session.metadata["session"],
                    "event": int(index),
                    "reason": "boundary_or_overlapping_control_or_key",
                }
            )
        mel, mfcc = features_at(audio, session.times[valid], config)
        mels.append(mel)
        mfccs.append(mfcc)
        targets.extend(session.labels[valid])
        numbers.extend([session.metadata["session"]] * int(valid.sum()))
        identifiers.extend(f"{session.metadata['session']}:{i}" for i in np.flatnonzero(valid))
    y = np.asarray(targets, dtype=np.int64)
    session_numbers = np.asarray(numbers, dtype=np.int16)
    for role, mask in [("train", session_numbers <= 4), ("validation", session_numbers == 5)]:
        if set(y[mask]) != set(range(len(LABELS))):
            raise ResearchError(f"Seluruh 26 kelas harus tersedia pada split {role}.")
    fresh_directory(output)
    np.savez_compressed(
        output / "features.npz",
        mel=np.concatenate(mels),
        mfcc=np.concatenate(mfccs),
        y=y,
        sessions=session_numbers,
        sample_ids=np.asarray(identifiers),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now(),
        "labels": LABELS,
        "feature_config": config.to_dict(),
        "synthetic": synthetic.pop(),
        "split": {"train": [1, 2, 3, 4], "validation": [5], "test": [6]},
        "counts": {
            "train": int((session_numbers <= 4).sum()),
            "validation": int((session_numbers == 5).sum()),
        },
        "raw_sessions": [
            {"session": s.metadata["session"], "hashes": s.hashes, "per_key": s.metadata["per_key"]}
            for s in sessions
        ],
        "detector_selection": {"source_session": 5, "candidates": choices},
        "rejected_events": rejected,
        "features_sha256": file_hash(output / "features.npz"),
        "environment": environment(),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_prepared(path: Path) -> tuple[dict, dict]:
    manifest = read_json(path / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("labels") != list(LABELS):
        raise ResearchError("Format dataset/pemetaan kelas tidak sesuai.")
    if file_hash(path / "features.npz") != manifest.get("features_sha256"):
        raise ResearchError("Isi fitur berubah setelah prepare. Buat dataset baru.")
    with np.load(path / "features.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    sessions = arrays["sessions"]
    if set(sessions.tolist()) != {1, 2, 3, 4, 5}:
        raise ResearchError("Dataset pelatihan harus berisi sesi 1–5 saja.")
    for name in ["mel", "mfcc", "y"]:
        if len(arrays[name]) != len(sessions) or not np.isfinite(arrays[name]).all():
            raise ResearchError(f"Fitur dataset tidak valid: {name}")
    return arrays, manifest
