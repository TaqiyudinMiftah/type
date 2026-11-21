"""Held-out evaluation with audio-only detection and full sequence errors."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from .audio import detect, features_at, load_audio, mask_protocol, prediction_audio
from .common import LABELS, ResearchError, file_hash, fresh_directory, now, write_json
from .dataset import clean_markers, load_session
from .metrics import classify, detection, edit_distance
from .models import Predictor


def _prediction_rows(times, scores) -> list[dict]:
    rows = []
    for seconds, score in zip(times, scores, strict=True):
        order = np.argsort(-score, kind="stable")[:3]
        rows.append(
            {
                "time_seconds": float(seconds),
                "key": LABELS[int(order[0])],
                "score": float(score[order[0]]),
                "top3": [{"key": LABELS[int(i)], "score": float(score[i])} for i in order],
            }
        )
    return rows


def predict(model: Path, audio_path: Path, requested: str = "cpu") -> dict:
    predictor = Predictor(model, requested)
    start = time.perf_counter()
    audio = prediction_audio(audio_path, predictor.config)
    times = detect(audio, predictor.config)
    mel, mfcc = features_at(audio, times, predictor.config)
    preprocessing = time.perf_counter() - start
    predictor.synchronize()
    start = time.perf_counter()
    scores = predictor.scores(mel, mfcc)
    predictor.synchronize()
    seconds = time.perf_counter() - start
    rows = _prediction_rows(times, scores)
    return {
        "created_at": now(),
        "model_sha256": file_hash(model / "model.json"),
        "audio_sha256": file_hash(audio_path),
        "device": predictor.device_info,
        "sequence": "".join(row["key"] for row in rows),
        "events": rows,
        "preprocessing_seconds": preprocessing,
        "inference_seconds": seconds,
        "score_kind": predictor.metadata["score_kind"],
    }


def _write_csv(path: Path, fields: list[str], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _one(
    predictor: Predictor, session, audio: np.ndarray, output: Path, cached: tuple | None = None
) -> tuple[dict, tuple]:
    config = predictor.config
    valid = clean_markers(session, len(audio) / config.sample_rate, config)
    label_times, label_targets = session.times[valid], session.labels[valid]
    if not len(label_times):
        raise ResearchError("Tidak ada potongan uji yang lolos pemeriksaan kualitas.")
    if cached is None:
        start = time.perf_counter()
        label_mel, label_mfcc = features_at(audio, label_times, config)
        detected = detect(audio, config)
        detected_mel, detected_mfcc = features_at(audio, detected, config)
        preprocessing = time.perf_counter() - start
        cached = (label_mel, label_mfcc, detected, detected_mel, detected_mfcc, preprocessing)
    label_mel, label_mfcc, detected, detected_mel, detected_mfcc, preprocessing = cached
    # Warm-up outside the measured section, then synchronize the GPU's work queue.
    predictor.scores(label_mel[:1], label_mfcc[:1])
    predictor.synchronize()
    start = time.perf_counter()
    label_scores = predictor.scores(label_mel, label_mfcc)
    detected_scores = predictor.scores(detected_mel, detected_mfcc)
    predictor.synchronize()
    inference = time.perf_counter() - start
    predicted = label_scores.argmax(axis=1)
    measured = classify(label_targets, predicted)
    rows = _prediction_rows(detected, detected_scores)
    actual_text = "".join(LABELS[int(i)] for i in session.truth_labels)
    guessed_text = "".join(row["key"] for row in rows)
    distance = edit_distance(actual_text, guessed_text)
    result = {
        "created_at": now(),
        "test_session": 6,
        "synthetic": predictor.metadata["synthetic"],
        "model_sha256": file_hash(predictor.path / "model.json"),
        "test_hashes": session.hashes,
        "device": predictor.device_info,
        "classification": measured,
        "classification_coverage": {
            "accepted_markers": len(session.times),
            "evaluated": len(label_times),
            "excluded": int((~valid).sum()),
        },
        "detection": detection(session.truth_times, detected, config.match_ms / 1000),
        "end_to_end": {
            "truth": actual_text,
            "prediction": guessed_text,
            "edit_distance": distance,
            "cer": distance / max(1, len(actual_text)),
        },
        "timing": {
            "preprocessing_seconds": preprocessing,
            "inference_seconds": inference,
            "batch_event_count": len(label_scores) + len(detected_scores),
            "ms_per_event_batched": inference
            * 1000
            / max(1, len(label_scores) + len(detected_scores)),
        },
        "target_macro_f1": 0.7,
        "target_met": None if predictor.metadata["synthetic"] else measured["macro_f1"] >= 0.7,
    }
    fresh_directory(output)
    _write_csv(
        output / "classification.csv",
        ["time_seconds", "truth", "prediction", "score"],
        [
            {
                "time_seconds": float(t),
                "truth": LABELS[int(y)],
                "prediction": LABELS[int(p)],
                "score": float(label_scores[i, p]),
            }
            for i, (t, y, p) in enumerate(zip(label_times, label_targets, predicted, strict=True))
        ],
    )
    _write_csv(
        output / "detections.csv",
        ["time_seconds", "key", "score"],
        [{k: row[k] for k in ["time_seconds", "key", "score"]} for row in rows],
    )
    write_json(output / "evaluation.json", result)
    return result, cached


def evaluate(model: Path, root: Path, output: Path | None = None, requested: str = "cpu") -> dict:
    output = model / "evaluation" if output is None else output
    if output.exists():
        raise ResearchError(f"Tujuan evaluasi sudah ada: {output}. Gunakan --output baru.")
    predictor = Predictor(model, requested)
    session = load_session(root, 6)
    if bool(session.metadata.get("synthetic", False)) != predictor.metadata["synthetic"]:
        raise ResearchError("Model dan data uji harus sama-sama nyata atau sama-sama sintetis.")
    if any(
        session.hashes["audio.wav"] == source["hashes"]["audio.wav"]
        for source in predictor.metadata["source_manifest"]["raw_sessions"]
    ):
        raise ResearchError("Audio uji identik dengan audio latih/validasi; evaluasi ditolak.")
    audio = load_audio(session.path / "audio.wav", predictor.config)
    audio = mask_protocol(audio, session.ignored_intervals, predictor.config)
    result, cache = _one(predictor, session, audio, output)
    per_seed = []
    for child in predictor.metadata.get("seed_models", []):
        child_model = Predictor(model / child, requested)
        if child_model.config != predictor.config:
            raise ResearchError("Konfigurasi fitur antar-seed berbeda.")
        seed_result, _ = _one(child_model, session, audio, output / child, cache)
        per_seed.append(
            {
                "seed": child_model.metadata["seed"],
                "accuracy": seed_result["classification"]["accuracy"],
                "macro_f1": seed_result["classification"]["macro_f1"],
                "cer": seed_result["end_to_end"]["cer"],
            }
        )
    if per_seed:
        result["seeds"] = per_seed
        result["seed_statistics"] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in per_seed])),
                "std": float(np.std([row[metric] for row in per_seed])),
            }
            for metric in ["accuracy", "macro_f1", "cer"]
        }
        write_json(output / "evaluation.json", result)
    return result
