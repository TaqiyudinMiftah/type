"""Replay every saved predictor on CPU and audit actual target label budgets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from sklearn.preprocessing import normalize

from ketik_audio.common import (
    ResearchError,
    environment,
    file_hash,
    fresh_directory,
    read_json,
    write_json,
)
from ketik_audio.fewshot import RATE, _beats, _normalization, _score, calibration_split
from ketik_audio.harrison import KEYS, _reference_model, measure, split_indices


def condition_directory(root, row):
    suffix = f"-s{row['support_seed']}" if row["k"] else ""
    return root / f"{row['source']}-to-{row['target']}-{row['method']}-k{row['k']}{suffix}"


def audit(root: Path, prepared: Path, cache: Path, checkpoint: Path, output: Path):
    import torch

    torch.set_num_threads(4)
    record = read_json(root / "results.json")
    manifest = read_json(root / "manifest.json")
    if record["status"] != "complete" or manifest["status"] != "complete":
        raise ResearchError("Eksperimen belum selesai.")
    if file_hash(prepared / "segments.npz") != record["signature"]["segments_sha256"]:
        raise ResearchError("Potongan berubah.")
    if file_hash(cache / "features.npz") != record["signature"]["features_sha256"]:
        raise ResearchError("Fitur berubah.")
    data_file = np.load(prepared / "segments.npz", allow_pickle=False)
    feature_file = np.load(cache / "features.npz", allow_pickle=False)
    data = {key: data_file[key] for key in data_file.files}
    features = {key: feature_file[key] for key in feature_file.files}
    fresh_directory(output)
    verified, predictions, hashes = [], [], {}
    for row in record["conditions"]:
        directory = condition_directory(root, row)
        source, target, k = row["source"], row["target"], row["k"]
        y, order = data[f"{target}_y"], data[f"{target}_order"]
        test = np.flatnonzero(order >= 15)
        scores_file = np.load(directory / "scores.npz", allow_pickle=False)
        if not np.array_equal(scores_file["indices"], test) or not np.array_equal(
            scores_file["y"], y[test]
        ):
            raise ResearchError(f"Indeks/label uji berubah: {directory}")
        if k:
            support = calibration_split(y, order, k, row["support_seed"])["support"]
            config = read_json(directory / "config.json")
            if (
                config["target_labels_used"] != 36 * k
                or config["target_validation_labels"] != 0
                or config["support_ids"] != data[f"{target}_ids"][support].tolist()
                or config["test_ids"] != data[f"{target}_ids"][test].tolist()
                or not np.array_equal(scores_file["support"], support)
            ):
                raise ResearchError(f"Anggaran/identitas dukungan tidak sesuai: {directory}")
        else:
            support = None
        if row["method"] == "beats_linear":
            head = joblib.load(directory / "head.joblib")
            if not np.array_equal(head.classes_, np.arange(36)):
                raise ResearchError("Urutan kelas kepala linear berubah.")
            if not k:
                config = read_json(directory / "config.json")
                source_train = split_indices(data[f"{source}_y"], data[f"{source}_order"], 42)[
                    "train"
                ]
                if (
                    config["fit_domain"] != source
                    or config["target_labels_used"] != 0
                    or config["support_ids"] != data[f"{source}_ids"][source_train].tolist()
                ):
                    raise ResearchError("BEATs K=0 tidak hanya memakai train sumber.")
            scores = head.decision_function(normalize(features[f"{target}_beats"][test]))
        else:
            source_dir = root / f"source-{source}"
            if k:
                stats_path = directory / "normalization.npz"
                expected_mean, expected_std = _normalization(features[f"{target}_mel"], support)
                weights = directory / "last.pt"
            else:
                stats_path = source_dir / "normalization.npz"
                source_train = split_indices(data[f"{source}_y"], data[f"{source}_order"], 42)[
                    "train"
                ]
                expected_mean, expected_std = _normalization(
                    features[f"{source}_mel"], source_train
                )
                weights = source_dir / "best.pt"
            stats = np.load(stats_path, allow_pickle=False)
            if not (
                np.array_equal(stats["mean"], expected_mean)
                and np.array_equal(stats["std"], expected_std)
            ):
                raise ResearchError(f"Statistik normalisasi memakai data berbeda: {directory}")
            model, _ = _reference_model()
            model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            scores = _score(
                model,
                features[f"{target}_mel"][test],
                torch.from_numpy(stats["mean"]),
                torch.from_numpy(stats["std"]),
                "cpu",
            )
            del model
        measured = measure(y[test], scores)
        for field in ["confusion_matrix", "correct", "n", "accuracy", "macro_f1", "top5"]:
            if measured[field] != row["test"][field]:
                raise ResearchError(f"Replay CPU berbeda ({field}): {directory}")
        difference = float(np.max(np.abs(scores - scores_file["scores"])))
        verified.append(
            {
                "directory": directory.name,
                "confusion_match": True,
                "max_score_absolute_difference": difference,
                "support_and_normalization_match": True,
            }
        )
        prediction = scores.argmax(axis=1)
        top5 = np.argsort(-scores, axis=1)[:, :5]
        exported = 0
        for pos, index in enumerate(test):
            predictions.append(
                {
                    "method": row["method"],
                    "source": source,
                    "target": target,
                    "k": k,
                    "support_seed": row["support_seed"],
                    "event_id": str(data[f"{target}_ids"][index]),
                    "actual": KEYS[y[index]],
                    "predicted": KEYS[prediction[pos]],
                    "correct": int(y[index] == prediction[pos]),
                    "top5": "".join(KEYS[i] for i in top5[pos]),
                }
            )
            if (
                k in [1, 10]
                and row["support_seed"] == 101
                and y[index] != prediction[pos]
                and exported < 3
            ):
                path = output / "error-audio" / directory.name
                path.mkdir(parents=True, exist_ok=True)
                event_id = str(data[f"{target}_ids"][index]).replace(":", "-")
                sf.write(
                    path / f"{event_id}_as-{KEYS[prediction[pos]]}.wav",
                    data[f"{target}_wave"][index],
                    RATE,
                    subtype="FLOAT",
                )
                exported += 1
        for path in directory.iterdir():
            if path.is_file():
                hashes[str(path.relative_to(root))] = file_hash(path)
        print(f"CPU replay and budget audit match: {directory.name}", flush=True)
    # Check encoder cache against raw clips, including test-only event IDs. This
    # also checks GPU versus CPU execution, independently of the linear heads.
    beats, _ = _beats(checkpoint, "cpu")
    cache_audit = {}
    with torch.inference_mode():
        for domain in ["phone", "zoom"]:
            sample = np.array([0, 14, 15, 224, 399, 424, 449, 899])
            tokens, _ = beats.extract_features(torch.from_numpy(data[f"{domain}_wave"][sample]))
            vectors = tokens.mean(dim=1).numpy()
            cached = features[f"{domain}_beats"][sample]
            max_difference = float(np.max(np.abs(vectors - cached)))
            if not np.allclose(vectors, cached, atol=3e-4, rtol=3e-4):
                raise ResearchError(f"BEATs CPU/raw cache berbeda: {domain}: {max_difference}")
            cache_audit[domain] = {
                "raw_events_checked": sample.tolist(),
                "max_absolute_difference": max_difference,
            }
    with (output / "classifications.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    write_json(
        output / "audit.json",
        {
            "source_results_sha256": file_hash(root / "results.json"),
            "conditions_replayed": len(verified),
            "confusion_matches": len(verified),
            "conditions": verified,
            "raw_beats_cache_check": cache_audit,
            "artifact_sha256": hashes,
            "environment": environment(),
            "device": "cpu",
        },
    )
    (output / "README.md").write_text(
        "# Audit pilot kalibrasi\n\nSeluruh kepala/model disimpan dan diulang pada CPU, "
        "tanpa pelatihan. Confusion matrix, top-5, jumlah benar dan metrik harus sama "
        "persis dengan hasil asli. Identitas dukungan, uji, anggaran label dan statistik "
        "normalisasi diperiksa terhadap protokol. BEATs juga diulang dari 16 klip WAV "
        "cache pada CPU untuk memeriksa encoder. `classifications.csv` memuat semua "
        "prediksi; `error-audio/` memuat maksimal tiga kesalahan per kondisi pada "
        "K=1/10, seed 101, mengikuti urutan kelas/event, bukan dipilih manual. "
        "Label sumber phone/J tetap tidak pasti.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        audit(args.run, args.prepared, args.cache, args.checkpoint, args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
