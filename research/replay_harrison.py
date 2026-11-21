"""Audit saved checkpoints on CPU, export predictions and misclassified audio."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf

from ketik_audio.common import ResearchError, file_hash, fresh_directory, read_json, write_json
from ketik_audio.harrison import KEYS, _reference_model, cached_features, measure


def replay(root: Path, prepared: Path, output: Path):
    import torch

    torch.set_num_threads(4)
    record = read_json(root / "results.json")
    if record["status"] != "complete":
        raise ResearchError("Tunggu seluruh eksperimen selesai sebelum audit akhir.")
    manifest = read_json(prepared / "manifest.json")
    if file_hash(prepared / "segments.npz") != manifest["segments_sha256"]:
        raise ResearchError("Checksum dataset berubah.")
    with np.load(prepared / "segments.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    features = {
        domain: cached_features(arrays[f"{domain}_wave"], int(arrays[f"{domain}_rate"]))
        for domain in ["phone", "zoom"]
    }
    fresh_directory(output)
    rows, verified = [], []
    for row in record["results"]:
        directory = root / row["directory"]
        destination = output / row["directory"]
        destination.mkdir()
        domain = row["domain"]
        target = "zoom" if domain == "phone" else "phone"
        ids = arrays[f"{domain}_ids"]
        indices = {name: i for i, name in enumerate(ids)}
        split = read_json(directory / "split.json")
        test = np.asarray([indices[name] for name in split["test"]])
        mel, mfcc = features[domain]
        target_mel, target_mfcc = features[target]
        y, target_y = arrays[f"{domain}_y"], arrays[f"{target}_y"]
        if row["kind"].startswith("mfcc"):
            model = joblib.load(directory / "model.joblib")
            scores = model.decision_function(mfcc[test])
            cross_scores = model.decision_function(target_mfcc)
        else:
            model, _ = _reference_model()
            model.load_state_dict(
                torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
            )
            model.eval()
            with np.load(directory / "normalization.npz", allow_pickle=False) as normalization:
                mean, std = normalization["mean"], normalization["std"]

            def predict(images, model=model, mean=mean, std=std):
                with torch.inference_mode():
                    values = [
                        model(
                            torch.from_numpy(((images[i : i + 32] - mean) / std)[:, None])
                        ).numpy()
                        for i in range(0, len(images), 32)
                    ]
                return np.concatenate(values)

            scores, cross_scores = predict(mel[test]), predict(target_mel)
        for metric_name, labels, values in [
            ("test", y[test], scores),
            ("cross_channel", target_y, cross_scores),
        ]:
            measured = measure(labels, values)
            if measured["confusion_matrix"] != row[metric_name]["confusion_matrix"]:
                raise ResearchError(
                    f"Prediksi CPU berbeda dari artefak asli: {row['directory']}/{metric_name}"
                )
        prediction = scores.argmax(axis=1)
        np.savez_compressed(
            destination / "scores.npz",
            test_ids=ids[test],
            y=y[test],
            scores=scores,
            cross_ids=arrays[f"{target}_ids"],
            cross_y=target_y,
            cross_scores=cross_scores,
        )
        for index, actual, predicted in zip(test, y[test], prediction, strict=True):
            rows.append(
                {
                    "domain": domain,
                    "model": row["kind"],
                    "split": row["split"],
                    "seed": row["seed"],
                    "event_id": ids[index],
                    "actual": KEYS[actual],
                    "predicted": KEYS[predicted],
                    "correct": int(actual == predicted),
                }
            )
            if row["kind"].startswith("coatnet") and actual != predicted:
                sf.write(
                    destination / f"{ids[index].replace(':', '-')}_as-{KEYS[predicted]}.wav",
                    arrays[f"{domain}_wave"][index],
                    int(arrays[f"{domain}_rate"]),
                    subtype="FLOAT",
                )
        verified.append({"directory": row["directory"], "test_and_cross_confusions_match": True})
        print(f"CPU replay cocok: {row['directory']}", flush=True)
    with (output / "classifications.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output / "audit.json",
        {
            "device": "cpu",
            "experiments": verified,
            "source_results_sha256": file_hash(root / "results.json"),
            "torch": torch.__version__,
        },
    )
    (output / "README.md").write_text(
        "# Audit checkpoint\n\nPrediksi diulang pada CPU tanpa pelatihan atau tuning. "
        "Confusion matrix uji dan transfer harus sama persis dengan artefak eksperimen. "
        "`classifications.csv` memuat prediksi per contoh; setiap `scores.npz` memuat "
        "skor dan identitasnya. WAV menyimpan kesalahan CoAtNet dengan label sumber "
        "dan prediksi pada nama file. Label mengikuti isolasi yang diaudit, dengan "
        "ketidakpastian `phone/J` tetap berlaku.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        replay(args.run, args.prepared, args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
