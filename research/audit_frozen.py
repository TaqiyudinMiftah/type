"""Audit every frozen-head predictor, target budget, normalization and encoder."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import normalize

from ketik_audio.common import ResearchError, file_hash, fresh_directory, read_json, write_json
from ketik_audio.fewshot import KS, SUPPORT_SEEDS, _normalization, calibration_split
from ketik_audio.frozen import assert_unchanged, extract, fit_head, frozen_encoder, state_snapshot
from ketik_audio.harrison import KEYS, _reference_model, measure, split_indices


def run(root: Path, primary: Path, prepared: Path, cache: Path, output: Path):
    torch.set_num_threads(4)
    result = read_json(root / "results.json")
    sig = result["signature"]
    if result["status"] != "complete":
        raise ResearchError("Eksperimen beku belum selesai.")
    if (
        file_hash(primary / "results.json") != sig["primary_results_sha256"]
        or file_hash(prepared / "segments.npz") != sig["segments_sha256"]
        or file_hash(cache / "features.npz") != sig["input_features_sha256"]
    ):
        raise ResearchError("Input audit berubah.")
    identities = {
        (r["source"], r["target"], r["method"], r["k"], r["support_seed"])
        for r in result["conditions"]
    }
    expected = {
        (source, target, method, k, seed)
        for source, target in [("phone", "zoom"), ("zoom", "phone")]
        for method in ["pretrained_source_norm", "random_source_norm", "pretrained_support_norm"]
        for k in KS
        for seed in SUPPORT_SEEDS
    }
    expected |= {
        (s, t, m, 0, None)
        for s, t in [("phone", "zoom"), ("zoom", "phone")]
        for m in ["pretrained_source_norm", "random_source_norm"]
    }
    if identities != expected or len(result["conditions"]) != 94:
        raise ResearchError("Jumlah/identitas 94 kondisi tidak sesuai protokol.")
    with np.load(prepared / "segments.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(cache / "features.npz", allow_pickle=False) as archive:
        mel = {domain: archive[f"{domain}_mel"] for domain in ["phone", "zoom"]}
    fresh_directory(output)
    models, vectors, encoder_audits, snapshots = {}, {}, [], {}
    for row in result["encoders"]:
        identity = f"{row['source']}-{row['kind']}"
        folder = root / "encoders" / identity
        for filename, key in [
            ("model.pt", "weights_sha256"),
            ("features.npz", "features_sha256"),
            ("normalization.npz", "normalization_sha256"),
        ]:
            if file_hash(folder / filename) != row[key]:
                raise ResearchError(f"Artefak encoder berubah: {identity}/{filename}")
        model, _ = _reference_model()
        saved = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        if row["kind"] == "pretrained":
            original = torch.load(
                primary / f"source-{row['source']}/best.pt", map_location="cpu", weights_only=True
            )
        else:
            torch.manual_seed(42)
            initial, _ = _reference_model()
            original = initial.state_dict()
        if any(not torch.equal(saved[key], original[key]) for key in original):
            raise ResearchError("Bobot encoder berbeda dari checkpoint sumber/inisialisasi.")
        model.load_state_dict(saved)
        models[identity] = frozen_encoder(model)
        snapshots[identity] = state_snapshot(models[identity])
        with np.load(folder / "features.npz", allow_pickle=False) as archive:
            vectors[identity] = {key: archive[key] for key in archive.files}
        with np.load(folder / "normalization.npz", allow_pickle=False) as stats:
            mean, std = stats["mean"], stats["std"]
        source = row["source"]
        train = split_indices(data[f"{source}_y"], data[f"{source}_order"], 42)["train"]
        expected_mean, expected_std = _normalization(mel[source], train)
        if not (np.array_equal(mean, expected_mean) and np.array_equal(std, expected_std)):
            raise ResearchError("Normalisasi sumber bukan source-train saja.")
        for domain in ["phone", "zoom"]:
            sample = np.array([0, 14, 15, 899])
            actual = extract(models[identity], mel[domain][sample], mean, std, "cpu")
            cached = vectors[identity][domain][sample]
            if not np.allclose(actual, cached, rtol=1e-3, atol=1e-4):
                raise ResearchError(f"Fitur encoder CPU/GPU berbeda: {identity}/{domain}")
            encoder_audits.append(
                {
                    "encoder": identity,
                    "domain": domain,
                    "events": sample.tolist(),
                    "max_absolute_difference": float(np.max(np.abs(actual - cached))),
                }
            )
    verified, rows, feature_checks, artifact_hashes = [], [], [], {}
    for row in result["conditions"]:
        folder = root / row["directory"]
        cfg = read_json(folder / "config.json")
        source, target, k = row["source"], row["target"], row["k"]
        fit_domain = target if k else source
        support = (
            calibration_split(data[f"{target}_y"], data[f"{target}_order"], k, row["support_seed"])[
                "support"
            ]
            if k
            else split_indices(data[f"{source}_y"], data[f"{source}_order"], 42)["train"]
        )
        test = np.flatnonzero(data[f"{target}_order"] >= 15)
        if (
            cfg["fit_domain"] != fit_domain
            or cfg["support_ids"] != data[f"{fit_domain}_ids"][support].tolist()
            or cfg["test_ids"] != data[f"{target}_ids"][test].tolist()
            or cfg["target_labels_used"] != 36 * k
            or cfg["target_validation_labels"] != 0
        ):
            raise ResearchError(f"Anggaran/ID dukungan tidak sesuai: {folder}")
        encoder_id = Path(cfg["encoder_directory"]).name
        if row["method"] == "pretrained_support_norm":
            mean, std = _normalization(mel[target], support)
            with np.load(folder / "vectors.npz", allow_pickle=False) as archive:
                fit_x, test_x = archive["fit"], archive["test"]
            selected = [support[0], support[-1], test[0], test[-1]]
            actual = extract(models[encoder_id], mel[target][selected], mean, std, "cpu")
            cached = np.stack([fit_x[0], fit_x[-1], test_x[0], test_x[-1]])
            if not np.allclose(actual, cached, rtol=1e-3, atol=1e-4):
                raise ResearchError(f"Fitur support_norm berbeda: {folder}")
            feature_checks.append(
                {
                    "directory": row["directory"],
                    "max_absolute_difference": float(np.max(np.abs(actual - cached))),
                }
            )
        else:
            with np.load(
                root / cfg["encoder_directory"] / "normalization.npz", allow_pickle=False
            ) as stats:
                mean, std = stats["mean"], stats["std"]
            fit_x, test_x = (
                vectors[encoder_id][fit_domain][support],
                vectors[encoder_id][target][test],
            )
        with np.load(folder / "normalization.npz", allow_pickle=False) as stats:
            if not (np.array_equal(mean, stats["mean"]) and np.array_equal(std, stats["std"])):
                raise ResearchError(
                    "Statistik kondisi tidak sesuai dukungan/sumber yang dinyatakan."
                )
        head = joblib.load(folder / "head.joblib")
        if not np.array_equal(head.classes_, np.arange(36)):
            raise ResearchError("Urutan kelas berubah.")
        scores = head.decision_function(normalize(test_x))
        measured = measure(data[f"{target}_y"][test], scores)
        for field in ["confusion_matrix", "accuracy", "macro_f1", "top5", "correct", "n"]:
            if measured[field] != row["test"][field]:
                raise ResearchError(f"Replay berbeda: {folder}/{field}")
        with np.load(folder / "scores.npz", allow_pickle=False) as saved:
            if (
                not np.array_equal(saved["indices"], test)
                or not np.array_equal(saved["support"], support)
                or not np.array_equal(saved["scores"], scores)
            ):
                raise ResearchError("Skor/indeks tersimpan berbeda.")
        support_accuracy = float(
            np.mean(head.predict(normalize(fit_x)) == data[f"{fit_domain}_y"][support])
        )
        if support_accuracy != row["support_accuracy"]:
            raise ResearchError("Akurasi dukungan tidak cocok.")
        for index, prediction in zip(test, scores.argmax(axis=1), strict=True):
            rows.append(
                {
                    "method": row["method"],
                    "source": source,
                    "target": target,
                    "k": k,
                    "support_seed": row["support_seed"],
                    "event_id": str(data[f"{target}_ids"][index]),
                    "actual": KEYS[data[f"{target}_y"][index]],
                    "predicted": KEYS[prediction],
                }
            )
        verified.append(
            {
                "directory": row["directory"],
                "confusion_match": True,
                "budget_and_normalization_match": True,
            }
        )
        for path in folder.iterdir():
            if path.is_file():
                artifact_hashes[str(path.relative_to(root))] = file_hash(path)
    for key, encoder in models.items():
        assert_unchanged(encoder, snapshots[key])
    source_checks = []
    for row in result["source_checks"]:
        source = row["source"]
        split = split_indices(data[f"{source}_y"], data[f"{source}_order"], 42)
        x = vectors[f"{source}-{row['kind']}"][source]
        _, scores, _, _ = fit_head(
            x[split["train"]], data[f"{source}_y"][split["train"]], x[split["test"]]
        )
        if (
            measure(data[f"{source}_y"][split["test"]], scores)["confusion_matrix"]
            != row["test"]["confusion_matrix"]
        ):
            raise ResearchError("Pemeriksaan representasi sumber tidak dapat direproduksi.")
        source_checks.append({"source": source, "kind": row["kind"], "confusion_match": True})
    with (output / "classifications.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output / "audit.json",
        {
            "source_results_sha256": file_hash(root / "results.json"),
            "conditions_replayed": len(verified),
            "conditions": verified,
            "encoder_state_unchanged": True,
            "encoder_checks": encoder_audits,
            "support_norm_feature_checks": feature_checks,
            "source_checks": source_checks,
            "artifact_sha256": artifact_hashes,
            "device": "cpu",
        },
    )
    print(f"Verified {len(verified)} heads, 4 source checks, and frozen encoder CPU/GPU parity.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.run, args.primary, args.prepared, args.cache, args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
