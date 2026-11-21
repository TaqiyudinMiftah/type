"""CPU replay of every embedding and independent ridge solution for every fold."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from audit_mka_zeroshot import rebuild
from scipy.linalg import solve
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.fewshot import _beats
from ketik_audio.frozen import extract, frozen_encoder
from ketik_audio.mka import KEYS
from ketik_audio.mka_lopo import (
    CACHE,
    CHECKPOINT,
    ENCODERS,
    PARENT,
    ROOT,
    SEEDS,
    SOURCE,
    extract_beats,
    fit_head,
    load_cache,
    parent_scores,
    permuted_parent_labels,
    signature,
    split_plan,
    verified_records,
)
from ketik_audio.mka_lopo_analysis import check_conditions
from ketik_audio.mka_zeroshot import mel_cpu, metrics
from ketik_audio.resnet_fewshot import initialize_model, state_digest


def direct_ridge(x, labels):
    """Solve the centered primal/dual objective without RidgeClassifier."""
    targets = -np.ones((len(labels), 36), dtype=np.float64)
    targets[np.arange(len(labels)), labels] = 1
    x_mean, y_mean = x.mean(0), targets.mean(0)
    centered_x, centered_y = x - x_mean, targets - y_mean
    if len(x) < x.shape[1]:
        gram = centered_x @ centered_x.T + np.eye(len(x))
        coefficients = centered_x.T @ solve(gram, centered_y, assume_a="pos")
    else:
        gram = centered_x.T @ centered_x + np.eye(x.shape[1])
        coefficients = solve(gram, centered_x.T @ centered_y, assume_a="pos")
    return coefficients.T, y_mean - x_mean @ coefficients


def check_arrays(actual, expected, description, tolerance=1e-9):
    if not np.allclose(actual, expected, atol=tolerance, rtol=tolerance):
        raise ResearchError(f"Audit berbeda: {description}")
    return float(np.max(np.abs(actual - expected)))


def cpu_embeddings(records, cached, output):
    with np.load(PARENT / "inputs.npz", allow_pickle=False) as data:
        saved = {name: data[name] for name in data.files}
    reconstruction = rebuild(records, saved)
    wave = saved["peak_wave"]
    mel = mel_cpu(wave)
    if not np.array_equal(mel, saved["peak_mel"]):
        raise ResearchError("Frontend ResNet berbeda dari cache induk.")
    del saved
    fresh, states = {}, []
    model, _ = _beats(CHECKPOINT, "cpu")
    before = state_digest(model.state_dict())
    fresh["beats"] = extract_beats(model, wave, "cpu", progress=True)
    with np.load(ROOT / "data/external/harrison-fewshot/segments.npz", allow_pickle=False) as data:
        indices = np.arange(36) * 25
        source_vectors = extract_beats(model, data["zoom_wave"][indices], "cpu")
    with np.load(
        ROOT / "data/external/harrison-fewshot-features/features.npz", allow_pickle=False
    ) as data:
        source_difference = check_arrays(
            source_vectors, data["zoom_beats"][indices], "kontrol BEATs sumber", tolerance=1e-3
        )
    if state_digest(model.state_dict()) != before:
        raise ResearchError("State BEATs berubah dalam audit CPU.")
    states.append({"encoder": "beats", "state_digest": before, "unchanged": True})
    del model
    for seed in SEEDS:
        folder = SOURCE / f"sources/s{seed}"
        model = (
            initialize_model(
                seed, torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            )
            .requires_grad_(False)
            .eval()
        )
        before = state_digest(model.state_dict())
        with np.load(folder / "normalization.npz", allow_pickle=False) as stats:
            fresh[f"resnet_s{seed}"] = extract(
                frozen_encoder(model), mel, stats["mean"], stats["std"], "cpu"
            )
        if state_digest(model.state_dict()) != before:
            raise ResearchError("State ResNet berubah dalam audit CPU.")
        states.append({"encoder": f"resnet_s{seed}", "state_digest": before, "unchanged": True})
    comparisons = []
    for key in ENCODERS:
        maximum = check_arrays(fresh[key], cached[key], f"seluruh embedding {key}", tolerance=1e-3)
        comparisons.append(
            {
                "encoder": key,
                "maximum_absolute_difference": maximum,
                "rms_difference": float(np.sqrt(np.mean((fresh[key] - cached[key]) ** 2))),
                "rows_checked": len(records),
            }
        )
    np.savez_compressed(output / "cpu-features.npz", **fresh)
    return fresh, {
        "crop_reconstruction": reconstruction,
        "feature_comparisons": comparisons,
        "source_beats_control_max_abs_difference": source_difference,
        "state_checks": states,
    }


def run(root, cache, output):
    torch.set_num_threads(4)
    started = time.perf_counter()
    result = read_json(root / "results.json")
    check_conditions(result)
    manifest, features = load_cache(cache)
    config = read_json(root / "lock.json")["config"]
    if (
        config != result["config"]
        or config["input_sha256"] != manifest["input_sha256"]
        or config["cache_manifest_sha256"] != file_hash(cache / "manifest.json")
        or config["cache_features_sha256"] != file_hash(cache / "features.npz")
    ):
        raise ResearchError("Lock/cache/run berbeda.")
    for name, digest in result["artifact_sha256"].items():
        if file_hash(root / name) != digest:
            raise ResearchError(f"Artefak run berubah: {name}")
    records = verified_records()
    plan = split_plan(records)
    if read_json(root / "split-plan.json")["splits"] != plan:
        raise ResearchError("Split tersimpan berbeda dari protokol.")
    if file_hash(root / "split-plan.json") != read_json(root / "lock.json")["split_plan_sha256"]:
        raise ResearchError("Hash rencana split berbeda.")
    by_split = {(r["heldout"], r["k"], r["draw"]): r for r in plan}
    fresh_directory(output)
    fresh, extraction = cpu_embeddings(records, features, output)
    cache_states = {r["encoder"]: r for r in manifest["encoders"]}
    for row in extraction["state_checks"]:
        if cache_states[row["encoder"]]["before"] != row["state_digest"]:
            raise ResearchError("State encoder cache/CPU berbeda.")
    labels = np.asarray([r["label"] for r in records])
    normalized = {key: normalize(value.astype(np.float64)) for key, value in features.items()}
    cpu_normalized = {key: normalize(value.astype(np.float64)) for key, value in fresh.items()}
    checks, predictions, changed = [], [], []
    with threadpool_limits(limits=4):
        for number, row in enumerate(result["conditions"], 1):
            split = by_split[row["heldout"], row["k"], row["draw"]]
            support, test = (np.asarray(split[k]) for k in ("support", "test"))
            if (
                any(records[i]["platform"] == row["heldout"] for i in support)
                or any(records[i]["platform"] != row["heldout"] for i in test)
                or {records[i]["parent_group_id"] for i in support}
                & {records[i]["parent_group_id"] for i in test}
                or {records[i]["pcm_sha256"] for i in support}
                & {records[i]["pcm_sha256"] for i in test}
            ):
                raise ResearchError("Kebocoran induk/PCM/platform terdeteksi.")
            fit_y = permuted_parent_labels(records, support) if row["permuted"] else labels[support]
            if row["support_count"] != 180 * row["k"] or row["support_parents"] != 180:
                raise ResearchError("Anggaran label/induk berbeda.")
            x, cpu_x = normalized[row["encoder"]], cpu_normalized[row["encoder"]]
            with np.load(root / row["artifact"], allow_pickle=False) as saved:
                for key, value in (
                    ("support", support),
                    ("test", test),
                    ("fit_y", fit_y),
                    ("test_y", labels[test]),
                    ("classes", np.arange(36)),
                ):
                    if not np.array_equal(saved[key], value):
                        raise ResearchError(f"Indeks/label tersimpan berbeda: {key}")
                head, refit, fitted = fit_head(
                    features[row["encoder"]], labels, support, test, fit_y
                )
                coef, intercept = direct_ridge(x[support], fit_y)
                maximum_coef = check_arrays(coef, saved["coef"], "koefisien ridge langsung")
                check_arrays(intercept, saved["intercept"], "intercept ridge langsung")
                check_arrays(head.coef_, saved["coef"], "fit ulang koefisien")
                check_arrays(head.intercept_, saved["intercept"], "fit ulang intercept")
                scores = x[test] @ coef.T + intercept
                check_arrays(scores, saved["scores"], "skor solusi langsung")
                check_arrays(refit, saved["scores"], "skor fit ulang")
                check_arrays(fitted, saved["support_scores"], "skor dukungan")
                if not np.array_equal(scores.argmax(1), saved["scores"].argmax(1)):
                    raise ResearchError("Prediksi solusi langsung berbeda.")
                # Use stored arithmetic for metrics; the direct solution was checked above.
                scores = saved["scores"].copy()
                cpu_scores = cpu_x[test] @ saved["coef"].T + saved["intercept"]
            parent_y, parent_values, parents = parent_scores(records, test, scores)
            _, cpu_parent, _ = parent_scores(records, test, cpu_scores)
            if (
                metrics(labels[test], refit) != row["test"]
                or metrics(parent_y, parent_values) != row["parent"]
                or float((fitted.argmax(1) == fit_y).mean()) != row["support_accuracy"]
            ):
                raise ResearchError("Metrik fit ulang tidak identik.")
            clip_changes = int((scores.argmax(1) != cpu_scores.argmax(1)).sum())
            parent_changes = int((parent_values.argmax(1) != cpu_parent.argmax(1)).sum())
            if clip_changes or parent_changes:
                changed.append(
                    {
                        "id": row["id"],
                        "clip_changes": clip_changes,
                        "parent_changes": parent_changes,
                        "cpu_accuracy": float((cpu_scores.argmax(1) == labels[test]).mean()),
                    }
                )
            checks.append(
                {
                    "id": row["id"],
                    "maximum_direct_coef_difference": maximum_coef,
                    "maximum_refit_logit_difference": float(np.abs(refit - scores).max()),
                    "cpu_embedding_clip_prediction_changes": clip_changes,
                    "cpu_embedding_parent_prediction_changes": parent_changes,
                }
            )
            for unit, ids, truth, values, cpu_values in (
                ("clip", [records[i]["path"] for i in test], labels[test], scores, cpu_scores),
                ("parent", parents, parent_y, parent_values, cpu_parent),
            ):
                for identity, true, predicted, cpu_pred in zip(
                    ids, truth, values.argmax(1), cpu_values.argmax(1), strict=True
                ):
                    predictions.append(
                        {
                            "condition": row["id"],
                            "heldout": row["heldout"],
                            "unit": unit,
                            "id": identity,
                            "true_key": KEYS[int(true)],
                            "predicted_key": KEYS[int(predicted)],
                            "cpu_feature_predicted_key": KEYS[int(cpu_pred)],
                            "correct": int(true == predicted),
                        }
                    )
            if number % 52 == 0 or number == 336:
                print(f"Audited heads {number}/336", flush=True)
    with (output / "classifications.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    audit = {
        "created_at": now(),
        "status": "passed",
        "run_results_sha256": file_hash(root / "results.json"),
        "cache_manifest_sha256": file_hash(cache / "manifest.json"),
        "code_sha256": file_hash(Path(__file__)),
        "analysis_code_sha256": file_hash(ROOT / "src/ketik_audio/mka_lopo_analysis.py"),
        **extraction,
        "head_checks": checks,
        "heads_verified": len(checks),
        "cpu_feature_prediction_changes": changed,
        "classification_rows": len(predictions),
        "all_metrics_match": True,
        "all_splits_parent_disjoint": True,
        "gpu_used_for_audit": False,
        "seconds": time.perf_counter() - started,
        "artifact_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir())},
    }
    if signature() != manifest["input_sha256"]:
        raise ResearchError("Input berubah selama audit.")
    write_json(output / "audit.json", audit)
    print(f"Full LOPO audit passed; CPU-feature changes in {len(changed)} conditions", flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/mka-lopo")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/mka-lopo-audit")
    args = parser.parse_args()
    try:
        run(args.run, args.cache, args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
