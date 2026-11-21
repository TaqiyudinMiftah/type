"""Audit every inner candidate and locked final head with an independent ridge solve."""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.mka import KEYS
from ketik_audio.mka_lopo import ROOT, parent_scores, permuted_parent_labels
from ketik_audio.mka_nested import (
    ALPHAS,
    BASELINE,
    PREVIOUS_AUDIT,
    baseline_lookup,
    inputs,
    nested_plan,
    ridge,
    verify_selection,
)
from ketik_audio.mka_zeroshot import metrics


def direct_ridge(x, labels, alpha):
    target = -np.ones((len(labels), 36), dtype=np.float64)
    target[np.arange(len(labels)), labels] = 1
    mx, my = x.mean(0), target.mean(0)
    centered, targets = x - mx, target - my
    if len(x) < x.shape[1]:
        gram = centered @ centered.T + alpha * np.eye(len(x))
        coef = centered.T @ solve(gram, targets, assume_a="pos")
    else:
        gram = centered.T @ centered + alpha * np.eye(x.shape[1])
        coef = solve(gram, centered.T @ targets, assume_a="pos")
    return coef.T, my - mx @ coef


def check(actual, expected, label):
    if not np.allclose(actual, expected, atol=1e-7, rtol=1e-7):
        raise ResearchError(f"Replay berbeda: {label}")
    return float(np.abs(actual - expected).max())


def run(root, output):
    started = time.perf_counter()
    signature, features, records, baseline = inputs()
    selection = verify_selection(root, signature)
    result = read_json(root / "results.json")
    if result["status"] != "complete" or result["config"] != selection["config"]:
        raise ResearchError("Run belum selesai/konfigurasi berbeda.")
    lock_sha = file_hash(root / "selections-locked.json")
    evaluation = read_json(root / "evaluation-start.json")
    if (
        result["selection_lock_sha256"] != lock_sha
        or evaluation["selection_lock_sha256"] != lock_sha
    ):
        raise ResearchError("Evaluasi tidak memakai lock seleksi yang sama.")
    lock_time = datetime.fromisoformat(read_json(root / "selections-locked.json")["created_at"])
    if (
        not lock_time
        < datetime.fromisoformat(evaluation["created_at"])
        <= datetime.fromisoformat(result["created_at"])
    ):
        raise ResearchError("Urutan lock/evaluasi tidak sesuai.")
    for name, digest in result["artifact_sha256"].items():
        if file_hash(root / name) != digest:
            raise ResearchError(f"Artefak run berubah: {name}")
    plan = {p["heldout"]: p for p in nested_plan(records)}
    if read_json(root / "split-plan.json")["outer_splits"] != list(plan.values()):
        raise ResearchError("Rencana split berubah.")
    conditions = {r["id"]: r for r in result["conditions"]}
    if len(result["conditions"]) != 48 or set(conditions) != {t["id"] for t in selection["tasks"]}:
        raise ResearchError("Evaluasi harus memiliki semua 48 tugas unik.")
    previous = baseline_lookup(baseline)
    y = np.asarray([r["label"] for r in records])
    normalized = {key: normalize(value.astype(np.float64)) for key, value in features.items()}
    with np.load(PREVIOUS_AUDIT / "cpu-features.npz", allow_pickle=False) as saved:
        cpu = {key: normalize(saved[key].astype(np.float64)) for key in saved.files}
    fresh_directory(output)
    inner_checks, head_checks, predictions, cpu_changes = [], [], [], []
    with threadpool_limits(limits=4):
        for number, task in enumerate(selection["tasks"], 1):
            outer = plan[task["heldout"]]
            support, test = (np.asarray(outer[k]) for k in ("support", "test"))
            labels = permuted_parent_labels(records, support) if task["permuted"] else y[support]
            x = normalized[task["encoder"]]
            counts = np.zeros((len(ALPHAS), 5), dtype=np.int64)
            with np.load(root / task["artifact"], allow_pickle=False) as saved:
                for key, actual in (
                    ("alphas", np.asarray(ALPHAS)),
                    ("support", support),
                    ("fit_y", labels),
                    ("validation_indices", np.asarray([s["validation"] for s in outer["inner"]])),
                    ("train_indices", np.asarray([s["train"] for s in outer["inner"]])),
                ):
                    if not np.array_equal(saved[key], actual):
                        raise ResearchError(f"Metadata seleksi berbeda: {key}")
                for ai, alpha in enumerate(ALPHAS):
                    for fold, split in enumerate(outer["inner"]):
                        train, val = (np.asarray(split[k]) for k in ("train", "validation"))
                        local_train, local_val = (
                            np.asarray(split[k]) for k in ("train_local", "validation_local")
                        )
                        inner_y, val_y = labels[local_train], labels[local_val]
                        head = ridge(x[train], inner_y, alpha)
                        coef, intercept = direct_ridge(x[train], inner_y, alpha)
                        dc = check(coef, saved["coef"][ai, fold], "inner coefficient langsung")
                        di = check(
                            intercept, saved["intercept"][ai, fold], "inner intercept langsung"
                        )
                        check(head.coef_, saved["coef"][ai, fold], "inner coefficient refit")
                        check(
                            head.intercept_, saved["intercept"][ai, fold], "inner intercept refit"
                        )
                        direct = x[val] @ coef.T + intercept
                        replay = head.decision_function(x[val])
                        ds = check(
                            direct, saved["validation_scores"][ai, fold], "inner score langsung"
                        )
                        check(replay, saved["validation_scores"][ai, fold], "inner score refit")
                        if not np.array_equal(
                            direct.argmax(1), saved["validation_scores"][ai, fold].argmax(1)
                        ) or not np.array_equal(replay.argmax(1), direct.argmax(1)):
                            raise ResearchError("Prediksi inner berbeda.")
                        counts[ai, fold] = int((replay.argmax(1) == val_y).sum())
                        train_acc = float((head.predict(x[train]) == inner_y).mean())
                        if train_acc != saved["support_accuracy"][ai, fold]:
                            raise ResearchError("Akurasi dukungan inner berbeda.")
                        inner_checks.append(
                            {
                                "task": task["id"],
                                "alpha": alpha,
                                "validation_folder": split["validation_folder"],
                                "correct": int(counts[ai, fold]),
                                "n": 216,
                                "coef_max_abs_difference": dc,
                                "intercept_max_abs_difference": di,
                                "score_max_abs_difference": ds,
                            }
                        )
                if (
                    not np.array_equal(counts, saved["correct"])
                    or int(saved["selected_index"]) != task["selected_index"]
                ):
                    raise ResearchError("Jumlah benar/indeks pemenang berubah.")
            totals = counts.sum(1)
            selected = max(i for i, value in enumerate(totals) if value == totals.max())
            if (
                selected != task["selected_index"]
                or ALPHAS[selected] != task["selected_alpha"]
                or counts.tolist() != task["inner_correct"]
                or (totals / 1080).tolist() != task["inner_accuracy"]
            ):
                raise ResearchError(
                    "Pilihan alpha tidak berasal dari validasi inner yang ditetapkan."
                )
            row = conditions[task["id"]]
            if (
                row["selected_alpha"] != ALPHAS[selected]
                or row["inner_selected_accuracy"] != totals[selected] / 1080
            ):
                raise ResearchError("Head luar tidak memakai alpha terkunci.")
            with np.load(root / row["artifact"], allow_pickle=False) as saved:
                for key, actual in (
                    ("support", support),
                    ("test", test),
                    ("fit_y", labels),
                    ("test_y", y[test]),
                    ("classes", np.arange(36)),
                ):
                    if not np.array_equal(saved[key], actual):
                        raise ResearchError(f"Metadata outer berbeda: {key}")
                head = ridge(x[support], labels, ALPHAS[selected])
                coef, intercept = direct_ridge(x[support], labels, ALPHAS[selected])
                error = check(coef, saved["coef"], "outer coefficient langsung")
                check(intercept, saved["intercept"], "outer intercept langsung")
                check(head.coef_, saved["coef"], "outer refit coefficient")
                scores = head.decision_function(x[test])
                check(scores, saved["scores"], "outer score refit")
                check(x[test] @ coef.T + intercept, saved["scores"], "outer score langsung")
                fitted = head.decision_function(x[support])
                check(fitted, saved["support_scores"], "outer support score")
                if not np.array_equal(scores.argmax(1), saved["scores"].argmax(1)):
                    raise ResearchError("Prediksi outer tidak identik.")
                cpu_scores = cpu[task["encoder"]][test] @ saved["coef"].T + saved["intercept"]
            baseline_row = previous[task["heldout"], task["encoder"], task["permuted"]]
            with np.load(BASELINE / baseline_row["artifact"], allow_pickle=False) as saved:
                base = ridge(x[support], labels, 1.0)
                check(base.coef_, saved["coef"], "baseline alpha1 coefficient")
                baseline_scores = base.decision_function(x[test])
                check(baseline_scores, saved["scores"], "baseline alpha1 score")
                if not np.array_equal(baseline_scores.argmax(1), saved["scores"].argmax(1)):
                    raise ResearchError("Baseline alpha1 prediksi berbeda.")
            py, parent_values, parents = parent_scores(records, test, scores)
            _, cpu_parent, _ = parent_scores(records, test, cpu_scores)
            _, base_parent, _ = parent_scores(records, test, baseline_scores)
            if (
                metrics(y[test], scores) != row["test"]
                or metrics(py, parent_values) != row["parent"]
                or metrics(y[test], baseline_scores) != row["baseline_test"]
                or metrics(py, base_parent) != row["baseline_parent"]
                or float((fitted.argmax(1) == labels).mean()) != row["support_accuracy"]
                or row["difference_accuracy"]
                != row["test"]["accuracy"] - row["baseline_test"]["accuracy"]
            ):
                raise ResearchError("Metrik/selisih berpasangan tidak identik.")
            changed = int((scores.argmax(1) != cpu_scores.argmax(1)).sum())
            changed_parent = int((parent_values.argmax(1) != cpu_parent.argmax(1)).sum())
            if changed or changed_parent:
                cpu_changes.append(
                    {
                        "id": row["id"],
                        "clip_changes": changed,
                        "parent_changes": changed_parent,
                        "cpu_accuracy": float((cpu_scores.argmax(1) == y[test]).mean()),
                    }
                )
            head_checks.append(
                {
                    "id": row["id"],
                    "selected_alpha": row["selected_alpha"],
                    "maximum_coef_difference": error,
                    "cpu_clip_prediction_changes": changed,
                    "cpu_parent_prediction_changes": changed_parent,
                }
            )
            for unit, identities, truth, values, cpu_values, baseline_values in (
                (
                    "clip",
                    [records[i]["path"] for i in test],
                    y[test],
                    scores,
                    cpu_scores,
                    baseline_scores,
                ),
                ("parent", parents, py, parent_values, cpu_parent, base_parent),
            ):
                for identity, true, predicted, cpu_pred, base_pred in zip(
                    identities,
                    truth,
                    values.argmax(1),
                    cpu_values.argmax(1),
                    baseline_values.argmax(1),
                    strict=True,
                ):
                    predictions.append(
                        {
                            "task": row["id"],
                            "unit": unit,
                            "id": identity,
                            "true_key": KEYS[int(true)],
                            "predicted_key": KEYS[int(predicted)],
                            "cpu_feature_key": KEYS[int(cpu_pred)],
                            "baseline_key": KEYS[int(base_pred)],
                            "correct": int(true == predicted),
                            "baseline_correct": int(true == base_pred),
                        }
                    )
            print(f"Audited nested task {number}/48: {task['id']}", flush=True)
    for name, rows in (("inner-checks.csv", inner_checks), ("classifications.csv", predictions)):
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    verify_selection(root, signature)
    if inputs()[0] != signature:
        raise ResearchError("Input berubah selama audit.")
    audit = {
        "created_at": now(),
        "status": "passed",
        "code_sha256": file_hash(Path(__file__)),
        "run_results_sha256": file_hash(root / "results.json"),
        "selection_lock_sha256": lock_sha,
        "source_cpu_audit_sha256": file_hash(PREVIOUS_AUDIT / "audit.json"),
        "inner_heads_replayed": len(inner_checks),
        "outer_heads_replayed": len(head_checks),
        "baseline_heads_replayed": 48,
        "all_alpha_choices_match": True,
        "all_metrics_match": True,
        "inner_maximum_coef_difference": max(r["coef_max_abs_difference"] for r in inner_checks),
        "inner_maximum_score_difference": max(r["score_max_abs_difference"] for r in inner_checks),
        "head_checks": head_checks,
        "cpu_feature_prediction_changes": cpu_changes,
        "prediction_rows": len(predictions),
        "seconds": time.perf_counter() - started,
        "gpu_used": False,
        "artifact_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir())},
    }
    write_json(output / "audit.json", audit)
    print(f"Nested audit passed; CPU-feature changes in {len(cpu_changes)} tasks", flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/mka-nested")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/mka-nested-audit")
    args = parser.parse_args()
    try:
        run(args.run, args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
