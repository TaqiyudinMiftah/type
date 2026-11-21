"""Diagnose direct/precomputed SVM discrepancies without changing locked choices."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.skaid_pilot_common import CACHE, ENCODERS, ROOT, verify
from ketik_audio.skaid_pilot_probe import permuted_labels
from ketik_audio.skaid_selection_core import transform
from ketik_audio.skaid_svm import OUTPUT as RUN
from ketik_audio.skaid_svm import aggregate_tables, load_data
from ketik_audio.skaid_svm_core import (
    RECIPES,
    fit_selected,
    kernel_matrices,
    select_from_table,
    svc,
)

AUDIT = ROOT / "reports/skaid-svm-audit"


def run(output):
    verify(RUN, read_json(RUN / "results.json"))
    audit = read_json(AUDIT / "audit.json")
    verify(AUDIT, audit)
    fresh_directory(output)
    paths = [Path(__file__), AUDIT / "audit.json", RUN / "results.json", RUN / "selection.json"]
    sig = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    write_json(output / "lock.json", {"created_at": now(), "input_sha256": sig})
    labels, _, conditions = load_data()
    choices = read_json(RUN / "selection.json")["choices"]
    tables = aggregate_tables(RUN, conditions)
    direct_tables = {k: np.zeros_like(v) for k, v in tables.items()}
    outliers = []
    with (AUDIT / "validation-checks.csv").open() as f:
        for row in csv.DictReader(f):
            p, family = conditions[int(row["condition_index"])], row["encoder"].split("_")[0]
            ri = int(row["recipe_index"])
            direct_tables[family, p["k"], p["permuted"]][p["subject"], ri] += int(
                row["direct_correct"]
            )
            if row["within_score_tolerance"] == "False" or int(row["prediction_changes"]):
                outliers.append(row)
    changed_choices = []
    for c in choices:
        alternative = select_from_table(
            direct_tables[c["family"], c["k"], c["permuted"]], c["held"], c["policy"]
        )
        if alternative["recipe_index"] != c["recipe_index"]:
            changed_choices.append(
                {
                    "choice_id": c["id"],
                    "primary_recipe_index": c["recipe_index"],
                    "direct_recipe_index": alternative["recipe_index"],
                }
            )
    diagnostics = []
    with (
        np.load(CACHE / "features.npz", allow_pickle=False) as features,
        threadpool_limits(limits=1),
    ):
        for encoder in ENCODERS:
            raw = features[encoder].astype(float)
            with np.load(RUN / "validation" / f"{encoder}.npz", allow_pickle=False) as z:
                stored_scores = z["query_scores"]
            for row in outliers:
                if row["encoder"] != encoder:
                    continue
                ci, ri = int(row["condition_index"]), int(row["recipe_index"])
                p, recipe = conditions[ci], RECIPES[ri]
                s, q = p["support"], p["query"]
                y = permuted_labels(labels, p) if p["permuted"] else labels[s]
                head, scores, _, _, _, median, gamma = fit_selected(raw[s], y, raw[q], recipe)
                if not np.array_equal(scores, stored_scores[ci, ri]):
                    raise ResearchError("Refit diagnosis berbeda dari kandidat asli.")
                x, query, _, _ = transform(raw[s], raw[q], recipe["normalizer"])
                _, cross, _ = kernel_matrices(x, query, recipe)
                direct = svc(recipe["c"], recipe["kernel"], gamma if gamma else "scale").fit(x, y)
                direct_scores = direct.decision_function(query)
                head.decision_function_shape = direct.decision_function_shape = "ovo"
                primary_margins, direct_margins = (
                    head.decision_function(cross),
                    direct.decision_function(query),
                )
                sign_changes = (primary_margins < 0) != (direct_margins < 0)
                selected = [
                    c
                    for c in choices
                    if c["family"] == encoder.split("_")[0]
                    and c["held"] == p["subject"]
                    and c["k"] == p["k"]
                    and c["permuted"] == p["permuted"]
                    and c["recipe_index"] == ri
                ]
                diagnostics.append(
                    {
                        "encoder": encoder,
                        "condition_index": ci,
                        "subject": p["subject"],
                        "k": p["k"],
                        "draw": p["draw"],
                        "permuted": p["permuted"],
                        "recipe_index": ri,
                        "recipe": recipe,
                        "selected_for_own_held_id": bool(selected),
                        "median": median,
                        "gamma": gamma,
                        "max_ovr_score_error": float(np.max(np.abs(scores - direct_scores))),
                        "max_ovo_margin_error": float(
                            np.max(np.abs(primary_margins - direct_margins))
                        ),
                        "binary_sign_changes": int(sign_changes.sum()),
                        "max_primary_abs_margin_at_sign_change": float(
                            np.abs(primary_margins[sign_changes]).max()
                        )
                        if sign_changes.any()
                        else None,
                        "prediction_changes": int(
                            np.sum(scores.argmax(1) != direct_scores.argmax(1))
                        ),
                        "primary_correct": int(np.sum(scores.argmax(1) == labels[q])),
                        "direct_correct": int(np.sum(direct_scores.argmax(1) == labels[q])),
                        "primary_max_n_iter": int(head.n_iter_.max()),
                        "direct_max_n_iter": int(direct.n_iter_.max()),
                        "max_intercept_difference": float(
                            np.abs(head.intercept_ - direct.intercept_).max()
                        ),
                        "same_support_positions": bool(
                            np.array_equal(head.support_, direct.support_)
                        ),
                        "max_dual_coefficient_difference_if_same_support": float(
                            np.abs(head.dual_coef_ - direct.dual_coef_).max()
                        )
                        if np.array_equal(head.support_, direct.support_)
                        else None,
                    }
                )
    checks = read_json(AUDIT / "final-checks.json")["checks"]
    final_summary = {}
    for policy in ("linear", "rbf"):
        for permuted in (False, True):
            rows = [r for r in checks if r["policy"] == policy and r["permuted"] == permuted]
            final_summary[f"{policy}-{'control' if permuted else 'normal'}"] = {
                "conditions": len(rows),
                "cpu_prediction_changes": sum(r["cpu_prediction_changes"] for r in rows),
                "dual_prediction_changes": sum(r["dual_prediction_changes"] for r in rows),
                "cpu_accuracy_changed_conditions": sum(
                    r["primary_correct"] != r["cpu_correct"] for r in rows
                ),
                "cpu_minus_primary_correct": sum(
                    r["cpu_correct"] - r["primary_correct"] for r in rows
                ),
            }
    write_json(output / "candidate-diagnostics.json", {"diagnostics": diagnostics})
    record = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "candidate_heads_outside_score_tolerance": audit["candidate_heads_outside_score_tolerance"],
        "candidate_prediction_changes": audit["candidate_direct_prediction_disagreements"],
        "outlier_heads_selected_for_own_held_id": sum(
            d["selected_for_own_held_id"] for d in diagnostics
        ),
        "source_correct_count_entries_changed": sum(
            int(np.sum(tables[k] != direct_tables[k])) for k in tables
        ),
        "diagnostic_source_choices_that_would_change_with_direct_counts": changed_choices,
        "final_numeric_summary": final_summary,
        "locked_choices_updated": False,
        "parameter_updates": 0,
    }
    record["artifact_sha256"] = {
        str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    if any(file_hash(ROOT / p) != digest for p, digest in sig.items()):
        raise ResearchError("Masukan diagnosis berubah.")
    write_json(output / "summary.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/skaid-svm-numeric")
    args = parser.parse_args()
    try:
        run(args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
