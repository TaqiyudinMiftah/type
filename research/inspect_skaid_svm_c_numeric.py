"""Summarize which expanded-C numerical differences reached locked predictions."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from audit_skaid_selection import independent_transform
from audit_skaid_svm import independent_median
from inspect_skaid_svm_bounds import bounds, pair_coefficients
from threadpoolctl import threadpool_limits

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.skaid_pilot_common import CACHE, ENCODERS, ROOT, verify
from ketik_audio.skaid_svm_c import OUTPUT as RUN
from ketik_audio.skaid_svm_c import artifacts, choice_key, condition_labels, fixed_data
from ketik_audio.skaid_svm_c_core import OLD_RECIPES, RECIPES, fit_selected, svc
from ketik_audio.skaid_svm_core import kernel_matrices

AUDIT = ROOT / "reports/skaid-svm-c-audit"
OUTPUT = ROOT / "reports/skaid-svm-c-numeric"


def outlier_rows():
    with (AUDIT / "validation-checks.csv").open(encoding="utf-8") as f:
        return [
            r
            for r in csv.DictReader(f)
            if r["within_score_tolerance"] == "False" or int(r["prediction_changes"])
        ]


def run(output):
    verify(RUN, read_json(RUN / "results.json"))
    audit = read_json(AUDIT / "audit.json")
    verify(AUDIT, audit)
    labels, _, conditions = fixed_data()
    choices = {choice_key(c): c for c in read_json(RUN / "selection.json")["choices"]}
    paths = [
        Path(__file__),
        RUN / "results.json",
        AUDIT / "audit.json",
        ROOT / "research/inspect_skaid_svm_bounds.py",
        ROOT / "research/audit_skaid_svm.py",
        ROOT / "research/audit_skaid_selection.py",
    ]
    sig = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    fresh_directory(output)
    write_json(output / "lock.json", {"created_at": now(), "input_sha256": sig})
    rows = outlier_rows()
    diagnostics = []
    pairs = [(i, j) for i in range(12) for j in range(i + 1, 12)]
    with (
        np.load(CACHE / "features.npz", allow_pickle=False) as features,
        threadpool_limits(limits=1),
    ):
        for encoder in ENCODERS:
            raw = features[encoder].astype(float)
            family = encoder.split("_")[0]
            with np.load(RUN / "validation" / f"{encoder}.npz", allow_pickle=False) as z:
                saved = z["query_scores"]
            for r in rows:
                if r["encoder"] != encoder:
                    continue
                ci, ri = int(r["condition_index"]), int(r["recipe_index"])
                p, recipe = conditions[ci], RECIPES[ri]
                s, q, y = p["support"], p["query"], condition_labels(labels, p)
                primary, scores, *_ = fit_selected(raw[s], y, raw[q], recipe)
                if not np.array_equal(scores, saved[ci, ri - len(OLD_RECIPES)]):
                    raise ResearchError("Diagnosis tidak mereproduksi kandidat utama.")
                x, query, _, _ = independent_transform(raw[s], raw[q], recipe["normalizer"])
                g = recipe["multiplier"]
                med = independent_median(x)
                gamma = "scale" if g is None else (g / med if med > 1e-12 else g)
                direct = svc(recipe["c"], recipe["kernel"], gamma).fit(x, y)
                direct_scores = direct.decision_function(query)
                if not np.isclose(
                    np.abs(scores - direct_scores).max(), float(r["max_score_error"])
                ):
                    raise ResearchError("Selisih diagnosis berbeda dari audit.")
                intercept_delta = np.abs(primary.intercept_ - direct.intercept_)
                pair = int(intercept_delta.argmax())
                i, j = pairs[pair]
                a, b = (pair_coefficients(h, i, j, len(s)) for h in (primary, direct))
                gram, cross, _ = kernel_matrices(x, query, recipe)
                primary.decision_function_shape = direct.decision_function_shape = "ovo"
                ovo_a, ovo_b = primary.decision_function(cross), direct.decision_function(query)
                selected = choices[recipe["kernel"], family, p["subject"], p["k"], p["permuted"]]
                diagnostics.append(
                    {
                        "encoder": encoder,
                        "family": family,
                        "condition_index": ci,
                        "subject": p["subject"],
                        "k": p["k"],
                        "draw": p["draw"],
                        "permuted": p["permuted"],
                        "recipe_index": ri,
                        "recipe": recipe,
                        "selected_for_own_held_id": selected["recipe_index"] == ri,
                        "max_ovr_score_error": float(r["max_score_error"]),
                        "max_ovo_margin_error": float(np.abs(ovo_a - ovo_b).max()),
                        "binary_sign_changes": int(np.sum((ovo_a < 0) != (ovo_b < 0))),
                        "top1_changes": int(r["prediction_changes"]),
                        "primary_correct": int(r["primary_correct"]),
                        "direct_correct": int(r["direct_correct"]),
                        "pair": [i, j],
                        "max_intercept_error": float(intercept_delta[pair]),
                        "primary_intercept": float(primary.intercept_[pair]),
                        "direct_intercept": float(direct.intercept_[pair]),
                        "max_pair_coefficient_error": float(np.abs(a - b).max()),
                        "primary_bounds": bounds(gram, a, y, i, j, recipe["c"]),
                        "direct_bounds": bounds(gram, b, y, i, j, recipe["c"]),
                    }
                )
    if len(diagnostics) != len(rows):
        raise ResearchError("Coverage diagnosis outlier tidak lengkap.")
    changes = read_json(AUDIT / "prediction-changes.json")["changes"]
    write_json(output / "candidate-diagnostics.json", {"diagnostics": diagnostics})
    write_json(output / "final-prediction-changes.json", {"changes": changes})
    if any(file_hash(ROOT / p) != digest for p, digest in sig.items()):
        raise ResearchError("Masukan diagnosis berubah.")
    record = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "new_candidate_heads_diagnosed": len(diagnostics),
        "outlier_counts_by_family": dict(Counter(d["family"] for d in diagnostics)),
        "outlier_counts_by_kernel": dict(Counter(d["recipe"]["kernel"] for d in diagnostics)),
        "outlier_heads_selected_for_own_id": sum(
            d["selected_for_own_held_id"] for d in diagnostics
        ),
        "cases_with_binary_sign_changes": sum(d["binary_sign_changes"] > 0 for d in diagnostics),
        "largest_score_difference": max(
            diagnostics, key=lambda d: d["max_ovr_score_error"], default=None
        ),
        "diagnostic_direct_choices_changed": audit["diagnostic_direct_choices_changed"],
        "cpu_change_counts_by_encoder": dict(Counter(r["encoder"] for r in changes)),
        "normal_cpu_accuracy_changes": sum(
            not r["permuted"] and r["primary_correct"] != r["cpu_correct"] for r in changes
        ),
        "control_cpu_accuracy_changes": sum(
            r["permuted"] and r["primary_correct"] != r["cpu_correct"] for r in changes
        ),
        "locked_choices_updated": False,
        "parameter_updates": 0,
        "artifact_sha256": artifacts(output),
    }
    write_json(output / "summary.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        run(args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
