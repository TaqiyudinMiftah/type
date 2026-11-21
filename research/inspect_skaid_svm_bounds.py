"""Post-hoc inspection of SVM intercept changes and dual coefficients near C."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.skaid_pilot_common import CACHE, ENCODERS, ROOT, verify
from ketik_audio.skaid_pilot_probe import permuted_labels
from ketik_audio.skaid_selection_core import transform
from ketik_audio.skaid_svm import OUTPUT as RUN
from ketik_audio.skaid_svm import load_data
from ketik_audio.skaid_svm_core import fit_selected, kernel_matrices, svc

NUMERIC = ROOT / "reports/skaid-svm-numeric"
SOURCE = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.9.1/sklearn/svm/src/libsvm/svm.cpp"


def pair_coefficients(head, i, j, length):
    starts = np.r_[0, np.cumsum(head.n_support_)]
    a, b = slice(starts[i], starts[i + 1]), slice(starts[j], starts[j + 1])
    result = np.zeros(length)
    result[head.support_[a]] = head.dual_coef_[j - 1, a]
    result[head.support_[b]] = head.dual_coef_[i, b]
    return result


def bounds(gram, signed, y, i, j, c):
    indices = np.flatnonzero((y == i) | (y == j))
    labels = np.where(y[indices] == i, 1, -1)
    alpha = np.abs(signed[indices])
    tolerance = 1e-10 * max(1, c)
    gy = gram[indices] @ signed - labels
    lower, upper = alpha <= tolerance, alpha >= c - tolerance
    lo = gy[(upper & (labels == 1)) | (lower & (labels == -1))]
    hi = gy[(upper & (labels == -1)) | (lower & (labels == 1))]
    interval = [-float(hi.min()), -float(lo.max())] if len(lo) and len(hi) else None
    gaps = c - alpha[(alpha > 0) & (alpha < c)]
    return {
        "strictly_free_coefficients": int(np.sum((alpha > 0) & (alpha < c))),
        "coefficients_at_C_exactly": int(np.sum(alpha == c)),
        "coefficients_positive_within_tolerance_of_C": int(
            np.sum((gaps > 0) & (gaps <= tolerance))
        ),
        "minimum_positive_C_minus_alpha": float(gaps.min()) if len(gaps) else None,
        "genuinely_free_at_diagnostic_tolerance": int(np.sum(~lower & ~upper)),
        "intercept_interval_from_rounded_bound_status": interval,
    }


def run(output):
    verify(NUMERIC, read_json(NUMERIC / "summary.json"))
    verify(RUN, read_json(RUN / "results.json"))
    fresh_directory(output)
    paths = [Path(__file__), NUMERIC / "summary.json", RUN / "results.json"]
    sig = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    write_json(output / "lock.json", {"created_at": now(), "input_sha256": sig})
    diagnostics = read_json(NUMERIC / "candidate-diagnostics.json")["diagnostics"]
    labels, _, conditions = load_data()
    cases = []
    pairs = [(i, j) for i in range(12) for j in range(i + 1, 12)]
    with (
        np.load(CACHE / "features.npz", allow_pickle=False) as features,
        threadpool_limits(limits=1),
    ):
        for encoder in ENCODERS:
            raw = features[encoder].astype(float)
            for d in diagnostics:
                if d["encoder"] != encoder:
                    continue
                p, recipe = conditions[d["condition_index"]], d["recipe"]
                s, q = p["support"], p["query"]
                y = permuted_labels(labels, p) if p["permuted"] else labels[s]
                primary, _, _, _, _, _, gamma = fit_selected(raw[s], y, raw[q], recipe)
                x, query, _, _ = transform(raw[s], raw[q], recipe["normalizer"])
                gram, _, _ = kernel_matrices(x, query, recipe)
                direct = svc(recipe["c"], recipe["kernel"], gamma if gamma else "scale").fit(x, y)
                pair = int(np.abs(primary.intercept_ - direct.intercept_).argmax())
                i, j = pairs[pair]
                a, b = (pair_coefficients(h, i, j, len(s)) for h in (primary, direct))
                cases.append(
                    {
                        "encoder": encoder,
                        "condition_index": d["condition_index"],
                        "recipe_index": d["recipe_index"],
                        "selected_for_own_held_id": d["selected_for_own_held_id"],
                        "pair": [i, j],
                        "c": recipe["c"],
                        "intercept_difference": float(
                            abs(primary.intercept_[pair] - direct.intercept_[pair])
                        ),
                        "maximum_pair_coefficient_difference": float(np.abs(a - b).max()),
                        "primary_intercept": float(primary.intercept_[pair]),
                        "direct_intercept": float(direct.intercept_[pair]),
                        "primary": bounds(gram, a, y, i, j, recipe["c"]),
                        "direct": bounds(gram, b, y, i, j, recipe["c"]),
                    }
                )
    write_json(output / "cases.json", {"cases": cases})
    if any(file_hash(ROOT / p) != digest for p, digest in sig.items()):
        raise ResearchError("Masukan diagnosis berubah.")
    record = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "kind": "post-hoc diagnostic; no model correction or recipe reselection",
        "libsvm_primary_source": SOURCE,
        "source_locations": {"update_alpha_status": 558, "calculate_rho": 1069},
        "diagnostic_relative_bound_tolerance": 1e-10,
        "heads_examined": len(cases),
        "largest_intercept_difference_case": max(cases, key=lambda d: d["intercept_difference"]),
        "cases_with_different_strict_free_counts": sum(
            d["primary"]["strictly_free_coefficients"] != d["direct"]["strictly_free_coefficients"]
            for d in cases
        ),
        "cases_with_no_genuinely_free_coefficients_both": sum(
            d["primary"]["genuinely_free_at_diagnostic_tolerance"] == 0
            and d["direct"]["genuinely_free_at_diagnostic_tolerance"] == 0
            for d in cases
        ),
        "model_updates": 0,
    }
    record["artifact_sha256"] = {
        str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    write_json(output / "summary.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/skaid-svm-bounds")
    args = parser.parse_args()
    try:
        run(args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
