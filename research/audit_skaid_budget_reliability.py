"""Count observed threshold attainment in the frozen SKAID evaluation grid."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from itertools import combinations, product
from pathlib import Path
from statistics import fmean, median

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (1, 2, 3, 5, 10)
DRAWS = (101, 102, 103)
METHODS = ("ridge", "linear", "rbf")
EXAMPLES = 60
THRESHOLDS = {80: 48, 90: 54}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def cell(row):
    return (row["subject"], row["k"], row["draw"])


def verified_runs():
    files = {str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__))}
    ranges, condition_path, budgets_hash, rows = {}, None, None, []
    for name in ("skaid-selection", "skaid-svm-c"):
        run = ROOT / "runs" / name
        audit_path = ROOT / "reports" / f"{name}-audit" / "audit.json"
        results_path = run / "results.json"
        audit, results = read(audit_path), read(results_path)
        require(audit["status"] == results["status"] == "complete", f"Run belum lengkap: {name}")

        def record(path, expected=None):
            key = str(path.relative_to(ROOT))
            digest = sha(path)
            require(expected is None or digest == expected, f"Hash berubah: {key}")
            files[key] = digest
            return digest

        record(audit_path)
        record(results_path, audit["input_sha256"][str(results_path.relative_to(ROOT))])
        for item in ("conditions.json", "budgets.json", "trials.json"):
            record(run / item, results["artifact_sha256"][item])
        if condition_path is None:
            condition_path = run / "conditions.json"
            budgets_hash = results["artifact_sha256"]["budgets.json"]
        else:
            require(
                files[str(condition_path.relative_to(ROOT))]
                == results["artifact_sha256"]["conditions.json"],
                "Pembanding memakai pembagian data lain.",
            )
            require(
                budgets_hash == results["artifact_sha256"]["budgets.json"], "Budget sumber lain."
            )
        trials = read(run / "trials.json")["trials"]
        require(len(trials) == results["outer_refits"], f"Jumlah fit berbeda: {name}")
        rows.extend(t for t in trials if t["encoder"] == "mfcc" and not t["permuted"])
        ranges[name] = len(trials)
    return read(condition_path)["conditions"], rows, files, ranges


def group_counts(conditions, trials):
    by_split = {cell(c): c for c in conditions if not c["permuted"]}
    subjects = sorted({s for s, _, _ in by_split})
    require(len(subjects) == 17, "Cohort SKAID berubah.")
    require(by_split.keys() == set(product(subjects, BUDGETS, DRAWS)), "Grid split berubah.")
    require(sum(not c["permuted"] for c in conditions) == len(by_split), "Split ganda.")
    all_query_events = set()
    for subject in subjects:
        query_order = by_split[subject, BUDGETS[0], DRAWS[0]]["query"]
        query = set(query_order)
        require(len(query_order) == len(query) == EXAMPLES, "Query tidak unik atau tidak lengkap.")
        require(not query & all_query_events, "Query dipakai silang ID.")
        all_query_events |= query
        for draw in DRAWS:
            previous = set()
            for k in BUDGETS:
                split = by_split[subject, k, draw]
                support = set(split["support"])
                require(len(support) == len(split["support"]) == 12 * k, "Anggaran target berbeda.")
                require(
                    split["query"] == query_order and not query & support, "Query berubah/overlap."
                )
                require(previous < support, "Dukungan bukan subset dari K berikutnya.")
                previous = support

    counts, costs = {}, {k: set() for k in BUDGETS}
    labels_by_cell = {}
    for row in trials:
        method, seed = row.get("policy", "ridge"), cell(row)
        split = by_split[seed]
        token = row["session_token"]
        require(split["session_token"] == token, "ID sesi trial berbeda dari split.")
        if "condition_index" in row:
            require(conditions[row["condition_index"]] == split, "Referensi split SVM berbeda.")
        k = row["k"]
        require(row["target_classifier_labels"] == 12 * k, "Jumlah label target berbeda.")
        require(row["target_validation_labels"] == 0, "Ada label validasi target per fit.")
        source_cost = row["source_labels_unique_total"]
        require(
            labels_by_cell.setdefault(seed, source_cost) == source_cost, "Biaya sumber berbeda."
        )
        costs[k].add(source_cost)
        metrics = row["query"]
        wins = metrics["correct"]
        require(type(wins) is int and 0 <= wins <= EXAMPLES, "Jumlah prediksi benar tidak valid.")
        require(metrics["examples"] == EXAMPLES, "Ukuran query tidak cocok.")
        require(
            math.isclose(metrics["accuracy"], wins / EXAMPLES), "Akurasi berbeda dari hitungan."
        )
        require((method, *seed) not in counts, "Hasil terduplikasi.")
        counts[method, *seed] = wins
    require(counts.keys() == set(product(METHODS, subjects, BUDGETS, DRAWS)), "Grid hasil berubah.")
    return counts, subjects, costs, len(all_query_events)


def summarize(counts, subjects, costs, unique_queries):
    output, hit = {}, {}
    for method in METHODS:
        vectors = {
            (s, k): [counts[method, s, k, d] for d in DRAWS] for s in subjects for k in BUDGETS
        }
        curves = []
        for k in BUDGETS:
            per_id = {s: vectors[s, k] for s in subjects}
            checks = {}
            for pct, cut in THRESHOLDS.items():
                consistent = {s for s, xs in per_id.items() if min(xs) >= cut}
                margin = {s for s, xs in per_id.items() if min(xs) > cut}
                hit[method, k, pct] = consistent
                hit[method, k, pct, "one_error_margin"] = margin
                checks[str(pct)] = {
                    "at_least_once": sum(max(xs) >= cut for xs in per_id.values()),
                    "mean_of_draws": sum(sum(xs) >= len(DRAWS) * cut for xs in per_id.values()),
                    "every_draw": len(consistent),
                    "every_draw_even_one_more_error": len(margin),
                }
            rate = fmean(fmean(xs) / EXAMPLES for xs in per_id.values())
            span = [(max(xs) - min(xs)) / EXAMPLES for xs in per_id.values()]
            curves.append(
                {
                    "k": k,
                    "target_labels_per_fit": 12 * k,
                    "cohort_accuracy": rate,
                    "median_draw_range": median(span),
                    "maximum_draw_range": max(span),
                    "thresholds": checks,
                }
            )

        def crossing(method_name, threshold):
            first, sustained = Counter(), Counter()
            for s in subjects:
                met = [s in hit[method_name, k, threshold] for k in BUDGETS]
                first[
                    str(next((k for k, fits in zip(BUDGETS, met, strict=True) if fits), None))
                ] += 1
                stable = next((k for i, k in enumerate(BUDGETS) if all(met[i:])), None)
                sustained[str(stable)] += 1

            def ordered(count):
                out = {str(k): count[str(k)] for k in BUDGETS}
                return {**out, "not_observed": count["None"]}

            return {"first": ordered(first), "sustained_to_k10": ordered(sustained)}

        output[method] = {
            "curve": curves,
            "first_and_sustained_crossings": {
                str(pct): crossing(method, pct) for pct in THRESHOLDS
            },
            "k5_to_k10": {
                "mean_decreased_ids": sum(
                    sum(vectors[s, 10]) < sum(vectors[s, 5]) for s in subjects
                ),
                "at_least_one_draw_decreased_ids": sum(
                    any(a < b for a, b in zip(vectors[s, 10], vectors[s, 5], strict=True))
                    for s in subjects
                ),
            },
        }

    overlap = []
    for a, b in combinations(METHODS, 2):
        for k, pct in product(BUDGETS, THRESHOLDS):
            left, right = hit[a, k, pct], hit[b, k, pct]
            overlap.append(
                {
                    "a": a,
                    "b": b,
                    "k": k,
                    "threshold": pct,
                    "both": len(left & right),
                    "a_only": len(left - right),
                    "b_only": len(right - left),
                }
            )
    return {
        "cohort_ids": len(subjects),
        "unique_query_events": unique_queries,
        "query_events_per_id": EXAMPLES,
        "draws": list(DRAWS),
        "correct_counts_at_threshold": THRESHOLDS,
        "source_label_range_by_k": {str(k): [min(costs[k]), max(costs[k])] for k in BUDGETS},
        "methods": output,
        "paired_method_overlap": overlap,
        "k10_all_methods_stable_80": len(set.intersection(*(hit[m, 10, 80] for m in METHODS))),
        "k10_all_methods_stable_80_with_margin": len(
            set.intersection(*(hit[m, 10, 80, "one_error_margin"] for m in METHODS))
        ),
    }


def main():
    cases, trials, sources, runs = verified_runs()
    counts, subjects, costs, queries = group_counts(cases, trials)
    result = summarize(counts, subjects, costs, queries)
    result.update(
        input_sha256=sources, source_fit_records=runs, evaluated_count_records=len(counts)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
