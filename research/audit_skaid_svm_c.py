"""Audit the added C grid and replay all locked heads without SVC prediction."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from audit_skaid_selection import independent_transform
from audit_skaid_svm import independent_median
from threadpoolctl import threadpool_limits

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.skaid_pilot_common import CACHE, DATA, ENCODERS, KEYS, KS, ROOT, verify
from ketik_audio.skaid_pilot_probe import metrics
from ketik_audio.skaid_selection_core import NORMALIZERS
from ketik_audio.skaid_svm import RIDGE
from ketik_audio.skaid_svm_c import (
    OLD,
    OLD_AUDIT,
    aggregate_tables,
    artifacts,
    choice_key,
    condition_labels,
    fixed_data,
    signature,
)
from ketik_audio.skaid_svm_c import OUTPUT as RUN
from ketik_audio.skaid_svm_c_core import (
    NEW_RECIPES,
    OLD_RECIPES,
    RECIPES,
    checked_head,
    fit_selected,
    select_from_table,
    svc,
)
from ketik_audio.skaid_svm_core import POLICIES, kernel_matrices, replay

OUTPUT = ROOT / "reports/skaid-svm-c-audit"
CPU_CACHE = ROOT / "reports/skaid-pilot-audit/cpu-features.npz"


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def max_error(a, b):
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ResearchError("Audit menemukan skor/parameter tidak finite.")
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def old_direct_counts(conditions):
    old = {e: np.full((len(conditions), len(OLD_RECIPES)), -1, dtype=int) for e in ENCODERS}
    with (OLD_AUDIT / "validation-checks.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            e, ci, ri = row["encoder"], int(row["condition_index"]), int(row["recipe_index"])
            if old[e][ci, ri] != -1:
                raise ResearchError("Audit lama menduplikasi kandidat.")
            old[e][ci, ri] = int(row["direct_correct"])
    if any(np.any((v < 0) | (v > 60)) for v in old.values()):
        raise ResearchError("Coverage audit lama tidak lengkap.")
    return old


def audit_budgets(plans):
    budgets = read_json(RUN / "budgets.json")["budgets"]
    if budgets != read_json(OLD / "budgets.json")["budgets"] or len(budgets) != 17 * len(KS):
        raise ResearchError("Biaya label berbeda dari baseline.")
    for b in budgets:
        source = [p for p in plans if p["subject"] != b["held"] and p["k"] == b["k"]]
        s = {i for p in source for i in p["support"]}
        q = {i for p in source for i in p["query"]}
        held = {i for p in plans if p["subject"] == b["held"] for i in p["support"] + p["query"]}
        if (
            len(source) != 48
            or len(q) != 960
            or s & q
            or (s | q) & held
            or b["source_support_indices"] != sorted(s)
            or b["source_query_indices"] != sorted(q)
            or b["source_labels_unique_total"] != len(s | q)
            or b["target_classifier_labels_per_trial"] != 12 * b["k"]
        ):
            raise ResearchError("Union indeks biaya label tidak valid.")
    return {(b["held"], b["k"]): b["source_labels_unique_total"] for b in budgets}


def independent_winner(totals, policy):
    return min(
        (i for i, r in enumerate(RECIPES) if r["kernel"] == policy),
        key=lambda i: (
            -totals[i],
            RECIPES[i]["c"],
            RECIPES[i]["multiplier"] or 0,
            NORMALIZERS.index(RECIPES[i]["normalizer"]),
        ),
    )


def audit_choices(choices, primary, direct, costs):
    checks, alternatives = [], []
    for c in choices:
        h, policy, key = c["held"], c["policy"], (c["family"], c["k"], c["permuted"])
        dev = [i for i in range(17) if i != h]
        table = primary[key]
        totals = table[dev].sum(0)
        ri = independent_winner(totals, policy)
        perturb = table.copy()
        perturb[h] = np.arange(len(RECIPES), dtype=np.int64) * 10**8
        if (
            c["source_ids"] != dev
            or c["source_correct_by_recipe"] != totals.tolist()
            or c["selected_source_correct"] != totals[ri]
            or c["recipe_index"] != ri
            or c["recipe"] != RECIPES[ri]
            or select_from_table(perturb, h, policy) != select_from_table(table, h, policy)
            or c["source_labels_unique_total"] != costs[h, c["k"]]
        ):
            raise ResearchError("Seleksi, counterfactual held atau biaya tidak cocok.")
        alternate = independent_winner(direct[key][dev].sum(0), policy)
        if ri != alternate:
            alternatives.append(
                {"choice_id": c["id"], "selected_recipe": ri, "direct_recipe": alternate}
            )
        checks.append({"choice_id": c["id"], "held_excluded": True, "recipe_replayed": ri})
    return checks, alternatives


def candidate_checks(raw, labels, conditions, encoder, new, writer, stats):
    direct_counts = np.empty_like(new["query_correct"])
    for ci, p in enumerate(conditions):
        s, q, y = p["support"], p["query"], condition_labels(labels, p)
        for ni, mode in enumerate(NORMALIZERS):
            x, query, mean, scale = independent_transform(raw[s], raw[q], mode)
            median = independent_median(x)
            for key, actual in (("mean", mean), ("scale", scale), ("median", median)):
                stored = new[key][ci, ni]
                stats[f"max_{key}_error"] = max(
                    stats[f"max_{key}_error"], max_error(actual, stored)
                )
                if not np.allclose(actual, stored, atol=1e-12, rtol=1e-10):
                    raise ResearchError("Normalisasi atau bandwidth tidak support-only.")
            for col, recipe in enumerate(NEW_RECIPES):
                if mode != recipe["normalizer"]:
                    continue
                g = recipe["multiplier"]
                gamma = 0.0 if g is None else (g / median if median > 1e-12 else g)
                if not np.isclose(gamma, new["gamma"][ci, col], atol=1e-12, rtol=1e-10):
                    raise ResearchError("Gamma kandidat berbeda dari median dukungan.")
                head = checked_head(
                    svc(recipe["c"], recipe["kernel"], gamma if gamma else "scale").fit(x, y)
                )
                expected = new["query_scores"][ci, col]
                scores = head.decision_function(query)
                support = head.decision_function(x).argmax(1)
                primary, actual = expected.argmax(1), scores.argmax(1)
                correct, direct_correct = (
                    int(np.sum(primary == labels[q])),
                    int(np.sum(actual == labels[q])),
                )
                if correct != new["query_correct"][ci, col] or new["fit_status"][ci, col] != 0:
                    raise ResearchError("Count/status kandidat tersimpan tidak konsisten.")
                direct_counts[ci, col] = direct_correct
                err = max_error(scores, expected)
                outlier = not np.allclose(scores, expected, atol=1e-5, rtol=1e-5)
                flips = int(np.sum(primary != actual))
                support_changed = (
                    np.sum(support == y) != new["support_fit_correct"][ci, col]
                    or np.sum(support == labels[s]) != new["support_true_correct"][ci, col]
                )
                stats["new_candidate_heads_checked"] += 1
                stats["new_candidate_direct_prediction_changes"] += flips
                stats["new_candidate_heads_outside_score_tolerance"] += int(outlier)
                stats["new_candidate_heads_with_support_count_difference"] += int(support_changed)
                stats["max_direct_score_error"] = max(stats["max_direct_score_error"], err)
                stats["max_direct_binary_iterations"] = max(
                    stats["max_direct_binary_iterations"], int(head.n_iter_.max())
                )
                writer.writerow(
                    [
                        encoder,
                        ci,
                        len(OLD_RECIPES) + col,
                        correct,
                        direct_correct,
                        flips,
                        err,
                        not outlier,
                        bool(support_changed),
                        int(head.n_iter_.max()),
                    ]
                )
        if (ci + 1) % 51 == 0:
            print(f"Audit kernel {encoder}: {ci + 1}/{len(conditions)}", flush=True)
    return direct_counts


def final_checks(
    encoder,
    output,
    raw,
    cpu_raw,
    labels,
    records,
    conditions,
    by_trial,
    chosen,
    old_validation,
    validation,
    writer,
    stats,
):
    final = load_npz(RUN / "final" / f"{encoder}.npz")
    old = load_npz(OLD / "final" / f"{encoder}.npz")
    with np.load(RIDGE / "final" / f"{encoder}.npz", allow_pickle=False) as z:
        ridge = z["query_scores"]
    family = encoder.split("_")[0]
    cpu_scores = np.empty_like(final["query_scores"])
    checks, changes = [], []
    for pi, policy in enumerate(POLICIES):
        for ci, p in enumerate(conditions):
            i = pi * len(conditions) + ci
            trial = by_trial[encoder, i]
            c = chosen[policy, family, p["subject"], p["k"], p["permuted"]]
            s, q, y = p["support"], p["query"], condition_labels(labels, p)
            ri, recipe = c["recipe_index"], c["recipe"]
            if (
                trial["choice_id"] != c["id"]
                or final["recipe_index"][i] != ri
                or not np.array_equal(final["query"][i], q)
                or not np.array_equal(final["support"][i, : len(s)], s)
                or not np.array_equal(final["fit_labels"][i, : len(s)], y)
                or trial["target_classifier_labels"] != 12 * p["k"]
                or trial["target_validation_labels"] != 0
                or trial["source_labels_unique_total"] != c["source_labels_unique_total"]
            ):
                raise ResearchError("Pilihan, split atau biaya final berbeda.")
            scores = final["query_scores"][i]
            expected = (
                old_validation["query_scores"][ci, ri]
                if ri < len(OLD_RECIPES)
                else validation["query_scores"][ci, ri - len(OLD_RECIPES)]
            )
            if not np.array_equal(scores, expected) or metrics(labels[q], scores) != trial["query"]:
                raise ResearchError("Refit final berbeda dari skor kandidat atau metrik.")
            if ri == c["old_recipe_index"]:
                if any(
                    not np.array_equal(v[i], old[k][i], equal_nan=True) for k, v in final.items()
                ):
                    raise ResearchError("Resep lama yang dipertahankan berubah.")
                stats["unchanged_recipe_refits_exact"] += 1
            x, query, mean, scale = independent_transform(raw[s], raw[q], recipe["normalizer"])
            gram, cross, gamma = kernel_matrices(x, query, recipe)
            checks_stats = (
                (mean, final["mean"][i]),
                (scale, final["scale"][i]),
                (gamma, final["gamma"][i]),
                (independent_median(x), final["median"][i]),
            )
            if not all(np.allclose(a, b, atol=1e-12, rtol=1e-10) for a, b in checks_stats):
                raise ResearchError("Scaler/gamma final berbeda dari dukungan.")
            counts = final["n_support"][i]
            ns = int(counts.sum())
            pos = final["support_positions"][i, :ns]
            if (
                len(np.unique(pos)) != ns
                or np.any((pos < 0) | (pos >= len(s)))
                or not np.array_equal(y[pos], np.repeat(np.arange(12), counts))
            ):
                raise ResearchError("Urutan support vector final tidak valid.")
            dual_args = pos, final["dual_coef"][i, :, :ns], final["intercept"][i], counts
            dual_scores = replay(cross, *dual_args)
            stored_support = final["support_scores"][i, : len(s)]
            dual_support = replay(gram, *dual_args)
            _, cpu, _, _, _, _, _ = fit_selected(cpu_raw[s], y, cpu_raw[q], recipe)
            cpu_scores[i] = cpu
            dual_error = max(
                max_error(dual_scores, scores), max_error(dual_support, stored_support)
            )
            stats["max_final_dual_score_error"] = max(
                stats["max_final_dual_score_error"], dual_error
            )
            stats["max_cpu_feature_score_error"] = max(
                stats["max_cpu_feature_score_error"], max_error(cpu, scores)
            )
            if (
                float(np.mean(stored_support.argmax(1) == y)) != trial["support_fit_label_accuracy"]
                or float(np.mean(stored_support.argmax(1) == labels[s]))
                != trial["support_true_label_accuracy"]
            ):
                raise ResearchError("Metrik dukungan final berbeda.")
            main, dual, cpu_pred = (z.argmax(1) for z in (scores, dual_scores, cpu))
            prev, reference = old["query_scores"][i].argmax(1), ridge[ci].argmax(1)
            checks.append(
                {
                    "trial_id": trial["id"],
                    "encoder": encoder,
                    "policy": policy,
                    "subject": p["subject"],
                    "k": p["k"],
                    "draw": p["draw"],
                    "permuted": p["permuted"],
                    "recipe_index": ri,
                    "primary_correct": int(np.sum(main == labels[q])),
                    "cpu_correct": int(np.sum(cpu_pred == labels[q])),
                    "dual_correct": int(np.sum(dual == labels[q])),
                    "cpu_prediction_changes": int(np.sum(cpu_pred != main)),
                    "dual_prediction_changes": int(np.sum(dual != main)),
                    "dual_score_max_error": dual_error,
                }
            )
            for j, clip in enumerate(q):
                row = [
                    trial["id"],
                    encoder,
                    policy,
                    p["subject"],
                    p["k"],
                    p["draw"],
                    p["permuted"],
                    records[clip]["id"],
                    KEYS[labels[clip]],
                    KEYS[reference[j]],
                    KEYS[prev[j]],
                    KEYS[main[j]],
                    KEYS[cpu_pred[j]],
                    KEYS[dual[j]],
                ]
                writer.writerow(row)
                if main[j] != cpu_pred[j] or main[j] != dual[j]:
                    changes.append(
                        {
                            **checks[-1],
                            "clip_id": records[clip]["id"],
                            "true_key": KEYS[labels[clip]],
                            "prediction": KEYS[main[j]],
                            "cpu_prediction": KEYS[cpu_pred[j]],
                            "dual_prediction": KEYS[dual[j]],
                            "primary_top1_margin": float(np.diff(np.sort(scores[j])[-2:])[0]),
                        }
                    )
    np.savez_compressed(output / f"cpu-scores-{encoder}.npz", query_scores=cpu_scores)
    return checks, changes


def run(output):
    result = read_json(RUN / "results.json")
    verify(RUN, result)
    if result["input_sha256"] != signature():
        raise ResearchError("Masukan run berubah.")
    selected = read_json(RUN / "selection.json")
    if file_hash(RUN / "selection.json") != result["selection_sha256"] or not (
        read_json(RUN / "validation.json")["created_at"]
        < selected["created_at"]
        < result["evaluation_started_at"]
    ):
        raise ResearchError("Pilihan tidak terkunci sebelum refit.")
    labels, plans, conditions = fixed_data()
    checked_costs = audit_budgets(plans)
    old_audit = read_json(OLD_AUDIT / "audit.json")
    old_direct = old_direct_counts(conditions)
    choices = selected["choices"]
    chosen = {choice_key(c): c for c in choices}
    trials = read_json(RUN / "trials.json")
    by_trial = {(t["encoder"], t["array_index"]): t for t in trials["trials"]}
    records = read_json(DATA / "records.json")["records"]
    primary = aggregate_tables(RUN, conditions)
    direct = {k: np.zeros_like(v) for k, v in primary.items()}
    if len(chosen) != 816 or len(by_trial) != 4896:
        raise ResearchError("Coverage pilihan/refit tidak lengkap.")
    paths = [
        Path(__file__),
        ROOT / "research/audit_skaid_svm.py",
        ROOT / "research/audit_skaid_selection.py",
        RUN / "results.json",
        OLD_AUDIT / "audit.json",
        CPU_CACHE,
    ]
    sig = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    fresh_directory(output)
    write_json(output / "lock.json", {"created_at": now(), "input_sha256": sig})
    stats = dict.fromkeys(
        (
            "max_mean_error",
            "max_scale_error",
            "max_median_error",
            "max_direct_score_error",
            "max_direct_binary_iterations",
            "new_candidate_heads_checked",
            "new_candidate_direct_prediction_changes",
            "new_candidate_heads_outside_score_tolerance",
            "new_candidate_heads_with_support_count_difference",
            "max_final_dual_score_error",
            "max_cpu_feature_score_error",
            "unchanged_recipe_refits_exact",
        ),
        0,
    )
    checks, changes = [], []
    with (
        np.load(CACHE / "features.npz", allow_pickle=False) as features,
        np.load(CPU_CACHE, allow_pickle=False) as cpu,
        threadpool_limits(limits=1),
        (output / "validation-checks.csv").open("w", newline="", encoding="utf-8") as vf,
        (output / "classifications.csv").open("w", newline="", encoding="utf-8") as cf,
    ):
        vw, cw = csv.writer(vf), csv.writer(cf)
        vw.writerow(
            [
                "encoder",
                "condition_index",
                "recipe_index",
                "primary_correct",
                "direct_correct",
                "prediction_changes",
                "max_score_error",
                "within_score_tolerance",
                "support_count_changed",
                "direct_max_binary_iterations",
            ]
        )
        cw.writerow(
            [
                "trial_id",
                "encoder",
                "policy",
                "subject",
                "k",
                "draw",
                "permuted",
                "clip_id",
                "true_key",
                "ridge_prediction",
                "old_svm_prediction",
                "svm_c_prediction",
                "cpu_prediction",
                "dual_prediction",
            ]
        )
        for encoder in ENCODERS:
            raw, cpu_raw = features[encoder].astype(float), cpu[encoder].astype(float)
            old = load_npz(OLD / "validation" / f"{encoder}.npz")
            extra = load_npz(RUN / "validation" / f"{encoder}.npz")
            new_direct = candidate_checks(raw, labels, conditions, encoder, extra, vw, stats)
            combined = np.concatenate((old_direct[encoder], new_direct), axis=1)
            for ci, p in enumerate(conditions):
                direct[encoder.split("_")[0], p["k"], p["permuted"]][p["subject"]] += combined[ci]
            finals, changed = final_checks(
                encoder,
                output,
                raw,
                cpu_raw,
                labels,
                records,
                conditions,
                by_trial,
                chosen,
                old,
                extra,
                cw,
                stats,
            )
            checks.extend(finals)
            changes.extend(changed)
    choice_checks, alternatives = audit_choices(choices, primary, direct, checked_costs)
    write_json(
        output / "choice-checks.json",
        {"checks": choice_checks, "direct_alternatives": alternatives},
    )
    write_json(output / "final-checks.json", {"checks": checks})
    write_json(output / "prediction-changes.json", {"changes": changes})
    if (
        stats["new_candidate_heads_checked"],
        old_audit["candidate_heads_checked"],
        len(checks),
    ) != (88128, 146880, 4896):
        raise ResearchError("Coverage audit tidak lengkap.")
    if (
        any(file_hash(ROOT / p) != digest for p, digest in sig.items())
        or signature() != result["input_sha256"]
    ):
        raise ResearchError("Masukan berubah selama audit C.")
    record = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        **stats,
        "old_candidate_audits_reused": old_audit["candidate_heads_checked"],
        "old_candidate_heads_outside_score_tolerance": old_audit[
            "candidate_heads_outside_score_tolerance"
        ],
        "old_candidate_direct_prediction_changes": old_audit[
            "candidate_direct_prediction_disagreements"
        ],
        "choices_checked": len(choice_checks),
        "source_correct_count_entries_differing_direct": sum(
            int(np.sum(primary[k] != direct[k])) for k in primary
        ),
        "diagnostic_direct_choices_changed": len(alternatives),
        "final_heads_replayed": len(checks),
        "normal_cpu_prediction_changes": sum(
            r["cpu_prediction_changes"] for r in checks if not r["permuted"]
        ),
        "control_cpu_prediction_changes": sum(
            r["cpu_prediction_changes"] for r in checks if r["permuted"]
        ),
        "final_dual_prediction_changes": sum(r["dual_prediction_changes"] for r in checks),
        "held_id_excluded_from_all_choices": True,
        "new_encoder_extractions": 0,
        "artifact_sha256": artifacts(output),
    }
    write_json(output / "audit.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        run(args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
