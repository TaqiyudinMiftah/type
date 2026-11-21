"""Audit all SVM candidates with direct kernels, locked choices and dual replay."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from audit_skaid_selection import independent_transform
from threadpoolctl import threadpool_limits

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.skaid_pilot_common import CACHE, DATA, ENCODERS, KEYS, ROOT, verify
from ketik_audio.skaid_pilot_probe import metrics, permuted_labels
from ketik_audio.skaid_selection_core import NORMALIZERS, source_budget
from ketik_audio.skaid_svm import OUTPUT as RUN
from ketik_audio.skaid_svm import RIDGE, aggregate_tables, load_data, signature
from ketik_audio.skaid_svm_core import (
    POLICIES,
    RECIPES,
    fit_selected,
    kernel_matrices,
    replay,
    select_from_table,
    svc,
)


def independent_median(x):
    values = np.concatenate([np.sum((x[i + 1 :] - x[i]) ** 2, axis=1) for i in range(len(x) - 1)])
    return float(np.percentile(values, 50))


def error(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def audit_choices(selection, tables, plans):
    checks = []
    for c in selection["choices"]:
        held, policy = c["held"], c["policy"]
        dev = [i for i in range(17) if i != held]
        table = tables[c["family"], c["k"], c["permuted"]]
        totals = table[dev].sum(0, dtype=np.int64)
        allowed = [i for i, r in enumerate(RECIPES) if r["kernel"] == policy]
        best = max(totals[allowed])
        tied = [i for i in allowed if totals[i] == best]
        winner = sorted(
            tied,
            key=lambda i: (
                RECIPES[i]["c"],
                RECIPES[i]["multiplier"] or 0,
                NORMALIZERS.index(RECIPES[i]["normalizer"]),
            ),
        )[0]
        changed = table.copy()
        changed[held] = np.arange(len(RECIPES)) * 10**8
        if (
            c["source_ids"] != dev
            or c["source_correct_by_recipe"] != totals.tolist()
            or c["recipe_index"] != winner
            or c["recipe"] != RECIPES[winner]
            or select_from_table(changed, held, policy) != select_from_table(table, held, policy)
            or c["source_labels_unique_total"]
            != source_budget(plans, held, c["k"])["source_labels_unique_total"]
        ):
            raise ResearchError("Audit seleksi, held ID, tie atau budget gagal.")
        checks.append(
            {"choice_id": c["id"], "held_excluded": True, "counterfactual_invariant": True}
        )
    expected = [source_budget(plans, h, k) for h in range(17) for k in (1, 2, 3, 5, 10)]
    if read_json(RUN / "budgets.json")["budgets"] != expected:
        raise ResearchError("Budget tidak cocok dengan union indeks.")
    return checks


def run(output):
    result = read_json(RUN / "results.json")
    verify(RUN, result)
    if result["input_sha256"] != signature():
        raise ResearchError("Masukan run berubah.")
    selection = read_json(RUN / "selection.json")
    if (
        result["selection_sha256"] != file_hash(RUN / "selection.json")
        or selection["created_at"] >= result["evaluation_started_at"]
    ):
        raise ResearchError("Pilihan tidak terkunci sebelum refit.")
    fresh_directory(output)
    paths = [
        Path(__file__),
        ROOT / "research/audit_skaid_selection.py",
        RUN / "results.json",
        RUN / "selection.json",
    ]
    sig = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    write_json(output / "lock.json", {"created_at": now(), "input_sha256": sig})
    labels, plans, conditions = load_data()
    records = read_json(DATA / "records.json")["records"]
    choice_checks = audit_choices(selection, aggregate_tables(RUN, conditions), plans)
    write_json(output / "choice-checks.json", {"choices": choice_checks})
    choices = {
        (c["policy"], c["family"], c["held"], c["k"], c["permuted"]): c
        for c in selection["choices"]
    }
    trials = read_json(RUN / "trials.json")
    by_trial = {(t["encoder"], t["array_index"]): t for t in trials["trials"]}
    if trials["zero_shot"] != read_json(RIDGE / "trials.json")["zero_shot"]:
        raise ResearchError("K0 berubah.")
    maxima = {
        k: 0.0
        for k in (
            "candidate_direct",
            "scaler",
            "bandwidth",
            "final_dual",
            "selected_vs_candidate",
            "cpu_features",
        )
    }
    candidate_count = direct_flips = outside_tolerance = support_count_differences = 0
    cpu_flips = dual_flips = 0
    final_checks, changes = [], []
    with (
        np.load(CACHE / "features.npz", allow_pickle=False) as features,
        np.load(ROOT / "reports/skaid-pilot-audit/cpu-features.npz", allow_pickle=False) as cpu,
        threadpool_limits(limits=1),
        (output / "validation-checks.csv").open("w", newline="") as vf,
        (output / "classifications.csv").open("w", newline="") as cf,
    ):
        vw = csv.writer(vf)
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
            ]
        )
        cw = csv.writer(cf)
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
                "svm_prediction",
                "cpu_prediction",
                "dual_prediction",
            ]
        )
        for encoder in ENCODERS:
            raw, cpu_raw = features[encoder].astype(float), cpu[encoder].astype(float)
            with np.load(RUN / "validation" / f"{encoder}.npz", allow_pickle=False) as z:
                validation = {k: z[k] for k in z.files}
            with np.load(RUN / "final" / f"{encoder}.npz", allow_pickle=False) as z:
                final = {k: z[k] for k in z.files}
            with np.load(RIDGE / "final" / f"{encoder}.npz", allow_pickle=False) as z:
                ridge_scores = z["query_scores"]
            cpu_scores = np.empty_like(final["query_scores"])
            for ci, p in enumerate(conditions):
                s, q = p["support"], p["query"]
                y = permuted_labels(labels, p) if p["permuted"] else labels[s]
                for ni, mode in enumerate(NORMALIZERS):
                    x, query, mu, scale = independent_transform(raw[s], raw[q], mode)
                    median = independent_median(x)
                    maxima["scaler"] = max(
                        maxima["scaler"],
                        error(mu, validation["mean"][ci, ni]),
                        error(scale, validation["scale"][ci, ni]),
                    )
                    maxima["bandwidth"] = max(
                        maxima["bandwidth"], error(median, validation["median"][ci, ni])
                    )
                    if not np.allclose(
                        mu, validation["mean"][ci, ni], atol=1e-12, rtol=1e-10
                    ) or not np.allclose(
                        scale, validation["scale"][ci, ni], atol=1e-12, rtol=1e-10
                    ):
                        raise ResearchError("Statistik transformasi berbeda.")
                    if not np.isclose(median, validation["median"][ci, ni], atol=1e-12, rtol=1e-10):
                        raise ResearchError("Median bandwidth berbeda.")
                    for ri, recipe in enumerate(RECIPES):
                        if recipe["normalizer"] != mode:
                            continue
                        gamma = (
                            recipe["multiplier"] / median
                            if recipe["kernel"] == "rbf" and median > 1e-12
                            else (recipe["multiplier"] or 0)
                        )
                        if not np.isclose(
                            gamma, validation["gamma"][ci, ri], atol=1e-12, rtol=1e-10
                        ):
                            raise ResearchError("Gamma berbeda dari dukungan saja.")
                        head = svc(recipe["c"], recipe["kernel"], gamma if gamma else "scale").fit(
                            x, y
                        )
                        if head.fit_status_ != 0 or validation["fit_status"][ci, ri] != 0:
                            raise ResearchError("Ada fit yang tidak konvergen.")
                        primary = validation["query_scores"][ci, ri]
                        scores, support = head.decision_function(query), head.decision_function(x)
                        expected_count = int(np.sum(primary.argmax(1) == labels[q]))
                        if expected_count != int(validation["query_correct"][ci, ri]):
                            raise ResearchError("Count sumber berbeda dari skor tersimpan.")
                        score_error = error(scores, primary)
                        within = bool(np.allclose(scores, primary, atol=1e-5, rtol=1e-5))
                        flips = int(np.sum(scores.argmax(1) != primary.argmax(1)))
                        support_changed = bool(
                            np.sum(support.argmax(1) == y)
                            != validation["support_fit_correct"][ci, ri]
                            or np.sum(support.argmax(1) == labels[s])
                            != validation["support_true_correct"][ci, ri]
                        )
                        maxima["candidate_direct"] = max(maxima["candidate_direct"], score_error)
                        candidate_count += 1
                        direct_flips += flips
                        outside_tolerance += int(not within)
                        support_count_differences += int(support_changed)
                        vw.writerow(
                            [
                                encoder,
                                ci,
                                ri,
                                expected_count,
                                int(np.sum(scores.argmax(1) == labels[q])),
                                flips,
                                score_error,
                                within,
                                support_changed,
                            ]
                        )
                for pi, policy in enumerate(POLICIES):
                    i = pi * len(conditions) + ci
                    c = choices[policy, encoder.split("_")[0], p["subject"], p["k"], p["permuted"]]
                    trial, ri = by_trial[encoder, i], c["recipe_index"]
                    if (
                        final["recipe_index"][i] != ri
                        or trial["choice_id"] != c["id"]
                        or not np.array_equal(final["support"][i, : len(s)], s)
                        or not np.array_equal(final["query"][i], q)
                        or not np.array_equal(final["fit_labels"][i, : len(s)], y)
                    ):
                        raise ResearchError("Final tidak sesuai pilihan/split/label.")
                    scores = final["query_scores"][i]
                    maxima["selected_vs_candidate"] = max(
                        maxima["selected_vs_candidate"],
                        error(scores, validation["query_scores"][ci, ri]),
                    )
                    if not np.array_equal(scores, validation["query_scores"][ci, ri]):
                        raise ResearchError("Refit berbeda dari kandidat dengan resep sama.")
                    x, query, mu, scale = independent_transform(
                        raw[s], raw[q], c["recipe"]["normalizer"]
                    )
                    gram, cross, gamma = kernel_matrices(x, query, c["recipe"])
                    if (
                        not np.allclose(mu, final["mean"][i], atol=1e-12, rtol=1e-10)
                        or not np.allclose(scale, final["scale"][i], atol=1e-12, rtol=1e-10)
                        or not np.isclose(gamma, final["gamma"][i], atol=1e-12, rtol=1e-10)
                        or not np.isclose(
                            independent_median(x), final["median"][i], atol=1e-12, rtol=1e-10
                        )
                    ):
                        raise ResearchError("Statistik/bandwidth final tidak support-only.")
                    ns = int(final["n_support"][i].sum())
                    positions = final["support_positions"][i, :ns]
                    if (
                        len(set(positions.tolist())) != ns
                        or np.any(positions < 0)
                        or np.any(positions >= len(s))
                        or not np.array_equal(
                            np.bincount(y[positions], minlength=12), final["n_support"][i]
                        )
                    ):
                        raise ResearchError("Urutan/jumlah support vector tidak valid.")
                    args = [
                        final["support_positions"][i, :ns],
                        final["dual_coef"][i, :, :ns],
                        final["intercept"][i],
                        final["n_support"][i],
                    ]
                    reconstructed = replay(cross, *args)
                    support_replay = replay(gram, *args)
                    maxima["final_dual"] = max(
                        maxima["final_dual"],
                        error(reconstructed, scores),
                        error(support_replay, final["support_scores"][i, : len(s)]),
                    )
                    current_dual_flips = int(np.sum(reconstructed.argmax(1) != scores.argmax(1)))
                    dual_flips += current_dual_flips
                    _, cscores, _, _, _, _, _ = fit_selected(cpu_raw[s], y, cpu_raw[q], c["recipe"])
                    cpu_scores[i] = cscores
                    maxima["cpu_features"] = max(maxima["cpu_features"], error(cscores, scores))
                    current_cpu_flips = int(np.sum(cscores.argmax(1) != scores.argmax(1)))
                    cpu_flips += current_cpu_flips
                    if metrics(labels[q], scores) != trial["query"]:
                        raise ResearchError("Metrik final berbeda dari prediksi.")
                    if (
                        float(np.mean(final["support_scores"][i, : len(s)].argmax(1) == y))
                        != trial["support_fit_label_accuracy"]
                        or float(
                            np.mean(final["support_scores"][i, : len(s)].argmax(1) == labels[s])
                        )
                        != trial["support_true_label_accuracy"]
                        or trial["source_labels_unique_total"] != c["source_labels_unique_total"]
                    ):
                        raise ResearchError("Metrik dukungan/budget final berbeda.")
                    check = {
                        "trial_id": trial["id"],
                        "policy": policy,
                        "encoder": encoder,
                        "permuted": p["permuted"],
                        "subject": p["subject"],
                        "k": p["k"],
                        "draw": p["draw"],
                        "cpu_prediction_changes": current_cpu_flips,
                        "dual_prediction_changes": current_dual_flips,
                        "primary_correct": int(trial["query"]["correct"]),
                        "cpu_correct": int(np.sum(cscores.argmax(1) == labels[q])),
                    }
                    final_checks.append(check)
                    for j, clip in enumerate(q):
                        actual, prediction, cpu_prediction, dual_prediction = (
                            labels[clip],
                            scores[j].argmax(),
                            cscores[j].argmax(),
                            reconstructed[j].argmax(),
                        )
                        cw.writerow(
                            [
                                trial["id"],
                                encoder,
                                policy,
                                p["subject"],
                                p["k"],
                                p["draw"],
                                p["permuted"],
                                records[clip]["id"],
                                KEYS[actual],
                                KEYS[ridge_scores[ci, j].argmax()],
                                KEYS[prediction],
                                KEYS[cpu_prediction],
                                KEYS[dual_prediction],
                            ]
                        )
                        if prediction != cpu_prediction or prediction != dual_prediction:
                            changes.append(
                                {
                                    **check,
                                    "clip_id": records[clip]["id"],
                                    "true_key": KEYS[actual],
                                    "prediction": KEYS[prediction],
                                    "cpu_prediction": KEYS[cpu_prediction],
                                    "dual_prediction": KEYS[dual_prediction],
                                    "recipe_index": ri,
                                    "recipe": c["recipe"],
                                    "top1_minus_top2": float(np.diff(np.sort(scores[j])[-2:])[0]),
                                }
                            )
                if (ci + 1) % 51 == 0:
                    print(f"Audit {encoder}: {ci + 1}/{len(conditions)}", flush=True)
            np.savez_compressed(output / f"cpu-scores-{encoder}.npz", query_scores=cpu_scores)
    write_json(output / "final-checks.json", {"checks": final_checks})
    write_json(output / "prediction-changes.json", {"changes": changes})
    if signature() != result["input_sha256"] or any(
        file_hash(ROOT / p) != digest for p, digest in sig.items()
    ):
        raise ResearchError("Masukan audit berubah.")
    record = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "candidate_heads_checked": candidate_count,
        "choices_checked": len(choice_checks),
        "final_heads_replayed": len(final_checks),
        "maximum_errors": maxima,
        "candidate_direct_prediction_disagreements": direct_flips,
        "candidate_heads_outside_score_tolerance": outside_tolerance,
        "candidate_heads_with_support_count_difference": support_count_differences,
        "dual_prediction_disagreements": dual_flips,
        "cpu_feature_prediction_disagreements": cpu_flips,
        "held_id_excluded_from_all_choices": True,
        "new_encoder_extractions": 0,
    }
    record["artifact_sha256"] = {
        str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    write_json(output / "audit.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/skaid-svm-audit")
    args = parser.parse_args()
    try:
        run(args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
