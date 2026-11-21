"""Audit all source recipe candidates, held-ID exclusions, budgets and selected refits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.skaid_pilot_common import CACHE, DATA, ENCODERS, KEYS, ROOT, verify
from ketik_audio.skaid_pilot_common import RUN as PILOT
from ketik_audio.skaid_pilot_probe import metrics, permuted_labels
from ketik_audio.skaid_selection import OUTPUT as RUN
from ketik_audio.skaid_selection import aggregate_source_tables, load_data
from ketik_audio.skaid_selection_core import (
    ALPHAS,
    BASELINE_INDEX,
    NORMALIZERS,
    RECIPES,
    fit_selected,
    ridge_path,
    select_from_table,
    source_budget,
    transform,
)


def close(actual, expected, label, atol=1e-6, rtol=1e-5):
    error = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ResearchError(f"Perbedaan numerik melewati toleransi: {label}, max={error}")
    return error


def independent_transform(x, q, mode):
    if mode == "row_l2":
        mean, scale = np.zeros(x.shape[1]), np.ones(x.shape[1])
    else:
        mean = np.sum(x, axis=0) / len(x)
        variance = np.sum((x - mean) ** 2, axis=0) / len(x)
        sd = np.sqrt(variance)
        scale = np.where(sd <= 1e-12, 1.0, sd)
    return normalize((x - mean) / scale), normalize((q - mean) / scale), mean, scale


def dual(x, y, alpha):
    target = 2 * np.eye(12)[y] - 1
    xc, yc = x - x.mean(0), target - target.mean(0)
    coefficient = np.linalg.solve(xc @ xc.T + alpha * np.eye(len(x)), yc).T @ xc
    intercept = target.mean(0) - coefficient @ x.mean(0)
    return coefficient, intercept


def audit_choices(selection, tables, plans):
    checks = []
    for choice in selection["choices"]:
        held, family, k, permuted = (choice[name] for name in ("held", "family", "k", "permuted"))
        dev = [i for i in range(17) if i != held]
        if choice["source_ids"] != dev:
            raise ResearchError("Held ID masuk sumber seleksi atau ID sumber hilang.")
        table = tables[family, k, permuted]
        totals = np.sum(table[dev], axis=0, dtype=np.int64)
        tied = np.flatnonzero(totals == totals.max()).tolist()
        winner = sorted(
            tied, key=lambda i: (-RECIPES[i]["alpha"], NORMALIZERS.index(RECIPES[i]["normalizer"]))
        )[0]
        if (
            choice["recipe_index"] != winner
            or choice["recipe"] != RECIPES[winner]
            or choice["source_correct_by_recipe"] != totals.tolist()
        ):
            raise ResearchError("Pilihan/tie rule berbeda pada audit independen.")
        changed = table.copy()
        changed[held] = np.arange(len(RECIPES)) * 10**8
        if select_from_table(changed, held) != select_from_table(table, held):
            raise ResearchError("Pilihan berubah setelah hanya baris held ID diubah.")
        budget = source_budget(plans, held, k)
        for key in (
            "source_support_labels_unique",
            "source_query_labels_unique",
            "source_labels_unique_total",
        ):
            if choice[key] != budget[key]:
                raise ResearchError("Anggaran label sumber tidak cocok dengan union indeks.")
        checks.append(
            {
                "choice_id": choice["id"],
                "held_id_excluded": True,
                "held_row_perturbation_invariant": True,
                "independent_tie_rule_match": True,
            }
        )
    expected_budgets = [source_budget(plans, h, k) for h in range(17) for k in (1, 2, 3, 5, 10)]
    if read_json(RUN / "budgets.json")["budgets"] != expected_budgets:
        raise ResearchError("Tabel label cost berubah atau salah.")
    return checks


def run(output):
    result = read_json(RUN / "results.json")
    verify(RUN, result)
    for base, name in (
        (DATA, "manifest.json"),
        (CACHE, "manifest.json"),
        (PILOT, "results.json"),
        (ROOT / "reports/skaid-pilot-audit", "audit.json"),
    ):
        verify(base, read_json(base / name))
    selection = read_json(RUN / "selection.json")
    if (
        file_hash(RUN / "selection.json") != result["selection_sha256"]
        or selection["created_at"] >= result["evaluation_started_at"]
    ):
        raise ResearchError("Pilihan tidak terkunci sebelum refit luar.")
    fresh_directory(output)
    paths = [
        Path(__file__),
        RUN / "results.json",
        RUN / "selection.json",
        ROOT / "reports/skaid-pilot-audit/audit.json",
        ROOT / "reports/skaid-pilot-audit/cpu-features.npz",
    ]
    sig = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    write_json(output / "lock.json", {"created_at": now(), "input_sha256": sig})
    labels, plans, conditions = load_data()
    records = read_json(DATA / "records.json")["records"]
    tables, _ = aggregate_source_tables(RUN, conditions)
    choice_checks = audit_choices(selection, tables, plans)
    write_json(output / "choice-checks.json", {"choices": choice_checks})
    chosen = {(r["family"], r["held"], r["k"], r["permuted"]): r for r in selection["choices"]}
    trials = read_json(RUN / "trials.json")
    if trials["zero_shot"] != read_json(PILOT / "trials.json")["zero_shot"]:
        raise ResearchError("K0 yang dipakai ulang berbeda dari pilot.")
    by_trial = {(r["encoder"], r["array_index"]): r for r in trials["trials"]}
    maxima = {
        "candidate_sklearn": 0.0,
        "candidate_path_replay": 0.0,
        "scaler": 0.0,
        "final_replay": 0.0,
        "final_dual": 0.0,
        "cpu_feature_scores": 0.0,
    }
    candidate_flips, cpu_flips, dual_flips, path_final_flips, candidate_count = 0, 0, 0, 0, 0
    final_checks = []
    with (
        np.load(CACHE / "features.npz", allow_pickle=False) as features,
        np.load(ROOT / "reports/skaid-pilot-audit/cpu-features.npz", allow_pickle=False) as cpu,
        threadpool_limits(limits=1),
        (output / "validation-checks.csv").open("w", newline="", encoding="utf-8") as vf,
        (output / "classifications.csv").open("w", newline="", encoding="utf-8") as cf,
    ):
        validation_writer = csv.DictWriter(
            vf,
            fieldnames=[
                "encoder",
                "condition_index",
                "subject",
                "k",
                "draw",
                "permuted",
                "recipe_index",
                "normalizer",
                "alpha",
                "primary_correct",
                "sklearn_correct",
                "prediction_disagreements",
                "max_score_error",
            ],
        )
        validation_writer.writeheader()
        classification_writer = csv.DictWriter(
            cf,
            fieldnames=[
                "trial_id",
                "encoder",
                "subject",
                "k",
                "draw",
                "permuted",
                "recipe_index",
                "clip_id",
                "true_key",
                "baseline_prediction",
                "selected_prediction",
                "cpu_prediction",
            ],
        )
        classification_writer.writeheader()
        for encoder in ENCODERS:
            raw, cpu_raw = features[encoder].astype(float), cpu[encoder].astype(float)
            with np.load(RUN / "validation" / f"{encoder}.npz", allow_pickle=False) as z:
                validation = {k: z[k] for k in z.files}
            with np.load(RUN / "final" / f"{encoder}.npz", allow_pickle=False) as z:
                final = {k: z[k] for k in z.files}
            with np.load(PILOT / f"{encoder}.npz", allow_pickle=False) as z:
                baseline = {k: z[k] for k in ("query_scores", "support_scores")}
            final_cpu_scores = []
            for i, p in enumerate(conditions):
                s, q = p["support"], p["query"]
                y = permuted_labels(labels, p) if p["permuted"] else labels[s]
                for j, mode in enumerate(NORMALIZERS):
                    nx, nq, mu, scale = independent_transform(raw[s], raw[q], mode)
                    maxima["scaler"] = max(
                        maxima["scaler"],
                        close(mu, validation["mean"][i, j], "source mean", 1e-12, 1e-10),
                        close(scale, validation["scale"][i, j], "source scale", 1e-12, 1e-10),
                    )
                    px, pq, _, _ = transform(raw[s], raw[q], mode)
                    path_query, path_support = ridge_path(px, y, pq)
                    for ai, alpha in enumerate(ALPHAS):
                        ri = j * len(ALPHAS) + ai
                        primary = validation["query_scores"][i, ri]
                        maxima["candidate_path_replay"] = max(
                            maxima["candidate_path_replay"],
                            close(path_query[ai], primary, "path replay", 1e-8, 1e-8),
                        )
                        if mode == "row_l2" and ai == BASELINE_INDEX:
                            if not np.array_equal(primary, baseline["query_scores"][i]):
                                raise ResearchError(
                                    "Baseline source candidate tidak dipakai ulang persis."
                                )
                            support_prediction = baseline["support_scores"][i, : len(s)].argmax(1)
                        else:
                            support_prediction = path_support[ai].argmax(1)
                        if (
                            int(np.sum(primary.argmax(1) == labels[q]))
                            != validation["query_correct"][i, ri]
                            or int(np.sum(support_prediction == y))
                            != validation["support_fit_correct"][i, ri]
                            or int(np.sum(support_prediction == labels[s]))
                            != validation["support_true_correct"][i, ri]
                        ):
                            raise ResearchError("Hitungan benar source candidate salah.")
                        head = RidgeClassifier(
                            alpha=alpha, solver="cholesky", fit_intercept=True, class_weight=None
                        ).fit(nx, y)
                        scores = head.decision_function(nq)
                        error = close(scores, primary, "source candidate vs sklearn")
                        maxima["candidate_sklearn"] = max(maxima["candidate_sklearn"], error)
                        flips = int(np.sum(scores.argmax(1) != primary.argmax(1)))
                        candidate_flips += flips
                        candidate_count += 1
                        validation_writer.writerow(
                            {
                                "encoder": encoder,
                                "condition_index": i,
                                "subject": p["subject"],
                                "k": p["k"],
                                "draw": p["draw"],
                                "permuted": p["permuted"],
                                "recipe_index": ri,
                                "normalizer": mode,
                                "alpha": alpha,
                                "primary_correct": int(validation["query_correct"][i, ri]),
                                "sklearn_correct": int(np.sum(scores.argmax(1) == labels[q])),
                                "prediction_disagreements": flips,
                                "max_score_error": error,
                            }
                        )
                choice = chosen[encoder.split("_")[0], p["subject"], p["k"], p["permuted"]]
                recipe = choice["recipe"]
                trial = by_trial[encoder, i]
                if (
                    final["recipe_index"][i] != choice["recipe_index"]
                    or trial["choice_id"] != choice["id"]
                    or not np.array_equal(final["support"][i, : len(s)], s)
                    or not np.array_equal(final["query"][i], q)
                    or not np.array_equal(final["fit_labels"][i, : len(s)], y)
                ):
                    raise ResearchError("Final model tidak memakai pilihan/split/label terkunci.")
                head, scores, supports, mu, scale = fit_selected(raw[s], y, raw[q], recipe)
                for actual, expected in (
                    (head.coef_, final["coef"][i]),
                    (head.intercept_, final["intercept"][i]),
                    (scores, final["query_scores"][i]),
                    (supports, final["support_scores"][i, : len(s)]),
                    (mu, final["mean"][i]),
                    (scale, final["scale"][i]),
                ):
                    maxima["final_replay"] = max(
                        maxima["final_replay"],
                        close(actual, expected, "final replay", 1e-10, 1e-10),
                    )
                x, query, mean, standard = independent_transform(
                    raw[s], raw[q], recipe["normalizer"]
                )
                maxima["scaler"] = max(
                    maxima["scaler"],
                    close(mean, final["mean"][i], "final mean", 1e-12, 1e-10),
                    close(standard, final["scale"][i], "final scale", 1e-12, 1e-10),
                )
                coefficient, intercept = dual(x, y, recipe["alpha"])
                dual_scores = query @ coefficient.T + intercept
                for actual, expected in (
                    (coefficient, final["coef"][i]),
                    (intercept, final["intercept"][i]),
                    (dual_scores, scores),
                ):
                    maxima["final_dual"] = max(
                        maxima["final_dual"],
                        close(actual, expected, "selected vs independent dual"),
                    )
                dual_flips += int(np.sum(dual_scores.argmax(1) != scores.argmax(1)))
                path_final_flips += int(
                    np.sum(
                        validation["query_scores"][i, choice["recipe_index"]].argmax(1)
                        != scores.argmax(1)
                    )
                )
                if (
                    metrics(labels[q], scores) != trial["query"]
                    or float(np.mean(supports.argmax(1) == y))
                    != trial["support_fit_label_accuracy"]
                    or float(np.mean(supports.argmax(1) == labels[s]))
                    != trial["support_true_label_accuracy"]
                ):
                    raise ResearchError("Metrik final tidak sesuai replay.")
                _, cpu_scores, _, _, _ = fit_selected(cpu_raw[s], y, cpu_raw[q], recipe)
                flips = int(np.sum(cpu_scores.argmax(1) != scores.argmax(1)))
                cpu_flips += flips
                maxima["cpu_feature_scores"] = max(
                    maxima["cpu_feature_scores"], float(np.abs(cpu_scores - scores).max())
                )
                final_cpu_scores.append(cpu_scores)
                final_checks.append(
                    {
                        "trial_id": trial["id"],
                        "primary": trial["query"],
                        "cpu_features": metrics(labels[q], cpu_scores),
                        "prediction_disagreements": flips,
                    }
                )
                for row_index, clip in enumerate(q):
                    classification_writer.writerow(
                        {
                            "trial_id": trial["id"],
                            "encoder": encoder,
                            "subject": p["subject"],
                            "k": p["k"],
                            "draw": p["draw"],
                            "permuted": p["permuted"],
                            "recipe_index": choice["recipe_index"],
                            "clip_id": records[clip]["id"],
                            "true_key": KEYS[int(labels[clip])],
                            "baseline_prediction": KEYS[
                                int(baseline["query_scores"][i, row_index].argmax())
                            ],
                            "selected_prediction": KEYS[int(scores[row_index].argmax())],
                            "cpu_prediction": KEYS[int(cpu_scores[row_index].argmax())],
                        }
                    )
            np.savez_compressed(
                output / f"cpu-scores-{encoder}.npz", query_scores=np.stack(final_cpu_scores)
            )
            print(f"Audited {encoder}: 5508 candidates and306 selected heads", flush=True)
    write_json(output / "final-checks.json", {"trials": final_checks})
    verify(RUN, result)
    for name, digest in sig.items():
        if file_hash(ROOT / name) != digest:
            raise ResearchError("Masukan/kode audit berubah.")
    summary = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "candidate_heads_checked": candidate_count,
        "choices_checked": len(choice_checks),
        "final_heads_replayed": len(final_checks),
        "maximum_errors": maxima,
        "source_candidate_prediction_disagreements_sklearn": candidate_flips,
        "selected_path_vs_refit_prediction_disagreements": path_final_flips,
        "dual_prediction_disagreements": dual_flips,
        "cpu_feature_prediction_disagreements_repeated_trials": cpu_flips,
        "held_id_excluded_from_all_choices": True,
        "scalers_fit_on_support_only": True,
        "cpu_feature_cache_reused_from_pilot_audit": True,
        "new_encoder_extractions": 0,
    }
    summary["artifact_sha256"] = {
        p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()
    }
    write_json(output / "audit.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/skaid-selection-audit")
    args = parser.parse_args()
    try:
        run(args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
