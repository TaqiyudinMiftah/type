"""Validate recipes by source ID, lock cross-ID choices, then refit on held-ID supports."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .common import (
    ResearchError,
    environment,
    file_hash,
    fresh_directory,
    now,
    read_json,
    write_json,
)
from .skaid_pilot_common import CACHE, DATA, ENCODERS, KS, ROOT, verify
from .skaid_pilot_common import RUN as PILOT
from .skaid_pilot_probe import metrics, permuted_labels, validate_plan
from .skaid_selection_core import (
    ALPHAS,
    BASELINE_INDEX,
    FAMILIES,
    NORMALIZERS,
    RECIPES,
    fit_selected,
    ridge_path,
    select_from_table,
    source_budget,
    transform,
)

PROTOCOL = ROOT / "research/SKAID_SELECTION_PROTOCOL.md"
OUTPUT = ROOT / "runs/skaid-selection"
PILOT_AUDIT = ROOT / "reports/skaid-pilot-audit"


def signature():
    for base, name in (
        (DATA, "manifest.json"),
        (CACHE, "manifest.json"),
        (PILOT, "results.json"),
        (PILOT_AUDIT, "audit.json"),
    ):
        verify(base, read_json(base / name))
    paths = [
        Path(__file__),
        ROOT / "src/ketik_audio/skaid_selection_core.py",
        PROTOCOL,
        ROOT / "src/ketik_audio/skaid_pilot_probe.py",
        DATA / "manifest.json",
        DATA / "plan.json",
        CACHE / "manifest.json",
        CACHE / "features.npz",
        PILOT / "results.json",
        PILOT / "trials.json",
        PILOT_AUDIT / "audit.json",
        PILOT_AUDIT / "cpu-features.npz",
    ]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def load_data():
    with np.load(DATA / "segments.npz", allow_pickle=False) as z:
        labels = z["y"]
    plans = read_json(DATA / "plan.json")["plans"]
    validate_plan(read_json(DATA / "records.json")["records"], plans, labels)
    conditions = [{**p, "permuted": False} for p in plans] + [
        {**p, "permuted": True} for p in plans if p["k"] == 10
    ]
    return labels, plans, conditions


def validate(output):
    sig = signature()
    fresh_directory(output)
    (output / "validation").mkdir()
    labels, _, conditions = load_data()
    write_json(
        output / "validation-lock.json",
        {
            "created_at": now(),
            "input_sha256": sig,
            "environment": environment(),
            "recipes": RECIPES,
        },
    )
    write_json(output / "conditions.json", {"conditions": conditions})
    (output / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    max_error, flips, fits = 0.0, 0, 0
    with (
        np.load(CACHE / "features.npz", allow_pickle=False) as features,
        threadpool_limits(limits=1),
    ):
        for encoder in ENCODERS:
            raw = features[encoder].astype(np.float64)
            with np.load(PILOT / f"{encoder}.npz", allow_pickle=False) as old:
                old_values = {
                    k: old[k]
                    for k in ("query_scores", "support_scores", "support", "query", "fit_labels")
                }
            n, d = len(conditions), raw.shape[1]
            arrays = {
                "query_scores": np.empty((n, len(RECIPES), 60, 12)),
                "query_correct": np.empty((n, len(RECIPES)), dtype=np.int64),
                "support_fit_correct": np.empty((n, len(RECIPES)), dtype=np.int64),
                "support_true_correct": np.empty((n, len(RECIPES)), dtype=np.int64),
                "mean": np.empty((n, 2, d)),
                "scale": np.empty((n, 2, d)),
            }
            for i, p in enumerate(conditions):
                s, q = p["support"], p["query"]
                y = permuted_labels(labels, p) if p["permuted"] else labels[s]
                if (
                    not np.array_equal(old_values["support"][i, : len(s)], s)
                    or not np.array_equal(old_values["query"][i], q)
                    or not np.array_equal(old_values["fit_labels"][i, : len(s)], y)
                ):
                    raise ResearchError("Baseline/split tidak sama dengan pilot.")
                for j, mode in enumerate(NORMALIZERS):
                    x, query, mean, scale = transform(raw[s], raw[q], mode)
                    queries, supports = ridge_path(x, y, query)
                    arrays["mean"][i, j], arrays["scale"][i, j] = mean, scale
                    if mode == "row_l2":
                        for actual, expected in (
                            (queries[BASELINE_INDEX], old_values["query_scores"][i]),
                            (supports[BASELINE_INDEX], old_values["support_scores"][i, : len(s)]),
                        ):
                            max_error = max(max_error, float(np.abs(actual - expected).max()))
                            if not np.allclose(actual, expected, atol=1e-8, rtol=1e-8):
                                raise ResearchError("Path tidak mereproduksi baseline alpha1.")
                        flips += int(
                            np.sum(
                                queries[BASELINE_INDEX].argmax(1)
                                != old_values["query_scores"][i].argmax(1)
                            )
                        )
                        queries[BASELINE_INDEX] = old_values["query_scores"][i]
                        supports[BASELINE_INDEX] = old_values["support_scores"][i, : len(s)]
                    a, b = j * len(ALPHAS), (j + 1) * len(ALPHAS)
                    arrays["query_scores"][i, a:b] = queries
                    arrays["query_correct"][i, a:b] = np.sum(
                        queries.argmax(2) == labels[q][None], axis=1
                    )
                    arrays["support_fit_correct"][i, a:b] = np.sum(
                        supports.argmax(2) == y[None], axis=1
                    )
                    arrays["support_true_correct"][i, a:b] = np.sum(
                        supports.argmax(2) == labels[s][None], axis=1
                    )
                    fits += len(ALPHAS)
            np.savez_compressed(output / "validation" / f"{encoder}.npz", **arrays)
            print(f"Source validation {encoder}: {n * len(RECIPES)} recipe tasks", flush=True)
    if signature() != sig:
        raise ResearchError("Masukan/kode berubah selama validasi sumber.")
    result = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "candidate_heads": fits,
        "baseline_heads_reused": 2448,
        "new_candidate_heads": fits - 2448,
        "gram_factorizations": 4896,
        "baseline_path_max_abs_error": max_error,
        "baseline_argmax_disagreements_before_canonical_reuse": flips,
        "role": "each ID validates recipes for other held-ID folds; not an unseen test",
    }
    result["artifact_sha256"] = {
        str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    write_json(output / "validation.json", result)
    return result


def checked_validation(output):
    result = read_json(output / "validation.json")
    verify(output, result)
    if result["input_sha256"] != signature():
        raise ResearchError("Validasi berasal dari sumber/kode berbeda.")
    return result


def aggregate_source_tables(output, conditions):
    tables, counts = {}, {}
    for encoder in ENCODERS:
        family = encoder.split("_")[0]
        with np.load(output / "validation" / f"{encoder}.npz", allow_pickle=False) as z:
            correct = z["query_correct"]
        for i, p in enumerate(conditions):
            key = (family, p["k"], p["permuted"])
            if key not in tables:
                tables[key] = np.zeros((17, len(RECIPES)), dtype=np.int64)
                counts[key] = np.zeros(17, dtype=np.int64)
            tables[key][p["subject"]] += correct[i]
            counts[key][p["subject"]] += 1
    for key in tables:
        expected = 3 if key[0] in ("mfcc", "beats") else 9
        if not np.array_equal(counts[key], np.full(17, expected)):
            raise ResearchError("Seed/draw/ID sumber tidak lengkap atau tidak seimbang.")
    return tables, counts


def choices_from_tables(tables, plans):
    budgets = [source_budget(plans, held, k) for held in range(17) for k in KS]
    choices = []
    for held in range(17):
        for family in FAMILIES:
            for k, permuted in [*((k, False) for k in KS), (10, True)]:
                chosen = select_from_table(tables[family, k, permuted], held)
                budget = next(b for b in budgets if b["held"] == held and b["k"] == k)
                choices.append(
                    {
                        "id": f"{family}-h{held:02d}-k{k}-p{int(permuted)}",
                        "family": family,
                        "k": k,
                        "permuted": permuted,
                        **chosen,
                        "source_support_labels_unique": budget["source_support_labels_unique"],
                        "source_query_labels_unique": budget["source_query_labels_unique"],
                        "source_labels_unique_total": budget["source_labels_unique_total"],
                        "harrison_pretraining_labels_reused": 720 if family == "pretrained" else 0,
                    }
                )
    return choices, budgets


def select(output):
    validated = checked_validation(output)
    if (output / "selection.json").exists():
        raise ResearchError("Seleksi sudah ada; jangan menimpa pilihan terkunci.")
    _, plans, conditions = load_data()
    tables, counts = aggregate_source_tables(output, conditions)
    choices, budgets = choices_from_tables(tables, plans)
    write_json(output / "budgets.json", {"budgets": budgets})
    result = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": validated["input_sha256"],
        "validation_sha256": file_hash(output / "validation.json"),
        "choices": choices,
        "normal_choices": sum(not r["permuted"] for r in choices),
        "control_choices": sum(r["permuted"] for r in choices),
        "all_choices_locked_before_outer_refits": True,
        "source_predictors_per_id": {
            f"{k[0]}-k{k[1]}-p{int(k[2])}": v.tolist() for k, v in counts.items()
        },
    }
    if checked_validation(output) != validated:
        raise ResearchError("Validasi berubah selama seleksi.")
    write_json(output / "selection.json", result)
    print(f"Locked {len(choices)} source-only choices", flush=True)
    return result


def evaluate(output):
    validated = checked_validation(output)
    selected = read_json(output / "selection.json")
    if (
        selected["status"] != "complete"
        or selected["validation_sha256"] != file_hash(output / "validation.json")
        or not selected["all_choices_locked_before_outer_refits"]
    ):
        raise ResearchError("Pilihan belum lengkap/valid.")
    if (output / "evaluation-lock.json").exists():
        raise ResearchError("Evaluasi sudah dimulai; jangan menimpa run.")
    selection_hash = file_hash(output / "selection.json")
    started = now()
    write_json(
        output / "evaluation-lock.json", {"created_at": started, "selection_sha256": selection_hash}
    )
    (output / "final").mkdir()
    labels, _, conditions = load_data()
    choices = {(r["family"], r["held"], r["k"], r["permuted"]): r for r in selected["choices"]}
    old = read_json(PILOT / "trials.json")
    old_trials = {(r["encoder"], r["array_index"]): r for r in old["trials"]}
    trials = []
    with (
        np.load(CACHE / "features.npz", allow_pickle=False) as features,
        threadpool_limits(limits=1),
    ):
        for encoder in ENCODERS:
            family, raw = encoder.split("_")[0], features[encoder].astype(np.float64)
            n, d = len(conditions), raw.shape[1]
            arrays = {
                "coef": np.empty((n, 12, d)),
                "intercept": np.empty((n, 12)),
                "mean": np.empty((n, d)),
                "scale": np.empty((n, d)),
                "recipe_index": np.empty(n, dtype=np.int64),
                "query_scores": np.empty((n, 60, 12)),
                "support_scores": np.full((n, 120, 12), np.nan),
                "support": np.full((n, 120), -1, dtype=np.int64),
                "query": np.empty((n, 60), dtype=np.int64),
                "fit_labels": np.full((n, 120), -1, dtype=np.int64),
            }
            for i, p in enumerate(conditions):
                choice = choices[family, p["subject"], p["k"], p["permuted"]]
                s, q = p["support"], p["query"]
                y = permuted_labels(labels, p) if p["permuted"] else labels[s]
                head, scores, support_scores, mean, scale = fit_selected(
                    raw[s], y, raw[q], choice["recipe"]
                )
                arrays["coef"][i], arrays["intercept"][i] = head.coef_, head.intercept_
                arrays["mean"][i], arrays["scale"][i] = mean, scale
                arrays["recipe_index"][i] = choice["recipe_index"]
                arrays["query_scores"][i], arrays["query"][i] = scores, q
                arrays["support_scores"][i, : len(s)] = support_scores
                arrays["support"][i, : len(s)], arrays["fit_labels"][i, : len(s)] = s, y
                trials.append(
                    {
                        "id": old_trials[encoder, i]["id"],
                        "encoder": encoder,
                        "array_index": i,
                        "subject": p["subject"],
                        "session_token": p["session_token"],
                        "k": p["k"],
                        "draw": p["draw"],
                        "permuted": p["permuted"],
                        "choice_id": choice["id"],
                        "recipe_index": choice["recipe_index"],
                        "target_classifier_labels": len(s),
                        "target_validation_labels": 0,
                        "source_labels_unique_total": choice["source_labels_unique_total"],
                        "query": metrics(labels[q], scores),
                        "baseline_query": old_trials[encoder, i]["query"],
                        "support_fit_label_accuracy": float(np.mean(support_scores.argmax(1) == y)),
                        "support_true_label_accuracy": float(
                            np.mean(support_scores.argmax(1) == labels[s])
                        ),
                    }
                )
            np.savez_compressed(output / "final" / f"{encoder}.npz", **arrays)
            print(f"Outer refits {encoder}: {len(conditions)} selected heads", flush=True)
    write_json(output / "trials.json", {"trials": trials, "zero_shot": old["zero_shot"]})
    if (
        file_hash(output / "selection.json") != selection_hash
        or checked_validation(output) != validated
    ):
        raise ResearchError("Pilihan/validasi berubah selama refit.")
    result = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": validated["input_sha256"],
        "evaluation_started_at": started,
        "selection_locked_at": selected["created_at"],
        "selection_sha256": selection_hash,
        "candidate_heads": validated["candidate_heads"],
        "new_candidate_heads": validated["new_candidate_heads"],
        "baseline_candidates_reused": 2448,
        "outer_refits": len(trials),
        "normal_choices": selected["normal_choices"],
        "control_choices": selected["control_choices"],
        "target_validation_labels": 0,
        "zero_shot_reused": len(old["zero_shot"]),
        "backbone_updates": 0,
    }
    result["artifact_sha256"] = {
        str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    write_json(output / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "select", "evaluate"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        {"validate": validate, "select": select, "evaluate": evaluate}[args.stage](args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
