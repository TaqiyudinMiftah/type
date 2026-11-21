"""Validate fixed SVM recipes on source IDs, lock choices, then refit held-ID heads."""

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
from .skaid_pilot_common import CACHE, ENCODERS, KS, ROOT, verify
from .skaid_pilot_probe import metrics, permuted_labels
from .skaid_selection import OUTPUT as RIDGE
from .skaid_selection import load_data
from .skaid_selection import signature as ridge_signature
from .skaid_selection_core import FAMILIES, NORMALIZERS, source_budget, transform
from .skaid_svm_core import (
    POLICIES,
    RECIPES,
    bandwidth,
    fit_gram,
    fit_selected,
    kernel_matrices,
    select_from_table,
)

PROTOCOL = ROOT / "research/SKAID_SVM_PROTOCOL.md"
OUTPUT = ROOT / "runs/skaid-svm"


def signature():
    result = ridge_signature()
    verify(RIDGE, read_json(RIDGE / "results.json"))
    audit = ROOT / "reports/skaid-selection-audit"
    verify(audit, read_json(audit / "audit.json"))
    paths = [
        Path(__file__),
        ROOT / "src/ketik_audio/skaid_svm_core.py",
        PROTOCOL,
        RIDGE / "results.json",
        RIDGE / "trials.json",
        audit / "audit.json",
    ]
    result.update({str(p.relative_to(ROOT)): file_hash(p) for p in paths})
    return result


def validate(output):
    sig = signature()
    fresh_directory(output)
    (output / "validation").mkdir()
    labels, _, conditions = load_data()
    write_json(output / "conditions.json", {"conditions": conditions})
    write_json(
        output / "validation-lock.json",
        {
            "created_at": now(),
            "input_sha256": sig,
            "environment": environment(),
            "recipes": RECIPES,
        },
    )
    (output / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    heads, fallback, max_iterations = 0, 0, 0
    with (
        np.load(CACHE / "features.npz", allow_pickle=False) as features,
        threadpool_limits(limits=1),
    ):
        for encoder in ENCODERS:
            raw = features[encoder].astype(np.float64)
            n, d, r = len(conditions), raw.shape[1], len(RECIPES)
            arrays = {
                "query_scores": np.empty((n, r, 60, 12)),
                "query_correct": np.empty((n, r), dtype=np.int64),
                "support_fit_correct": np.empty((n, r), dtype=np.int64),
                "support_true_correct": np.empty((n, r), dtype=np.int64),
                "mean": np.empty((n, 2, d)),
                "scale": np.empty((n, 2, d)),
                "median": np.empty((n, 2)),
                "gamma": np.empty((n, r)),
                "n_iter": np.empty((n, r, 66), dtype=np.int32),
                "n_support": np.empty((n, r, 12), dtype=np.int16),
                "fit_status": np.empty((n, r), dtype=np.int8),
            }
            for i, p in enumerate(conditions):
                s, q = p["support"], p["query"]
                y = permuted_labels(labels, p) if p["permuted"] else labels[s]
                for j, mode in enumerate(NORMALIZERS):
                    x, query, mean, scale = transform(raw[s], raw[q], mode)
                    median, distances = bandwidth(x)
                    arrays["mean"][i, j], arrays["scale"][i, j] = mean, scale
                    arrays["median"][i, j] = median
                    fallback += int(median <= 1e-12)
                    kernels = {}
                    for ri, recipe in enumerate(RECIPES):
                        if recipe["normalizer"] != mode:
                            continue
                        key = recipe["kernel"], recipe["multiplier"]
                        if key not in kernels:
                            kernels[key] = kernel_matrices(x, query, recipe, median, distances)
                        gram, cross, gamma = kernels[key]
                        head = fit_gram(gram, y, recipe["c"])
                        scores, support = (
                            head.decision_function(cross),
                            head.decision_function(gram),
                        )
                        if not np.isfinite(scores).all() or not np.isfinite(support).all():
                            raise ResearchError("Skor kandidat tidak finite.")
                        arrays["query_scores"][i, ri] = scores
                        arrays["query_correct"][i, ri] = np.sum(scores.argmax(1) == labels[q])
                        arrays["support_fit_correct"][i, ri] = np.sum(support.argmax(1) == y)
                        arrays["support_true_correct"][i, ri] = np.sum(
                            support.argmax(1) == labels[s]
                        )
                        arrays["gamma"][i, ri] = gamma
                        arrays["n_iter"][i, ri] = head.n_iter_
                        arrays["n_support"][i, ri] = head.n_support_
                        arrays["fit_status"][i, ri] = head.fit_status_
                        max_iterations = max(max_iterations, int(head.n_iter_.max()))
                        heads += 1
                if (i + 1) % 51 == 0:
                    print(f"Source {encoder}: {i + 1}/{n} conditions", flush=True)
            np.savez_compressed(output / "validation" / f"{encoder}.npz", **arrays)
    if signature() != sig:
        raise ResearchError("Masukan berubah selama validasi.")
    record = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "candidate_heads": heads,
        "support_bandwidth_fallbacks": fallback,
        "max_binary_iterations": max_iterations,
        "all_fits_converged": True,
        "role": "development cross-ID validation on previously inspected cohort",
    }
    record["artifact_sha256"] = {
        str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    write_json(output / "validation.json", record)
    return record


def checked_validation(output):
    record = read_json(output / "validation.json")
    verify(output, record)
    if record["input_sha256"] != signature():
        raise ResearchError("Masukan/kode validasi berubah.")
    return record


def aggregate_tables(output, conditions):
    tables, counts = {}, {}
    for encoder in ENCODERS:
        family = encoder.split("_")[0]
        with np.load(output / "validation" / f"{encoder}.npz", allow_pickle=False) as z:
            correct = z["query_correct"]
        for i, p in enumerate(conditions):
            key = family, p["k"], p["permuted"]
            if key not in tables:
                tables[key] = np.zeros((17, len(RECIPES)), dtype=np.int64)
                counts[key] = np.zeros(17, dtype=np.int64)
            tables[key][p["subject"]] += correct[i]
            counts[key][p["subject"]] += 1
    for key, value in counts.items():
        if not np.array_equal(value, np.full(17, 3 if key[0] in ("mfcc", "beats") else 9)):
            raise ResearchError("Sumber tidak lengkap atau tidak seimbang.")
    return tables


def select(output):
    validated = checked_validation(output)
    if (output / "selection.json").exists():
        raise ResearchError("Jangan menimpa seleksi terkunci.")
    _, plans, conditions = load_data()
    tables = aggregate_tables(output, conditions)
    budgets = [source_budget(plans, held, k) for held in range(17) for k in KS]
    choices = []
    for policy in POLICIES:
        for family in FAMILIES:
            for held in range(17):
                for k, permuted in [*((k, False) for k in KS), (10, True)]:
                    choice = select_from_table(tables[family, k, permuted], held, policy)
                    budget = next(b for b in budgets if b["held"] == held and b["k"] == k)
                    choices.append(
                        {
                            "id": f"{policy}-{family}-h{held:02d}-k{k}-p{int(permuted)}",
                            "family": family,
                            "k": k,
                            "permuted": permuted,
                            **choice,
                            "source_labels_unique_total": budget["source_labels_unique_total"],
                            "harrison_pretraining_labels_reused": 720
                            if family == "pretrained"
                            else 0,
                        }
                    )
    write_json(output / "budgets.json", {"budgets": budgets})
    record = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": validated["input_sha256"],
        "validation_sha256": file_hash(output / "validation.json"),
        "choices": choices,
        "all_choices_locked_before_outer_refits": True,
        "normal_choices": sum(not c["permuted"] for c in choices),
        "control_choices": sum(c["permuted"] for c in choices),
    }
    if checked_validation(output) != validated:
        raise ResearchError("Validasi berubah selama seleksi.")
    write_json(output / "selection.json", record)
    print(f"Locked {len(choices)} choices", flush=True)
    return record


def evaluate(output):
    validated = checked_validation(output)
    selected = read_json(output / "selection.json")
    if (
        selected["status"] != "complete"
        or selected["validation_sha256"] != file_hash(output / "validation.json")
        or not selected["all_choices_locked_before_outer_refits"]
        or selected["input_sha256"] != validated["input_sha256"]
    ):
        raise ResearchError("Seleksi belum lengkap/valid.")
    if (output / "evaluation-lock.json").exists():
        raise ResearchError("Evaluasi sudah dimulai; jangan menimpa run.")
    started, selection_hash = now(), file_hash(output / "selection.json")
    write_json(
        output / "evaluation-lock.json", {"created_at": started, "selection_sha256": selection_hash}
    )
    (output / "final").mkdir()
    labels, _, conditions = load_data()
    choices = {
        (c["policy"], c["family"], c["held"], c["k"], c["permuted"]): c for c in selected["choices"]
    }
    old = read_json(RIDGE / "trials.json")
    ridge = {(r["encoder"], r["array_index"]): r for r in old["trials"]}
    trials = []
    with (
        np.load(CACHE / "features.npz", allow_pickle=False) as features,
        threadpool_limits(limits=1),
    ):
        for encoder in ENCODERS:
            raw, family = features[encoder].astype(np.float64), encoder.split("_")[0]
            n, d = len(conditions) * 2, raw.shape[1]
            arrays = {
                "dual_coef": np.full((n, 11, 120), np.nan),
                "intercept": np.empty((n, 66)),
                "n_support": np.empty((n, 12), dtype=np.int64),
                "support_positions": np.full((n, 120), -1, dtype=np.int64),
                "mean": np.empty((n, d)),
                "scale": np.empty((n, d)),
                "median": np.empty(n),
                "gamma": np.empty(n),
                "recipe_index": np.empty(n, dtype=np.int64),
                "query_scores": np.empty((n, 60, 12)),
                "support_scores": np.full((n, 120, 12), np.nan),
                "support": np.full((n, 120), -1, dtype=np.int64),
                "query": np.empty((n, 60), dtype=np.int64),
                "fit_labels": np.full((n, 120), -1, dtype=np.int64),
                "n_iter": np.empty((n, 66), dtype=np.int32),
            }
            for pi, policy in enumerate(POLICIES):
                for ci, p in enumerate(conditions):
                    i = pi * len(conditions) + ci
                    choice = choices[policy, family, p["subject"], p["k"], p["permuted"]]
                    s, q = p["support"], p["query"]
                    y = permuted_labels(labels, p) if p["permuted"] else labels[s]
                    head, scores, support, mean, scale, median, gamma = fit_selected(
                        raw[s], y, raw[q], choice["recipe"]
                    )
                    ns = len(head.support_)
                    arrays["dual_coef"][i, :, :ns] = head.dual_coef_
                    arrays["intercept"][i], arrays["n_support"][i] = (
                        head.intercept_,
                        head.n_support_,
                    )
                    arrays["support_positions"][i, :ns] = head.support_
                    arrays["mean"][i], arrays["scale"][i] = mean, scale
                    arrays["median"][i], arrays["gamma"][i] = median, gamma
                    arrays["recipe_index"][i] = choice["recipe_index"]
                    arrays["query_scores"][i], arrays["query"][i] = scores, q
                    arrays["support_scores"][i, : len(s)] = support
                    arrays["support"][i, : len(s)], arrays["fit_labels"][i, : len(s)] = s, y
                    arrays["n_iter"][i] = head.n_iter_
                    trials.append(
                        {
                            "id": f"{policy}:{ridge[encoder, ci]['id']}",
                            "encoder": encoder,
                            "array_index": i,
                            "condition_index": ci,
                            "policy": policy,
                            "subject": p["subject"],
                            "session_token": p["session_token"],
                            "k": p["k"],
                            "draw": p["draw"],
                            "permuted": p["permuted"],
                            "choice_id": choice["id"],
                            "recipe_index": choice["recipe_index"],
                            "target_classifier_labels": len(s),
                            "source_labels_unique_total": choice["source_labels_unique_total"],
                            "target_validation_labels": 0,
                            "query": metrics(labels[q], scores),
                            "ridge_query": ridge[encoder, ci]["query"],
                            "support_fit_label_accuracy": float(np.mean(support.argmax(1) == y)),
                            "support_true_label_accuracy": float(
                                np.mean(support.argmax(1) == labels[s])
                            ),
                        }
                    )
            np.savez_compressed(output / "final" / f"{encoder}.npz", **arrays)
            print(f"Final {encoder}: {n} selected heads", flush=True)
    write_json(output / "trials.json", {"trials": trials, "zero_shot": old["zero_shot"]})
    if (
        file_hash(output / "selection.json") != selection_hash
        or checked_validation(output) != validated
    ):
        raise ResearchError("Pilihan/validasi berubah selama refit.")
    record = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": validated["input_sha256"],
        "evaluation_started_at": started,
        "selection_locked_at": selected["created_at"],
        "selection_sha256": selection_hash,
        "candidate_heads": validated["candidate_heads"],
        "outer_refits": len(trials),
        "normal_choices": selected["normal_choices"],
        "control_choices": selected["control_choices"],
        "backbone_updates": 0,
        "new_encoder_extractions": 0,
        "zero_shot_reused": len(old["zero_shot"]),
    }
    record["artifact_sha256"] = {
        str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    write_json(output / "results.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "select", "evaluate"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        {"validate": validate, "select": select, "evaluate": evaluate}[args.stage](args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
