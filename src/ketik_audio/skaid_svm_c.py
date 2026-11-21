"""Append a fixed high-C grid, select using source IDs, and refit on held supports."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

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
from .skaid_selection_core import FAMILIES, NORMALIZERS, source_budget, transform
from .skaid_svm import OUTPUT as OLD
from .skaid_svm import load_data
from .skaid_svm import signature as old_signature
from .skaid_svm_c_core import (
    ANCHORS,
    MAX_ITER,
    NEW_RECIPES,
    OLD_RECIPES,
    RECIPES,
    fit_gram,
    fit_selected,
    select_from_table,
)
from .skaid_svm_core import POLICIES, bandwidth, kernel_matrices
from .skaid_svm_core import fit_gram as old_fit_gram

OUTPUT = ROOT / "runs/skaid-svm-c"
PROTOCOL = ROOT / "research/SKAID_SVM_C_PROTOCOL.md"
OLD_AUDIT = ROOT / "reports/skaid-svm-audit"
OLD_REPORT = ROOT / "reports/skaid-svm"


def signature():
    sig = old_signature()
    previous = read_json(OLD / "results.json")
    verify(OLD, previous)
    if previous["input_sha256"] != sig:
        raise ResearchError("Sumber eksperimen SVM lama berubah.")
    verify(OLD_AUDIT, read_json(OLD_AUDIT / "audit.json"))
    verify(OLD_REPORT, read_json(OLD_REPORT / "summary.json"))
    paths = [
        Path(__file__),
        ROOT / "src/ketik_audio/skaid_svm_c_core.py",
        PROTOCOL,
        OLD / "results.json",
        OLD / "validation.json",
        OLD / "conditions.json",
        OLD / "selection.json",
        OLD / "budgets.json",
        OLD_AUDIT / "audit.json",
        OLD_REPORT / "summary.json",
    ]
    sig.update({str(p.relative_to(ROOT)): file_hash(p) for p in paths})
    return sig


def artifacts(output):
    return {
        str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()
    }


def fixed_data():
    labels, plans, conditions = load_data()
    if read_json(OLD / "conditions.json")["conditions"] != conditions:
        raise ResearchError("Kondisi tidak sama dengan studi SVM lama.")
    return labels, plans, conditions


def condition_labels(labels, p):
    return permuted_labels(labels, p) if p["permuted"] else labels[p["support"]]


def check_anchor(gram, cross, y, validation, ci, ri):
    c = RECIPES[ri]["c"]
    old, expanded = old_fit_gram(gram, y, c), fit_gram(gram, y, c)
    for name in ("dual_coef_", "intercept_", "classes_", "support_", "n_support_", "n_iter_"):
        if not np.array_equal(getattr(old, name), getattr(expanded, name)):
            raise ResearchError(f"Replay anchor mengubah {name}.")
    for actual, stored in (
        (expanded.decision_function(cross), validation["query_scores"][ci, ri]),
        (expanded.n_iter_, validation["n_iter"][ci, ri]),
        (expanded.n_support_, validation["n_support"][ci, ri]),
        (expanded.fit_status_, validation["fit_status"][ci, ri]),
    ):
        if not np.array_equal(actual, stored):
            raise ResearchError("Anchor tidak mereproduksi cache lama secara identik.")


def validate_encoder(output, encoder, raw, labels, conditions):
    started = perf_counter()
    with np.load(OLD / "validation" / f"{encoder}.npz", allow_pickle=False) as z:
        old = {k: z[k] for k in ("query_scores", "n_iter", "n_support", "fit_status")}
    n, d, r = len(conditions), raw.shape[1], len(NEW_RECIPES)
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
    for ci, p in enumerate(conditions):
        s, q, y = p["support"], p["query"], condition_labels(labels, p)
        for ni, mode in enumerate(NORMALIZERS):
            x, query, mean, scale = transform(raw[s], raw[q], mode)
            median, distances = bandwidth(x)
            arrays["mean"][ci, ni], arrays["scale"][ci, ni] = mean, scale
            arrays["median"][ci, ni] = median
            kernels = {}
            for column, recipe in enumerate(NEW_RECIPES):
                if recipe["normalizer"] != mode:
                    continue
                key = recipe["kernel"], recipe["multiplier"]
                if key not in kernels:
                    kernels[key] = kernel_matrices(x, query, recipe, median, distances)
                gram, cross, gamma = kernels[key]
                try:
                    head = fit_gram(gram, y, recipe["c"])
                    scores = head.decision_function(cross)
                    support = head.decision_function(gram)
                    if not np.isfinite(scores).all() or not np.isfinite(support).all():
                        raise ResearchError("Skor kandidat tidak finite.")
                except (ResearchError, ValueError) as exc:
                    write_json(
                        output / "validation-failure.json",
                        {
                            "created_at": now(),
                            "status": "failed",
                            "encoder": encoder,
                            "condition_index": ci,
                            "recipe_index": len(OLD_RECIPES) + column,
                            "recipe": recipe,
                            "reason": str(exc),
                        },
                    )
                    raise
                arrays["query_scores"][ci, column] = scores
                arrays["query_correct"][ci, column] = np.sum(scores.argmax(1) == labels[q])
                arrays["support_fit_correct"][ci, column] = np.sum(support.argmax(1) == y)
                arrays["support_true_correct"][ci, column] = np.sum(support.argmax(1) == labels[s])
                arrays["gamma"][ci, column] = gamma
                arrays["n_iter"][ci, column] = head.n_iter_
                arrays["n_support"][ci, column] = head.n_support_
                arrays["fit_status"][ci, column] = head.fit_status_
            for ri in ANCHORS:
                anchor = RECIPES[ri]
                if anchor["normalizer"] == mode:
                    gram, cross, _ = kernels[anchor["kernel"], anchor["multiplier"]]
                    check_anchor(gram, cross, y, old, ci, ri)
        if (ci + 1) % 51 == 0:
            print(f"Validasi C {encoder}: {ci + 1}/{n}", flush=True)
    np.savez_compressed(output / "validation" / f"{encoder}.npz", **arrays)
    return {
        "encoder": encoder,
        "new_candidates": n * r,
        "old_candidates_reused": n * len(OLD_RECIPES),
        "anchor_pairs_identical": n * len(ANCHORS),
        "bandwidth_fallbacks": int(np.sum(arrays["median"] <= 1e-12)),
        "maximum_binary_iterations": int(arrays["n_iter"].max()),
        "candidates_with_any_iteration_above_old_cap": int(
            np.sum(arrays["n_iter"].max(2) > 100000)
        ),
        "seconds": perf_counter() - started,
    }


def validate(output):
    sig = signature()
    env = environment()
    if env["packages"]["scikit-learn"] != "1.9.1":
        raise ResearchError("Protokol memakai scikit-learn 1.9.1.")
    labels, _, conditions = fixed_data()
    fresh_directory(output)
    (output / "validation").mkdir()
    write_json(
        output / "validation-lock.json",
        {
            "created_at": now(),
            "input_sha256": sig,
            "environment": env,
            "recipes": RECIPES,
            "old_recipe_count": len(OLD_RECIPES),
            "anchor_indices": ANCHORS,
            "max_iter": MAX_ITER,
        },
    )
    write_json(output / "conditions.json", {"conditions": conditions})
    (output / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    with (
        np.load(CACHE / "features.npz", allow_pickle=False) as features,
        threadpool_limits(limits=1),
    ):
        details = [
            validate_encoder(output, e, features[e].astype(np.float64), labels, conditions)
            for e in ENCODERS
        ]
    if signature() != sig:
        raise ResearchError("Masukan berubah selama validasi C.")
    new = sum(r["new_candidates"] for r in details)
    reused = sum(r["old_candidates_reused"] for r in details)
    record = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "new_candidates": new,
        "old_candidates_reused": reused,
        "total_candidates": new + reused,
        "anchor_pairs_identical": sum(r["anchor_pairs_identical"] for r in details),
        "all_fits_converged": True,
        "maximum_binary_iterations": max(r["maximum_binary_iterations"] for r in details),
        "candidates_with_any_iteration_above_old_cap": sum(
            r["candidates_with_any_iteration_above_old_cap"] for r in details
        ),
        "encoders": details,
        "artifact_sha256": artifacts(output),
    }
    write_json(output / "validation.json", record)
    return record


def checked_validation(output):
    record = read_json(output / "validation.json")
    verify(output, record)
    if record["input_sha256"] != signature() or record["total_candidates"] != 235008:
        raise ResearchError("Validasi C tidak sesuai masukan atau coverage lengkap.")
    return record


def aggregate_tables(output, conditions):
    tables, counts = {}, {}
    for encoder in ENCODERS:
        family = encoder.split("_")[0]
        with (
            np.load(OLD / "validation" / f"{encoder}.npz", allow_pickle=False) as a,
            np.load(output / "validation" / f"{encoder}.npz", allow_pickle=False) as b,
        ):
            correct = np.concatenate((a["query_correct"], b["query_correct"]), axis=1)
        if correct.shape != (len(conditions), len(RECIPES)) or np.any(
            (correct < 0) | (correct > 60)
        ):
            raise ResearchError("Count validasi C tidak lengkap atau tidak valid.")
        for ci, p in enumerate(conditions):
            key = family, p["k"], p["permuted"]
            if key not in tables:
                tables[key] = np.zeros((17, len(RECIPES)), dtype=np.int64)
                counts[key] = np.zeros(17, dtype=np.int64)
            tables[key][p["subject"]] += correct[ci]
            counts[key][p["subject"]] += 1
    for (family, _, _), values in counts.items():
        if not np.all(values == (3 if family in ("mfcc", "beats") else 9)):
            raise ResearchError("Kontribusi masing-masing ID tidak seimbang.")
    return tables


def choice_key(c):
    return c["policy"], c["family"], c["held"], c["k"], c["permuted"]


def select(output):
    validated = checked_validation(output)
    if (output / "selection.json").exists():
        raise ResearchError("Jangan menimpa pilihan C terkunci.")
    _, plans, conditions = fixed_data()
    tables = aggregate_tables(output, conditions)
    budgets = [source_budget(plans, h, k) for h in range(17) for k in KS]
    if budgets != read_json(OLD / "budgets.json")["budgets"]:
        raise ResearchError("Biaya label berubah dari SVM lama.")
    costs = {(b["held"], b["k"]): b["source_labels_unique_total"] for b in budgets}
    previous = {choice_key(c): c for c in read_json(OLD / "selection.json")["choices"]}
    choices = []
    for policy in POLICIES:
        for family in FAMILIES:
            for held in range(17):
                for k, permuted in [*((k, False) for k in KS), (10, True)]:
                    prev = previous[policy, family, held, k, permuted]
                    choice = select_from_table(tables[family, k, permuted], held, policy)
                    if (
                        choice["source_correct_by_recipe"][: len(OLD_RECIPES)]
                        != prev["source_correct_by_recipe"]
                        or choice["selected_source_correct"] < prev["selected_source_correct"]
                    ):
                        raise ResearchError("Grid lama atau kriteria sumber berubah.")
                    choices.append(
                        {
                            "id": f"c-{prev['id']}",
                            "family": family,
                            "k": k,
                            "permuted": permuted,
                            **choice,
                            "old_recipe_index": prev["recipe_index"],
                            "old_source_correct": prev["selected_source_correct"],
                            "source_labels_unique_total": costs[held, k],
                            "harrison_pretraining_labels_reused": (
                                720 if family == "pretrained" else 0
                            ),
                        }
                    )
    write_json(output / "budgets.json", {"budgets": budgets})
    record = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": validated["input_sha256"],
        "validation_sha256": file_hash(output / "validation.json"),
        "budgets_sha256": file_hash(output / "budgets.json"),
        "choices": choices,
        "all_choices_locked_before_outer_refits": True,
        "normal_choices": sum(not c["permuted"] for c in choices),
        "control_choices": sum(c["permuted"] for c in choices),
        "changed_choices": sum(c["recipe_index"] != c["old_recipe_index"] for c in choices),
    }
    if len(choices) != 816 or checked_validation(output) != validated:
        raise ResearchError("Coverage pilihan atau validasi tidak konsisten.")
    write_json(output / "selection.json", record)
    print(f"Terkunci: {len(choices)} pilihan grid C", flush=True)
    return record


def evaluate_encoder(output, encoder, raw, labels, conditions, choices, previous):
    with np.load(OLD / "final" / f"{encoder}.npz", allow_pickle=False) as z:
        old = {k: z[k] for k in z.files}
    with (
        np.load(OLD / "validation" / f"{encoder}.npz", allow_pickle=False) as a,
        np.load(output / "validation" / f"{encoder}.npz", allow_pickle=False) as b,
    ):
        candidate_scores = (a["query_scores"], b["query_scores"])
    arrays = {k: np.empty_like(v) for k, v in old.items()}
    for k in ("dual_coef", "support_scores"):
        arrays[k].fill(np.nan)
    for k in ("support_positions", "support", "fit_labels"):
        arrays[k].fill(-1)
    trials = []
    family = encoder.split("_")[0]
    for pi, policy in enumerate(POLICIES):
        for ci, p in enumerate(conditions):
            i = pi * len(conditions) + ci
            c = choices[policy, family, p["subject"], p["k"], p["permuted"]]
            ri = c["recipe_index"]
            s, q, y = p["support"], p["query"], condition_labels(labels, p)
            if (
                not np.array_equal(old["query"][i], q)
                or not np.array_equal(old["support"][i, : len(s)], s)
                or not np.array_equal(old["fit_labels"][i, : len(s)], y)
                or old["recipe_index"][i] != c["old_recipe_index"]
                or RECIPES[ri] != c["recipe"]
            ):
                raise ResearchError("Split, label atau resep C tidak kompatibel dengan baseline.")
            try:
                head, scores, support, mean, scale, median, gamma = fit_selected(
                    raw[s], y, raw[q], c["recipe"]
                )
            except (ResearchError, ValueError) as exc:
                write_json(
                    output / "evaluation-failure.json",
                    {
                        "created_at": now(),
                        "status": "failed",
                        "encoder": encoder,
                        "condition_index": ci,
                        "choice_id": c["id"],
                        "reason": str(exc),
                    },
                )
                raise
            expected = (
                candidate_scores[0][ci, ri]
                if ri < len(OLD_RECIPES)
                else candidate_scores[1][ci, ri - len(OLD_RECIPES)]
            )
            if not np.array_equal(scores, expected):
                raise ResearchError("Refit luar berbeda dari kandidat dengan resep sama.")
            ns = len(head.support_)
            arrays["dual_coef"][i, :, :ns] = head.dual_coef_
            arrays["intercept"][i], arrays["n_support"][i] = head.intercept_, head.n_support_
            arrays["support_positions"][i, :ns] = head.support_
            arrays["mean"][i], arrays["scale"][i] = mean, scale
            arrays["median"][i], arrays["gamma"][i] = median, gamma
            arrays["recipe_index"][i] = ri
            arrays["query_scores"][i], arrays["query"][i] = scores, q
            arrays["support_scores"][i, : len(s)] = support
            arrays["support"][i, : len(s)], arrays["fit_labels"][i, : len(s)] = s, y
            arrays["n_iter"][i] = head.n_iter_
            unchanged = ri == c["old_recipe_index"]
            if unchanged and any(
                not np.array_equal(v[i], old[k][i], equal_nan=True) for k, v in arrays.items()
            ):
                raise ResearchError("Pilihan lama yang tetap terpilih berubah pada refit.")
            prev = previous[encoder, i]
            trials.append(
                {
                    **prev,
                    "id": f"c:{prev['id']}",
                    "choice_id": c["id"],
                    "recipe_index": ri,
                    "old_recipe_index": c["old_recipe_index"],
                    "recipe_unchanged": unchanged,
                    "source_labels_unique_total": c["source_labels_unique_total"],
                    "old_svm_query": prev["query"],
                    "query": metrics(labels[q], scores),
                    "support_fit_label_accuracy": float(np.mean(support.argmax(1) == y)),
                    "support_true_label_accuracy": float(np.mean(support.argmax(1) == labels[s])),
                }
            )
    np.savez_compressed(output / "final" / f"{encoder}.npz", **arrays)
    print(f"Evaluasi C {encoder}: {len(trials)} kepala final", flush=True)
    return trials


def evaluate(output):
    validated = checked_validation(output)
    selected = read_json(output / "selection.json")
    if (
        selected["status"] != "complete"
        or selected["validation_sha256"] != file_hash(output / "validation.json")
        or selected["budgets_sha256"] != file_hash(output / "budgets.json")
        or not selected["all_choices_locked_before_outer_refits"]
        or selected["input_sha256"] != validated["input_sha256"]
    ):
        raise ResearchError("Seleksi C belum lengkap atau tidak cocok.")
    if (output / "evaluation-lock.json").exists():
        raise ResearchError("Evaluasi C sudah dimulai; gunakan direktori baru.")
    started, selection_hash = now(), file_hash(output / "selection.json")
    if not validated["created_at"] < selected["created_at"] < started:
        raise ResearchError("Urutan validasi, seleksi dan evaluasi C tidak valid.")
    write_json(
        output / "evaluation-lock.json", {"created_at": started, "selection_sha256": selection_hash}
    )
    (output / "final").mkdir()
    labels, _, conditions = fixed_data()
    choices = {choice_key(c): c for c in selected["choices"]}
    old = read_json(OLD / "trials.json")
    previous = {(t["encoder"], t["array_index"]): t for t in old["trials"]}
    trials = []
    with (
        np.load(CACHE / "features.npz", allow_pickle=False) as features,
        threadpool_limits(limits=1),
    ):
        for encoder in ENCODERS:
            trials += evaluate_encoder(
                output,
                encoder,
                features[encoder].astype(np.float64),
                labels,
                conditions,
                choices,
                previous,
            )
    if len(trials) != 4896 or len(choices) != 816:
        raise ResearchError("Refit C tidak mencakup seluruh kondisi.")
    write_json(output / "trials.json", {"trials": trials, "zero_shot": old["zero_shot"]})
    if (
        file_hash(output / "selection.json") != selection_hash
        or checked_validation(output) != validated
    ):
        raise ResearchError("Validasi atau pilihan berubah selama refit C.")
    record = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": validated["input_sha256"],
        "evaluation_started_at": started,
        "selection_locked_at": selected["created_at"],
        "selection_sha256": selection_hash,
        "new_candidates": validated["new_candidates"],
        "old_candidates_reused": validated["old_candidates_reused"],
        "total_candidates": validated["total_candidates"],
        "outer_refits": len(trials),
        "outer_refits_with_unchanged_recipe_exact": sum(t["recipe_unchanged"] for t in trials),
        "normal_choices": selected["normal_choices"],
        "control_choices": selected["control_choices"],
        "changed_choices": selected["changed_choices"],
        "backbone_updates": 0,
        "new_encoder_extractions": 0,
        "target_validation_labels": 0,
        "zero_shot_reused": len(old["zero_shot"]),
        "artifact_sha256": artifacts(output),
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
