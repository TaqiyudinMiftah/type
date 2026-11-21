"""Matched target-label budgets with fixed linear probes on frozen SKAID features."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
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
from .skaid_pilot_common import CACHE, DATA, ENCODERS, KEYS, PROTOCOL, ROOT, RUN, SEEDS, verify


def signature():
    verify(DATA, read_json(DATA / "manifest.json"))
    verify(CACHE, read_json(CACHE / "manifest.json"))
    paths = [
        Path(__file__),
        ROOT / "src/ketik_audio/skaid_pilot_common.py",
        PROTOCOL,
        DATA / "manifest.json",
        DATA / "plan.json",
        DATA / "records.json",
        CACHE / "manifest.json",
        CACHE / "features.npz",
    ]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def feature_rows(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ResearchError("Matriks fitur harus finite.")
    return normalize(x, norm="l2", axis=1)


def permuted_labels(labels, plan):
    support = np.asarray(plan["support"], dtype=int)
    values = labels[support]
    if plan["k"] != 10 or not np.array_equal(np.bincount(values, minlength=12), np.full(12, 10)):
        raise ResearchError("Kontrol harus K10 seimbang.")
    rng = np.random.default_rng(997 + 100000 * plan["subject"] + plan["draw"])
    return rng.permutation(values)


def fit_head(x, labels, support, query, fit_labels=None):
    support, query = np.asarray(support, dtype=np.int64), np.asarray(query, dtype=np.int64)
    y = labels[support] if fit_labels is None else np.asarray(fit_labels, dtype=np.int64)
    if (
        len(set(support.tolist())) != len(support)
        or set(support.tolist()) & set(query.tolist())
        or len(y) != len(support)
        or set(y.tolist()) != set(range(12))
    ):
        raise ResearchError("Dukungan/query/fit labels tidak sesuai.")
    head = RidgeClassifier(alpha=1.0, solver="cholesky", fit_intercept=True, class_weight=None)
    with threadpool_limits(limits=4):
        head.fit(x[support], y)
        query_scores = head.decision_function(x[query])
        support_scores = head.decision_function(x[support])
    if not np.array_equal(head.classes_, np.arange(12)):
        raise ResearchError("Urutan kelas kepala berubah.")
    return head, query_scores, support_scores, y


def metrics(labels, scores):
    if scores.shape != (len(labels), 12) or not np.isfinite(scores).all():
        raise ResearchError("Skor query tidak sesuai.")
    prediction = scores.argmax(axis=1)
    top = np.argsort(-scores, axis=1, kind="stable")[:, :3]
    return {
        "examples": len(labels),
        "correct": int((prediction == labels).sum()),
        "accuracy": float(np.mean(prediction == labels)),
        "macro_f1": float(
            f1_score(labels, prediction, labels=np.arange(12), average="macro", zero_division=0)
        ),
        "top3": float(np.mean(np.any(top == labels[:, None], axis=1))),
    }


def validate_plan(records, plans, labels):
    if len(records) != len(labels) or len({r["id"] for r in records}) != len(records):
        raise ResearchError("Identitas/label clip tidak sesuai.")
    for i, row in enumerate(records):
        if (
            row["clip_index"] != i
            or row["label"] != int(labels[i])
            or row["key"] != KEYS[int(labels[i])]
        ):
            raise ResearchError("Urutan/kelas clip berbeda.")
    for p in plans:
        s, q = p["support"], p["query"]
        if (
            len(s) != 12 * p["k"]
            or len(q) != 60
            or len(set(s)) != len(s)
            or len(set(q)) != len(q)
            or set(s) & set(q)
        ):
            raise ResearchError("Anggaran atau disjointness split salah.")
        if not np.array_equal(
            np.bincount(labels[s], minlength=12), np.full(12, p["k"])
        ) or not np.array_equal(np.bincount(labels[q], minlength=12), np.full(12, 5)):
            raise ResearchError("Split tidak seimbang.")
        if {records[i]["phase"] for i in s} != {"Phase 1"} or {records[i]["phase"] for i in q} != {
            "Phase 2"
        }:
            raise ResearchError("Fase kalibrasi/query salah.")
        if {records[i]["audio"] for i in s} & {records[i]["audio"] for i in q}:
            raise ResearchError("Split berbagi raw recording.")
        if {records[i]["session_token"] for i in s + q} != {p["session_token"]}:
            raise ResearchError("Split mencampur ID.")


def run(output):
    sig = signature()
    fresh_directory(output)
    with np.load(DATA / "segments.npz", allow_pickle=False) as z:
        labels = z["y"]
    records = read_json(DATA / "records.json")["records"]
    plans = read_json(DATA / "plan.json")["plans"]
    validate_plan(records, plans, labels)
    conditions = [{**p, "permuted": False} for p in plans] + [
        {**p, "permuted": True} for p in plans if p["k"] == 10
    ]
    write_json(
        output / "trials-locked.json",
        {
            "created_at": now(),
            "input_sha256": sig,
            "encoders": ENCODERS,
            "conditions_per_encoder": conditions,
            "alpha": 1.0,
            "normalization": "per-row L2 only",
        },
    )
    write_json(
        output / "lock.json",
        {"created_at": now(), "input_sha256": sig, "environment": environment()},
    )
    (output / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    trials = []
    with np.load(CACHE / "features.npz", allow_pickle=False) as features:
        for encoder in ENCODERS:
            x = feature_rows(features[encoder])
            n, d = len(conditions), x.shape[1]
            arrays = {
                "coef": np.empty((n, 12, d)),
                "intercept": np.empty((n, 12)),
                "query_scores": np.empty((n, 60, 12)),
                "support_scores": np.full((n, 120, 12), np.nan),
                "support": np.full((n, 120), -1, dtype=np.int64),
                "query": np.empty((n, 60), dtype=np.int64),
                "fit_labels": np.full((n, 120), -1, dtype=np.int64),
            }
            for index, p in enumerate(conditions):
                s, q = p["support"], p["query"]
                fit_y = permuted_labels(labels, p) if p["permuted"] else None
                started = time.perf_counter()
                head, query_scores, support_scores, fit_y = fit_head(x, labels, s, q, fit_y)
                elapsed = time.perf_counter() - started
                arrays["coef"][index], arrays["intercept"][index] = head.coef_, head.intercept_
                arrays["query_scores"][index], arrays["query"][index] = query_scores, q
                arrays["support_scores"][index, : len(s)] = support_scores
                arrays["support"][index, : len(s)] = s
                arrays["fit_labels"][index, : len(s)] = fit_y
                trials.append(
                    {
                        "id": f"{encoder}:{p['id']}:{'permuted' if p['permuted'] else 'normal'}",
                        "encoder": encoder,
                        "array_index": index,
                        "subject": p["subject"],
                        "session_token": p["session_token"],
                        "k": p["k"],
                        "draw": p["draw"],
                        "permuted": p["permuted"],
                        "target_classifier_labels": len(s),
                        "target_validation_labels": 0,
                        "query": metrics(labels[q], query_scores),
                        "support_fit_label_accuracy": float(
                            np.mean(support_scores.argmax(1) == fit_y)
                        ),
                        "support_true_label_accuracy": float(
                            np.mean(support_scores.argmax(1) == labels[s])
                        ),
                        "fit_and_prediction_seconds": elapsed,
                    }
                )
            np.savez_compressed(output / f"{encoder}.npz", **arrays)
            print(f"Completed {encoder}: {len(conditions)} heads", flush=True)
        # The source class ordering is imported only for immutable head-column mapping.
        from .harrison import KEYS as source_keys

        columns = [source_keys.index(k) for k in KEYS]
        query_plans = [p for p in plans if p["k"] == 1 and p["draw"] == 101]
        zero, scores, queries = [], [], []
        for seed in SEEDS:
            logits = features[f"source_logits_s{seed}"][:, columns]
            for p in query_plans:
                q = p["query"]
                score = logits[q]
                zero.append(
                    {
                        "source_seed": seed,
                        "subject": p["subject"],
                        "session_token": p["session_token"],
                        "target_classifier_labels": 0,
                        "query": metrics(labels[q], score),
                    }
                )
                scores.append(score)
                queries.append(q)
        np.savez_compressed(
            output / "zero-shot.npz", scores=np.stack(scores), query=np.asarray(queries)
        )
    if signature() != sig:
        raise ResearchError("Masukan/kode berubah selama klasifikasi.")
    write_json(output / "trials.json", {"trials": trials, "zero_shot": zero})
    result = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "subjects": 17,
        "keys": KEYS,
        "query_examples_unique": 1020,
        "normal_heads": sum(not t["permuted"] for t in trials),
        "permuted_heads": sum(t["permuted"] for t in trials),
        "zero_shot_evaluations": len(zero),
        "oracle_segmentation_and_sampling": True,
        "backbone_updates": 0,
        "target_validation_labels": 0,
    }
    result["artifact_sha256"] = {
        p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()
    }
    write_json(output / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUN)
    args = parser.parse_args()
    try:
        run(args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
