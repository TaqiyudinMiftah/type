"""Support-only transforms, ridge paths and source-ID-only recipe selection."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import normalize

from .common import ResearchError

ALPHAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
NORMALIZERS = ("row_l2", "support_z_l2")
RECIPES = tuple({"normalizer": mode, "alpha": alpha} for mode in NORMALIZERS for alpha in ALPHAS)
BASELINE_INDEX = ALPHAS.index(1.0)
FAMILIES = ("mfcc", "beats", "pretrained", "random")


def transform(support, query, mode):
    support, query = (np.asarray(v, dtype=np.float64) for v in (support, query))
    if (
        support.ndim != 2
        or query.ndim != 2
        or not len(support)
        or support.shape[1] != query.shape[1]
        or not np.isfinite(support).all()
        or not np.isfinite(query).all()
    ):
        raise ResearchError("Fitur dukungan/query tidak valid.")
    if mode == "row_l2":
        mean, scale = np.zeros(support.shape[1]), np.ones(support.shape[1])
    elif mode == "support_z_l2":
        mean, sd = support.mean(0), support.std(0, ddof=0)
        scale = np.where(sd <= 1e-12, 1.0, sd)
    else:
        raise ResearchError("Normalizer tidak dikenal.")
    return normalize((support - mean) / scale), normalize((query - mean) / scale), mean, scale


def ridge_path(x, labels, query):
    labels = np.asarray(labels, dtype=np.int64)
    if len(x) != len(labels) or set(labels.tolist()) != set(range(12)):
        raise ResearchError("Ridge path memerlukan seluruh 12 kelas.")
    target = 2 * np.eye(12)[labels] - 1
    x_mean, y_mean = x.mean(0), target.mean(0)
    xc, yc = x - x_mean, target - y_mean
    gram = xc @ xc.T
    eigenvalues, vectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0)
    projected_y = vectors.T @ yc
    query_basis = ((query - x_mean) @ xc.T) @ vectors
    support_basis = gram @ vectors
    queries, supports = [], []
    for alpha in ALPHAS:
        dual = projected_y / (eigenvalues[:, None] + alpha)
        queries.append(query_basis @ dual + y_mean)
        supports.append(support_basis @ dual + y_mean)
    return np.stack(queries), np.stack(supports)


def fit_selected(support, labels, query, recipe):
    x, q, mean, scale = transform(support, query, recipe["normalizer"])
    if set(np.asarray(labels).tolist()) != set(range(12)):
        raise ResearchError("Kepala terpilih memerlukan seluruh 12 kelas.")
    model = RidgeClassifier(
        alpha=recipe["alpha"], solver="cholesky", fit_intercept=True, class_weight=None
    )
    model.fit(x, labels)
    if not np.array_equal(model.classes_, np.arange(12)):
        raise ResearchError("Urutan kelas berubah.")
    return model, model.decision_function(q), model.decision_function(x), mean, scale


def choose_recipe(dev_ids, correct_by_id, held):
    ids = np.asarray(dev_ids, dtype=np.int64)
    values = np.asarray(correct_by_id)
    if (
        held not in range(17)
        or set(ids.tolist()) != set(range(17)) - {held}
        or len(ids) != 16
        or values.shape != (16, len(RECIPES))
    ):
        raise ResearchError("Seleksi harus menerima tepat16 ID sumber selain held ID.")
    if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
        raise ResearchError("Kriteria sumber harus hitungan benar integer nonnegatif.")
    totals = values.sum(0, dtype=np.int64)
    index = max(
        range(len(RECIPES)),
        key=lambda i: (
            int(totals[i]),
            RECIPES[i]["alpha"],
            -NORMALIZERS.index(RECIPES[i]["normalizer"]),
        ),
    )
    return {
        "held": held,
        "source_ids": sorted(ids.tolist()),
        "recipe_index": index,
        "recipe": RECIPES[index],
        "source_correct_by_recipe": totals.tolist(),
        "selected_source_correct": int(totals[index]),
    }


def select_from_table(correct_by_id, held):
    table = np.asarray(correct_by_id)
    if table.shape != (17, len(RECIPES)):
        raise ResearchError("Tabel ID sumber tidak lengkap.")
    dev = [i for i in range(17) if i != held]
    return choose_recipe(dev, table[dev], held)


def source_budget(plans, held, k):
    dev = [p for p in plans if p["subject"] != held and p["k"] == k]
    if {p["subject"] for p in dev} != set(range(17)) - {held} or len(dev) != 48:
        raise ResearchError("Rencana label sumber tidak lengkap.")
    support = sorted({i for p in dev for i in p["support"]})
    query = sorted({i for p in dev for i in p["query"]})
    if set(support) & set(query) or len(query) != 960:
        raise ResearchError("Label sumber berulang lintas peran atau query tidak lengkap.")
    return {
        "held": held,
        "k": k,
        "source_ids": sorted({p["subject"] for p in dev}),
        "source_support_indices": support,
        "source_query_indices": query,
        "source_support_labels_unique": len(support),
        "source_query_labels_unique": len(query),
        "source_labels_unique_total": len(set(support) | set(query)),
        "target_classifier_labels_per_trial": 12 * k,
    }
