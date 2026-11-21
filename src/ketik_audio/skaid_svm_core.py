"""Fixed SVM recipes, support-only bandwidths and held-ID-excluding selection."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import SVC

from .common import ResearchError
from .skaid_selection_core import NORMALIZERS, transform

CS = (0.01, 0.1, 1.0, 10.0, 100.0)
MULTIPLIERS = (0.01, 0.1, 1.0, 10.0, 100.0)
POLICIES = ("linear", "rbf")
RECIPES = tuple(
    {"kernel": "linear", "normalizer": mode, "c": c, "multiplier": None}
    for mode in NORMALIZERS
    for c in CS
) + tuple(
    {"kernel": "rbf", "normalizer": mode, "c": c, "multiplier": g}
    for mode in NORMALIZERS
    for g in MULTIPLIERS
    for c in CS
)


def bandwidth(x):
    distances = cdist(x, x, metric="sqeuclidean")
    median = float(np.median(distances[np.triu_indices(len(x), k=1)]))
    if not np.isfinite(median) or median < 0:
        raise ResearchError("Median jarak dukungan tidak valid.")
    return median, distances


def kernel_matrices(x, q, recipe, median=None, distances=None):
    if recipe["kernel"] == "linear":
        return x @ x.T, q @ x.T, 0.0
    if recipe["kernel"] != "rbf":
        raise ResearchError("Kernel tidak dikenal.")
    if median is None or distances is None:
        median, distances = bandwidth(x)
    gamma = recipe["multiplier"] / median if median > 1e-12 else recipe["multiplier"]
    return np.exp(-gamma * distances), np.exp(-gamma * cdist(q, x, "sqeuclidean")), gamma


def svc(c, kernel="precomputed", gamma="scale"):
    return SVC(
        C=c,
        kernel=kernel,
        gamma=gamma,
        class_weight=None,
        shrinking=False,
        tol=1e-6,
        max_iter=100000,
        decision_function_shape="ovr",
        break_ties=True,
        cache_size=128,
    )


def fit_gram(gram, labels, c):
    if set(np.asarray(labels).tolist()) != set(range(12)):
        raise ResearchError("SVM memerlukan seluruh 12 kelas.")
    if not np.isfinite(gram).all():
        raise ResearchError("Gram tidak finite.")
    head = svc(c).fit(gram, labels)
    if head.fit_status_ != 0 or not np.array_equal(head.classes_, np.arange(12)):
        raise ResearchError("SVM tidak konvergen atau urutan kelas berubah.")
    return head


def fit_selected(support, labels, query, recipe):
    x, q, mean, scale = transform(support, query, recipe["normalizer"])
    median, distances = bandwidth(x)
    gram, cross, gamma = kernel_matrices(x, q, recipe, median, distances)
    head = fit_gram(gram, labels, recipe["c"])
    scores, support_scores = head.decision_function(cross), head.decision_function(gram)
    if not np.isfinite(scores).all() or not np.isfinite(support_scores).all():
        raise ResearchError("Skor SVM tidak finite.")
    return head, scores, support_scores, mean, scale, median, gamma


def choose_recipe(dev_ids, correct_by_id, held, policy):
    ids, values = np.asarray(dev_ids), np.asarray(correct_by_id)
    if (
        policy not in POLICIES
        or held not in range(17)
        or ids.shape != (16,)
        or set(ids.tolist()) != set(range(17)) - {held}
        or values.shape != (16, len(RECIPES))
        or not np.issubdtype(values.dtype, np.integer)
        or (values < 0).any()
    ):
        raise ResearchError("Seleksi SVM harus menerima tepat 16 ID selain held ID.")
    totals = values.sum(0, dtype=np.int64)
    candidates = [i for i, r in enumerate(RECIPES) if r["kernel"] == policy]
    chosen = max(
        candidates,
        key=lambda i: (
            int(totals[i]),
            -RECIPES[i]["c"],
            -(RECIPES[i]["multiplier"] or 0),
            -NORMALIZERS.index(RECIPES[i]["normalizer"]),
        ),
    )
    return {
        "held": held,
        "policy": policy,
        "source_ids": sorted(ids.tolist()),
        "recipe_index": chosen,
        "recipe": RECIPES[chosen],
        "source_correct_by_recipe": totals.tolist(),
        "selected_source_correct": int(totals[chosen]),
    }


def select_from_table(table, held, policy):
    table = np.asarray(table)
    if table.shape != (17, len(RECIPES)):
        raise ResearchError("Tabel sumber tidak lengkap.")
    dev = [i for i in range(17) if i != held]
    return choose_recipe(dev, table[dev], held, policy)


def replay(cross, support_positions, dual_coef, intercept, n_support):
    """Reconstruct all 66 pair margins and 12 vote/confidence scores from arrays."""
    starts = np.r_[0, np.cumsum(n_support)]
    kernel = cross[:, support_positions]
    votes, confidence = np.zeros((len(cross), 12)), np.zeros((len(cross), 12))
    pair = 0
    for i in range(12):
        a = slice(starts[i], starts[i + 1])
        for j in range(i + 1, 12):
            b = slice(starts[j], starts[j + 1])
            value = kernel[:, a] @ dual_coef[j - 1, a]
            value += kernel[:, b] @ dual_coef[i, b] + intercept[pair]
            votes[:, i] += value >= 0
            votes[:, j] += value < 0
            confidence[:, i] += value
            confidence[:, j] -= value
            pair += 1
    return votes + confidence / (3 * (np.abs(confidence) + 1))
