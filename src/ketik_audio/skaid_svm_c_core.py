"""Extended C grid with support-only fitting and source-ID-only selection."""

from __future__ import annotations

import numpy as np

from .common import ResearchError
from .skaid_selection_core import NORMALIZERS, transform
from .skaid_svm_core import MULTIPLIERS, POLICIES, bandwidth, kernel_matrices
from .skaid_svm_core import RECIPES as OLD_RECIPES
from .skaid_svm_core import svc as old_svc

NEW_CS = (1000.0, 10000.0, 100000.0)
MAX_ITER = 2000000
NEW_RECIPES = tuple(
    {"kernel": "linear", "normalizer": mode, "c": c, "multiplier": None}
    for mode in NORMALIZERS
    for c in NEW_CS
) + tuple(
    {"kernel": "rbf", "normalizer": mode, "c": c, "multiplier": g}
    for mode in NORMALIZERS
    for g in MULTIPLIERS
    for c in NEW_CS
)
RECIPES = OLD_RECIPES + NEW_RECIPES
ANCHORS = (4, 19)


def svc(c, kernel="precomputed", gamma="scale"):
    return old_svc(c, kernel, gamma).set_params(max_iter=MAX_ITER)


def checked_head(head):
    if (
        head.fit_status_ != 0
        or not np.array_equal(head.classes_, np.arange(12))
        or not np.isfinite(head.dual_coef_).all()
        or not np.isfinite(head.intercept_).all()
    ):
        raise ResearchError(
            f"Fit SVM C={head.C} tidak valid: status {head.fit_status_}, "
            f"maksimum {head.n_iter_.max()} iterasi."
        )
    return head


def fit_gram(gram, labels, c):
    labels = np.asarray(labels)
    if set(labels.tolist()) != set(range(12)) or not np.isfinite(gram).all():
        raise ResearchError("Dukungan/Gram tidak valid atau kelas tidak lengkap.")
    return checked_head(svc(c).fit(gram, labels))


def fit_selected(support, labels, query, recipe):
    x, q, mean, scale = transform(support, query, recipe["normalizer"])
    median, distances = bandwidth(x)
    gram, cross, gamma = kernel_matrices(x, q, recipe, median, distances)
    head = fit_gram(gram, labels, recipe["c"])
    scores, support_scores = head.decision_function(cross), head.decision_function(gram)
    if not np.isfinite(scores).all() or not np.isfinite(support_scores).all():
        raise ResearchError("Skor SVM C tidak finite.")
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
        or np.any(values < 0)
    ):
        raise ResearchError("Seleksi C harus menerima tepat 16 ID selain held ID.")
    totals = values.sum(0, dtype=np.int64)
    candidates = [i for i, r in enumerate(RECIPES) if r["kernel"] == policy]
    winner = max(
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
        "recipe_index": winner,
        "recipe": RECIPES[winner],
        "source_correct_by_recipe": totals.tolist(),
        "selected_source_correct": int(totals[winner]),
    }


def select_from_table(table, held, policy):
    table = np.asarray(table)
    if table.shape != (17, len(RECIPES)):
        raise ResearchError("Tabel grid C tidak lengkap.")
    dev = [i for i in range(17) if i != held]
    return choose_recipe(dev, table[dev], held, policy)
