import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifier

from ketik_audio.common import ResearchError
from ketik_audio.skaid_selection_core import (
    ALPHAS,
    NORMALIZERS,
    RECIPES,
    choose_recipe,
    fit_selected,
    ridge_path,
    select_from_table,
    source_budget,
    transform,
)


@pytest.mark.parametrize("count", [12, 120])
def test_path_matches_independent_sklearn_for_all_alphas_and_normalizers(count):
    rng = np.random.default_rng(1271 + count)
    x, q = rng.normal(size=(count, 31)), rng.normal(size=(60, 31))
    x[:, -1], q[:, -1] = 3.0, 8.0  # Constant support column with a shifted query value.
    y = np.tile(np.arange(12), count // 12)
    for mode in NORMALIZERS:
        tx, tq, _, _ = transform(x, q, mode)
        queries, supports = ridge_path(tx, y, tq)
        for j, alpha in enumerate(ALPHAS):
            head = RidgeClassifier(alpha=alpha, solver="cholesky").fit(tx, y)
            assert np.allclose(queries[j], head.decision_function(tq), atol=1e-6, rtol=1e-5)
            assert np.allclose(supports[j], head.decision_function(tx), atol=1e-6, rtol=1e-5)


def test_preprocessing_and_coefficients_cannot_use_query_statistics():
    rng = np.random.default_rng(199)
    x, q = rng.normal(size=(24, 15)), rng.normal(size=(60, 15))
    y = np.tile(np.arange(12), 2)
    recipe = {"normalizer": "support_z_l2", "alpha": 0.001}
    head, _, _, mean, scale = fit_selected(x, y, q, recipe)
    other, _, _, second_mean, second_scale = fit_selected(x, y, q * 3000 + 120, recipe)
    assert np.array_equal(mean, second_mean) and np.array_equal(scale, second_scale)
    assert np.array_equal(head.coef_, other.coef_)
    assert np.array_equal(head.intercept_, other.intercept_)


def test_source_only_selection_and_stronger_alpha_tie_rule():
    values = np.zeros((17, len(RECIPES)), dtype=np.int64)
    chosen = select_from_table(values, held=4)
    assert chosen["recipe"] == {"normalizer": "row_l2", "alpha": 100.0}
    values[:, 1] = 5
    before = select_from_table(values, held=4)
    values[4] = 10**8
    values[4, 15] = 10**10
    assert select_from_table(values, held=4) == before
    with pytest.raises(ResearchError):
        choose_recipe(list(range(16)), np.zeros((16, len(RECIPES)), dtype=int), held=4)
    with pytest.raises(ResearchError):
        choose_recipe([0] * 16, np.zeros((16, len(RECIPES)), dtype=int), held=16)


def test_label_cost_counts_union_instead_of_repeated_draws():
    plans = []
    for subject in range(17):
        support = list(range(subject * 1000, subject * 1000 + 12))
        query = list(range(subject * 1000 + 100, subject * 1000 + 160))
        for draw in (101, 102, 103):
            plans.append(
                {"subject": subject, "k": 1, "draw": draw, "support": support, "query": query}
            )
    budget = source_budget(plans, held=3, k=1)
    assert budget["source_support_labels_unique"] == 16 * 12
    assert budget["source_query_labels_unique"] == 960
    assert budget["source_labels_unique_total"] == 1152
    assert 3 not in budget["source_ids"]


def test_constant_features_keep_finite_ridge_path():
    x = np.ones((12, 18))
    for mode in NORMALIZERS:
        tx, tq, _, scale = transform(x, x * 20, mode)
        assert np.all(scale > 0)
        q, s = ridge_path(tx, np.arange(12), tq)
        assert np.isfinite(q).all() and np.isfinite(s).all()


def test_control_pipeline_uses_its_own_source_table():
    from ketik_audio.skaid_selection import choices_from_tables
    from ketik_audio.skaid_selection_core import FAMILIES

    plans = [
        {
            "subject": s,
            "k": k,
            "draw": d,
            "support": list(range(s * 10000, s * 10000 + 12 * k)),
            "query": list(range(s * 10000 + 1000, s * 10000 + 1060)),
        }
        for s in range(17)
        for k in (1, 2, 3, 5, 10)
        for d in (101, 102, 103)
    ]
    tables = {}
    for family in FAMILIES:
        for k, permuted in [*((k, False) for k in (1, 2, 3, 5, 10)), (10, True)]:
            table = np.zeros((17, len(RECIPES)), dtype=np.int64)
            table[:, 12 if permuted else 1] = 100
            tables[family, k, permuted] = table
    choices, budgets = choices_from_tables(tables, plans)
    assert len(choices) == 408 and len(budgets) == 85
    assert all(c["recipe_index"] == (12 if c["permuted"] else 1) for c in choices)
    assert all(c["held"] not in c["source_ids"] for c in choices)
