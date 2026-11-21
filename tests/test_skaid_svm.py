import numpy as np
import pytest
from sklearn.preprocessing import normalize

from ketik_audio.common import ResearchError
from ketik_audio.skaid_selection_core import transform
from ketik_audio.skaid_svm_core import (
    RECIPES,
    bandwidth,
    choose_recipe,
    fit_selected,
    kernel_matrices,
    replay,
    select_from_table,
    svc,
)


@pytest.mark.parametrize("kernel", ["linear", "rbf"])
@pytest.mark.parametrize("normalizer", ["row_l2", "support_z_l2"])
def test_serialized_dual_replay_and_direct_kernel(kernel, normalizer):
    rng = np.random.default_rng(20261111)
    x, q, y = rng.normal(size=(36, 9)), rng.normal(size=(24, 9)), np.tile(np.arange(12), 3)
    x[:, 0], q[:, 0] = 7, 7
    recipe = {"kernel": kernel, "normalizer": normalizer, "c": 10, "multiplier": 1}
    head, scores, support, _, _, median, gamma = fit_selected(x, y, q, recipe)
    xx, qq, _, _ = transform(x, q, normalizer)
    gram, cross, _ = kernel_matrices(xx, qq, recipe)
    direct = svc(recipe["c"], kernel, gamma if kernel == "rbf" else "scale").fit(xx, y)
    np.testing.assert_allclose(direct.decision_function(qq), scores, atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(direct.predict(qq), scores.argmax(1))
    actual = replay(cross, head.support_, head.dual_coef_, head.intercept_, head.n_support_)
    np.testing.assert_allclose(actual, scores, atol=1e-12, rtol=1e-12)
    actual = replay(gram, head.support_, head.dual_coef_, head.intercept_, head.n_support_)
    np.testing.assert_allclose(actual, support, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.diag(gram), 1, atol=1e-12)
    assert median >= 0


def test_query_cannot_change_bandwidth_or_fit():
    rng = np.random.default_rng(42)
    x, q, y = rng.normal(size=(24, 5)), rng.normal(size=(13, 5)), np.tile(np.arange(12), 2)
    recipe = {"kernel": "rbf", "normalizer": "support_z_l2", "c": 1, "multiplier": 0.1}
    a = fit_selected(x, y, q, recipe)
    b = fit_selected(x, y, q * 1000 + 99, recipe)
    for index in (3, 4, 5, 6):
        np.testing.assert_array_equal(a[index], b[index])
    np.testing.assert_array_equal(a[0].dual_coef_, b[0].dual_coef_)
    np.testing.assert_array_equal(a[0].intercept_, b[0].intercept_)
    median, dist = bandwidth(normalize(np.ones((12, 7))))
    assert median == 0
    gram, _, gamma = kernel_matrices(
        normalize(np.ones((12, 7))), np.zeros((1, 7)), recipe, median, dist
    )
    assert np.isfinite(gram).all() and gamma == recipe["multiplier"]


def test_source_selection_excludes_held_and_respects_policy_and_ties():
    table = np.zeros((17, len(RECIPES)), dtype=np.int64)
    for policy in ("linear", "rbf"):
        a = select_from_table(table, 3, policy)
        assert a["recipe"]["c"] == 0.01
        assert a["recipe"]["normalizer"] == "row_l2"
        assert a["recipe"]["kernel"] == policy
        if policy == "rbf":
            assert a["recipe"]["multiplier"] == 0.01
        changed = table.copy()
        changed[3] = np.arange(len(RECIPES)) * 999999
        assert select_from_table(changed, 3, policy) == a
    with pytest.raises(ResearchError):
        choose_recipe(list(range(16)), table[:16], 3, "rbf")
    with pytest.raises(ResearchError):
        choose_recipe([i for i in range(17) if i != 3], table[:16].astype(float), 3, "rbf")


def test_support_label_permutation_is_not_perfect_fit_on_true_labels():
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(24, 15)), np.tile(np.arange(12), 2)
    permuted = rng.permutation(y)
    recipe = {"kernel": "rbf", "normalizer": "row_l2", "c": 100, "multiplier": 100}
    _, _, scores, *_ = fit_selected(x, permuted, x[:3], recipe)
    np.testing.assert_array_equal(scores.argmax(1), permuted)
    assert np.mean(scores.argmax(1) == y) < 0.3


def test_control_source_table_is_separate(tmp_path, monkeypatch):
    from ketik_audio import skaid_svm

    monkeypatch.setattr(skaid_svm, "ENCODERS", ("mfcc",))
    (tmp_path / "validation").mkdir()
    conditions, rows = [], []
    for held in range(17):
        for permuted in (False, True):
            for _ in range(3):
                conditions.append({"subject": held, "k": 10, "permuted": permuted})
                counts = np.zeros(len(RECIPES), dtype=np.int64)
                counts[11 if permuted else 12] = 60
                rows.append(counts)
    np.savez(tmp_path / "validation/mfcc.npz", query_correct=rows)
    tables = skaid_svm.aggregate_tables(tmp_path, conditions)
    a = select_from_table(tables["mfcc", 10, False], 0, "rbf")
    b = select_from_table(tables["mfcc", 10, True], 0, "rbf")
    assert a["recipe_index"] == 12 and b["recipe_index"] == 11
