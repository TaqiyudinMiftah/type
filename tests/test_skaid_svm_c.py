import numpy as np
import pytest

from ketik_audio import skaid_svm_c
from ketik_audio.common import ResearchError
from ketik_audio.skaid_svm_c_core import (
    ANCHORS,
    MAX_ITER,
    NEW_RECIPES,
    OLD_RECIPES,
    RECIPES,
    checked_head,
    choose_recipe,
    fit_selected,
    select_from_table,
    svc,
)
from ketik_audio.skaid_svm_core import fit_selected as previous_fit
from ketik_audio.skaid_svm_core import select_from_table as previous_select


def test_extended_grid_only_adds_c_and_reuses_old_tie_rule():
    stripped = [{k: v for k, v in r.items() if k != "c"} for r in OLD_RECIPES]
    assert len(RECIPES) == 96 and len(NEW_RECIPES) == 36 and RECIPES[:60] == OLD_RECIPES
    assert len({tuple(r.items()) for r in RECIPES}) == 96
    assert {r["c"] for r in NEW_RECIPES} == {1e3, 1e4, 1e5}
    assert all({k: v for k, v in r.items() if k != "c"} in stripped for r in NEW_RECIPES)
    rng = np.random.default_rng(20261114)
    table = np.zeros((17, 96), dtype=np.int64)
    table[:, :60] = rng.integers(1, 8, size=(17, 60))
    for policy in ("linear", "rbf"):
        for h in range(17):
            a = previous_select(table[:, :60], h, policy)
            b = select_from_table(table, h, policy)
            assert a["recipe_index"] == b["recipe_index"]
            assert a["source_correct_by_recipe"] == b["source_correct_by_recipe"][:60]


def test_held_row_cannot_choose_new_c_and_controls_can_choose_separately():
    main = np.zeros((17, 96), dtype=np.int64)
    main[:, 69] = 19
    neg = main.copy()
    neg[:, 70] = 20
    main[11, 71] = 1_000_000
    a, b = (select_from_table(t, 11, "rbf") for t in (main, neg))
    assert a["recipe_index"] == 69 and b["recipe_index"] == 70
    assert a["source_ids"] == [h for h in range(17) if h != 11]
    with pytest.raises(ResearchError):
        choose_recipe(list(range(16)), main[:16], 11, "rbf")


@pytest.mark.parametrize("ri", ANCHORS)
def test_higher_iteration_cap_replays_converged_anchor(ri):
    rng = np.random.default_rng(2024)
    x, q = rng.normal(size=(36, 7)), rng.normal(size=(12, 7))
    y = np.tile(np.arange(12), 3)
    prev = previous_fit(x, y, q, RECIPES[ri])
    new = fit_selected(x, y, q, RECIPES[ri])
    for a, b in zip(prev[1:], new[1:], strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(prev[0].dual_coef_, new[0].dual_coef_)
    assert new[0].max_iter == MAX_ITER


def test_iteration_failure_is_not_accepted():
    rng = np.random.default_rng(19)
    x, y = rng.normal(size=(72, 6)), np.tile(np.arange(12), 6)
    with pytest.warns(UserWarning):
        head = svc(1e5).set_params(max_iter=1).fit(x @ x.T, y)
    with pytest.raises(ResearchError, match="status 1"):
        checked_head(head)


def test_aggregate_appends_candidates_without_overwriting_baseline(tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"
    for p in (old, new):
        (p / "validation").mkdir(parents=True)
    monkeypatch.setattr(skaid_svm_c, "OLD", old)
    monkeypatch.setattr(skaid_svm_c, "ENCODERS", ("mfcc",))
    conditions = [
        {"subject": h, "k": 10, "permuted": p}
        for h in range(17)
        for p in (False, True)
        for _ in range(3)
    ]
    a, b = np.zeros((len(conditions), 60), dtype=int), np.zeros((len(conditions), 36), dtype=int)
    a[:, 4] = 17
    for ci, p in enumerate(conditions):
        b[ci, int(p["permuted"])] = 19
    np.savez(old / "validation/mfcc.npz", query_correct=a)
    np.savez(new / "validation/mfcc.npz", query_correct=b)
    tables = skaid_svm_c.aggregate_tables(new, conditions)
    for p in (False, True):
        table = tables["mfcc", 10, p]
        assert np.all(table[:, 4] == 51)
        assert select_from_table(table, 7, "linear")["recipe_index"] == 60 + int(p)
