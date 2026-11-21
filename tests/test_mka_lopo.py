import numpy as np
import pytest

from ketik_audio.common import ResearchError
from ketik_audio.mka import KEYS, PLATFORMS
from ketik_audio.mka_lopo import (
    DRAWS,
    fit_head,
    parent_scores,
    permuted_parent_labels,
    split_plan,
)


def records():
    return [
        {"platform": platform, "key": key, "label": label, "parent_group_id": f"{platform}:{key}"}
        for platform in PLATFORMS
        for label, key in enumerate(KEYS)
        for _ in range(6)
    ]


def test_all_folds_have_disjoint_parent_wavs_and_nested_equal_label_budgets():
    rows = records()
    plan = split_plan(rows)
    assert len(plan) == 78
    for split in plan:
        support, test = split["support"], split["test"]
        assert len(support) == 180 * split["k"]
        assert len(test) == 216
        assert not set(split["support_parents"]) & set(split["test_parents"])
        assert len(split["support_parents"]) == 180
        assert all(rows[i]["platform"] != split["heldout"] for i in support)
        assert all(rows[i]["platform"] == split["heldout"] for i in test)
        assert np.array_equal(
            np.bincount([rows[i]["label"] for i in support], minlength=36),
            np.full(36, 5 * split["k"]),
        )
    for platform in PLATFORMS:
        for draw in DRAWS:
            selected = [s for s in plan if s["heldout"] == platform and s["draw"] == draw]
            selected.sort(key=lambda s: s["k"])
            for a, b in zip(selected[:-1], selected[1:], strict=True):
                assert set(a["support"]) <= set(b["support"])
                assert a["test"] == b["test"]
        assert sum(s["heldout"] == platform and s["k"] == 6 for s in plan) == 1


def test_shared_pcm_parent_across_folders_is_rejected():
    rows = records()
    for row in rows[:6]:
        row["parent_group_id"] = rows[216]["parent_group_id"]
    with pytest.raises(ResearchError):
        split_plan(rows)


def test_negative_control_shuffles_whole_wavs_and_preserves_class_balance():
    rows = records()
    support = next(s for s in split_plan(rows) if s["k"] == 6)["support"]
    labels = permuted_parent_labels(rows, support)
    assert np.array_equal(np.bincount(labels, minlength=36), np.full(36, 30))
    assert np.array_equal(labels, permuted_parent_labels(rows, support))
    assert any(labels[i] != rows[index]["label"] for i, index in enumerate(support))
    parents = {}
    for value, index in zip(labels, support, strict=True):
        parents.setdefault(rows[index]["parent_group_id"], set()).add(int(value))
    assert all(len(values) == 1 for values in parents.values())


def test_fixed_probe_handles_36_classes_and_never_fits_heldout_rows_or_labels():
    features = np.tile(np.eye(36), (3, 1))
    labels = np.tile(np.arange(36), 3)
    support, test = np.arange(72), np.arange(72, 108)
    head, scores, fitted = fit_head(features, labels, support, test)
    assert np.array_equal(scores.argmax(1), labels[test])
    assert np.array_equal(fitted.argmax(1), labels[support])
    changed_features, changed_labels = features.copy(), labels.copy()
    changed_features[test] = np.random.default_rng(8).normal(size=(36, 36)) * 100
    changed_labels[test] = 0
    other, _, _ = fit_head(changed_features, changed_labels, support, test)
    assert np.array_equal(head.coef_, other.coef_)
    assert np.array_equal(head.intercept_, other.intercept_)
    assert head.alpha == 1


def test_parent_aggregation_preserves_labels_and_first_seen_group_order():
    rows = records()
    indices = np.array([8, 0, 1, 6, 7, 2])
    values = np.array([[1, 0], [0, 1], [0, 3], [3, 0], [5, 0], [0, 5]])
    labels, grouped, parents = parent_scores(rows, indices, values)
    assert labels.tolist() == [1, 0]
    assert parents == ["hp:2", "hp:1"]
    assert np.array_equal(grouped, [[3, 0], [0, 3]])
