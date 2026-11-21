import numpy as np
import pytest

from ketik_audio.common import ResearchError
from ketik_audio.mka import KEYS, PLATFORMS
from ketik_audio.mka_nested import ALPHAS, choose_alpha, inner_select, nested_plan, ridge


def records():
    rows = [
        {"platform": platform, "key": key, "label": label, "parent_group_id": f"{platform}:{key}"}
        for platform in PLATFORMS
        for label, key in enumerate(KEYS)
        for _ in range(6)
    ]
    return [{**row, "pcm_sha256": f"unique-{index}"} for index, row in enumerate(rows)]


def test_nested_roles_are_parent_disjoint_and_validate_each_source_row_once():
    rows = records()
    plan = nested_plan(rows)
    assert len(plan) == 6
    for outer in plan:
        assert len(outer["support"]) == 1080
        assert len(outer["test"]) == 216
        seen = []
        for fold in outer["inner"]:
            assert len(fold["train"]) == 864
            assert len(fold["validation"]) == 216
            sets = [
                {rows[i]["parent_group_id"] for i in role}
                for role in (fold["train"], fold["validation"], outer["test"])
            ]
            assert [len(values) for values in sets] == [144, 36, 36]
            assert not any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2)))
            assert np.array_equal(
                np.bincount([rows[i]["label"] for i in fold["train"]]), np.full(36, 24)
            )
            seen.extend(fold["validation"])
        assert sorted(seen) == outer["support"]


def test_pcm_copy_between_inner_train_and_validation_is_rejected():
    rows = records()
    rows[216]["pcm_sha256"] = rows[432]["pcm_sha256"]
    with pytest.raises(ResearchError, match="Kebocoran"):
        nested_plan(rows)


def test_exact_integer_tie_prefers_largest_alpha_without_accuracy_tolerance():
    counts = np.full(len(ALPHAS), 30, dtype=np.int64)
    assert ALPHAS[choose_alpha(counts)] == 100
    counts[0] += 1
    assert ALPHAS[choose_alpha(counts)] == 1e-6
    with pytest.raises(ResearchError):
        choose_alpha(counts.astype(float))
    with pytest.raises(ResearchError):
        choose_alpha(counts[:-1])


def test_selection_sees_only_outer_training_rows_and_labels():
    rows = records()
    outer = nested_plan(rows)[0]
    support, test = np.array(outer["support"]), np.array(outer["test"])
    y = np.array([row["label"] for row in rows])
    features = np.eye(36)[y]
    result = inner_select(features[support], y[support], outer["inner"])
    assert result["correct"].sum(1).tolist() == [1080] * len(ALPHAS)
    assert ALPHAS[int(result["selected_index"])] == 100
    features[test] = np.random.default_rng(8).normal(size=(216, 36)) * 100
    y[test] = 0
    other = inner_select(features[support], y[support], outer["inner"])
    for key in ("coef", "intercept", "validation_scores", "correct", "selected_index"):
        assert np.array_equal(result[key], other[key])


def test_repeated_inner_validation_folder_is_rejected_before_fit():
    rows = records()
    outer = nested_plan(rows)[0]
    y = np.array([rows[i]["label"] for i in outer["support"]])
    with pytest.raises(ResearchError, match="tepat sekali"):
        inner_select(np.eye(36)[y], y, [outer["inner"][0]] * 5)


def test_alpha_changes_shrinkage_and_invalid_alpha_or_missing_class_is_rejected():
    y = np.tile(np.arange(36), 2)
    x = np.eye(36)[y]
    weak = ridge(x, y, 1e-6)
    strong = ridge(x, y, 100)
    assert np.linalg.norm(weak.coef_) > np.linalg.norm(strong.coef_)
    assert np.array_equal(weak.predict(x), y)
    assert np.array_equal(strong.predict(x), y)
    with pytest.raises(ResearchError):
        ridge(x, y, 0)
    with pytest.raises(ResearchError):
        ridge(x[:-2], np.zeros(len(x) - 2, dtype=int), 1)
