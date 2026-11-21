import numpy as np
import pytest

from ketik_audio.common import ResearchError
from ketik_audio.skaid_pilot_common import KEYS
from ketik_audio.skaid_pilot_probe import feature_rows, fit_head, permuted_labels, validate_plan


def test_ridge_matches_centered_dual_and_query_data_cannot_change_fit():
    rng = np.random.default_rng(4237)
    raw = rng.normal(size=(96, 31))
    labels = np.tile(np.arange(12), 8)
    support, query = np.arange(36), np.arange(36, 96)
    x = feature_rows(raw)
    head, scores, _, _ = fit_head(x, labels, support, query)
    train = x[support]
    target = 2 * np.eye(12)[labels[support]] - 1
    xc, yc = train - train.mean(0), target - target.mean(0)
    dual = np.linalg.solve(xc @ xc.T + np.eye(len(xc)), yc)
    coefficient = dual.T @ xc
    intercept = target.mean(0) - coefficient @ train.mean(0)
    assert np.allclose(coefficient, head.coef_, atol=1e-10, rtol=0)
    assert np.allclose(intercept, head.intercept_, atol=1e-10, rtol=0)
    assert np.allclose(x[query] @ coefficient.T + intercept, scores, atol=1e-10, rtol=0)
    changed = raw.copy()
    changed[query] = rng.normal(20, 40, changed[query].shape)
    changed_labels = labels.copy()
    changed_labels[query] = rng.permutation(changed_labels[query])
    other, _, _, _ = fit_head(feature_rows(changed), changed_labels, support, query)
    assert np.array_equal(head.coef_, other.coef_)
    assert np.array_equal(head.intercept_, other.intercept_)


def test_negative_control_preserves_budget_and_rejects_non_k10():
    labels = np.tile(np.arange(12), 15)
    plan = {"subject": 2, "draw": 101, "k": 10, "support": list(range(120))}
    shuffled = permuted_labels(labels, plan)
    assert np.array_equal(np.bincount(shuffled, minlength=12), np.full(12, 10))
    assert not np.array_equal(shuffled, labels[:120])
    assert np.array_equal(shuffled, permuted_labels(labels, plan))
    with pytest.raises(ResearchError):
        permuted_labels(labels, {**plan, "k": 5})


def test_probe_plan_rejects_label_and_recording_mismatches():
    labels = np.tile(np.arange(12), 6)
    records = [
        {
            "id": f"clip{i}",
            "clip_index": i,
            "label": int(y),
            "key": KEYS[y],
            "phase": "Phase 1" if i < 12 else "Phase 2",
            "session_token": "example",
            "audio": "calibration.m4a" if i < 12 else "query.m4a",
        }
        for i, y in enumerate(labels)
    ]
    plan = {
        "k": 1,
        "support": list(range(12)),
        "query": list(range(12, 72)),
        "session_token": "example",
    }
    validate_plan(records, [plan], labels)
    wrong = labels.copy()
    wrong[0] = 1
    with pytest.raises(ResearchError):
        validate_plan(records, [plan], wrong)
    shared = [{**r, "audio": "same.m4a"} for r in records]
    with pytest.raises(ResearchError):
        validate_plan(shared, [plan], labels)
