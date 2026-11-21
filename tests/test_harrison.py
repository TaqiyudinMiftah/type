from __future__ import annotations

import numpy as np
import pytest

from ketik_audio.harrison import KEYS, isolate, measure, split_indices


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("temporal", [False, True])
def test_external_split_partitions_original_events_before_augmentation(seed, temporal):
    y = np.repeat(np.arange(len(KEYS)), 25)
    order = np.tile(np.arange(25), len(KEYS))
    parts = split_indices(y, order, seed, temporal)
    expected = {"train": 18, "validation": 2, "test": 5}
    for name, per_key in expected.items():
        assert np.array_equal(np.bincount(y[parts[name]]), np.full(36, per_key))
    assert len(set(np.concatenate(list(parts.values())))) == 900
    if temporal:
        assert np.all(order[parts["train"]] < 18)
        assert np.all(order[parts["test"]] >= 20)


def test_count_assisted_isolation_recovers_known_fixture_events():
    # This verifies timing/window geometry, not recognition accuracy on real audio.
    rate = 44_100
    wave = np.zeros(rate * 17, dtype=np.float32)
    events = np.arange(25) * round(0.6 * rate) + rate
    wave[events] = 1
    clips, audit = isolate(wave, rate)
    assert clips.shape == (25, 14_400)
    assert audit["strategy"] == "author-adaptive"
    assert np.max(np.abs(np.asarray(audit["centers_samples"]) - events)) <= 24
    assert np.all(np.max(clips, axis=1) == 1)


def test_external_metrics_include_full_class_universe_and_binomial_uncertainty():
    y = np.repeat(np.arange(36), 5)
    scores = np.eye(36)[y]
    result = measure(y, scores)
    assert result["n"] == result["correct"] == 180
    assert result["accuracy"] == result["macro_f1"] == result["top5"] == 1
    assert 0.97 < result["wilson95"][0] < result["wilson95"][1] <= 1
    assert np.array_equal(np.asarray(result["confusion_matrix"]), np.eye(36) * 5)
