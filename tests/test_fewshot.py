"""Check the scientific boundaries that a plausible score alone cannot verify."""

import numpy as np
import pytest

from ketik_audio.common import ResearchError
from ketik_audio.fewshot import KS, RATE, SAMPLES, calibration_split, recrop


@pytest.mark.parametrize("seed", [101, 102, 103])
def test_support_is_nested_balanced_and_never_uses_fixed_test(seed):
    y = np.repeat(np.arange(36), 25)
    order = np.tile(np.arange(25), 36)
    # Input ordering must not change the physical event chosen.
    shuffle = np.random.default_rng(99).permutation(len(y))
    previous = set()
    for k in KS:
        split = calibration_split(y, order, k, seed)
        support, test = split["support"], split["test"]
        assert np.array_equal(np.bincount(y[support]), np.full(36, k))
        assert np.array_equal(np.bincount(y[test]), np.full(36, 10))
        assert set(support).isdisjoint(test)
        assert previous.issubset(support)
        assert np.all(order[support] < 15)
        assert np.array_equal(test, np.flatnonzero(order >= 15))
        reordered = calibration_split(y[shuffle], order[shuffle], k, seed)
        assert set(shuffle[reordered["support"]]) == set(support)
        previous = set(support)


def test_duplicate_or_missing_events_cannot_silently_change_label_budget():
    y = np.repeat(np.arange(36), 25)
    order = np.tile(np.arange(25), 36)
    order[1] = 0
    with pytest.raises(ResearchError):
        calibration_split(y, order, 1, 101)


@pytest.mark.parametrize("original_rate", [32_000, 44_100])
def test_recrop_preserves_common_physical_duration_and_event_alignment(original_rate):
    wave = np.zeros(original_rate * 2, dtype=np.float32)
    centers = [round(0.5 * original_rate), round(1.1 * original_rate)]
    wave[centers] = 1
    clips, audit = recrop(wave, original_rate, centers)
    assert clips.shape == (2, SAMPLES)
    assert SAMPLES / RATE == 0.4
    assert np.all(np.argmax(np.abs(clips), axis=1) == round(0.05 * RATE))
    assert np.all(np.asarray(audit["ends_16k"]) <= 2 * RATE)


def test_recrop_rejects_truncation_and_overlap_instead_of_fabricating_padding():
    wave = np.zeros(RATE * 2, dtype=np.float32)
    with pytest.raises(ResearchError, match="melewati"):
        recrop(wave, RATE, [RATE * 2 - 100])
    with pytest.raises(ResearchError, match="tumpang"):
        recrop(wave, RATE, [RATE, RATE + 100])
