import io
import shutil
import warnings
from zipfile import ZipFile

import numpy as np
import pytest

from ketik_audio.common import ResearchError
from ketik_audio.public_baseline import (
    DRAWS,
    check_roles,
    mulaw8k,
    permute,
    prefix,
    ranks,
    scores_metrics,
)
from ketik_audio.public_recordings import (
    archive_members,
    hear_crop,
    hear_pairs,
    isolated_window,
    parent_error,
)
from ketik_audio.skaid_pilot_common import KEYS


@pytest.mark.parametrize("names", [("../bad.wav",), ("one.wav", "one.wav")])
def test_zip_rejects_traversal_or_duplicate_members(names):
    data = io.BytesIO()
    with ZipFile(data, "w") as z, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Duplicate name:", category=UserWarning)
        for name in names:
            z.writestr(name, b"unused")
    with ZipFile(data) as z, pytest.raises(ResearchError):
        archive_members(z)


def test_hear_geometry_rejects_neighbors_and_excessive_parent_shift():
    times = np.array([120.0, 200.0, 265.0])
    assert isolated_window(200, times, 1, 500)
    assert not isolated_window(200.1, times, 1, 500)
    assert not isolated_window(200, times, 1, 250)
    parent = np.random.default_rng(51).normal(0, 0.01, (48000, 1)).astype(np.float32)
    clip, center = hear_crop(parent[6720:19200], 48000, 200, 1000, 60, 200)
    assert len(clip) == 3840 and center == 2880
    assert parent_error(clip, parent, 48000, 200) == (0, 0)
    shifted_one = np.pad(parent, ((1, 0), (0, 0)))
    assert parent_error(clip, shifted_one, 48000, 200) == (0, 1)
    shifted_three = np.pad(parent, ((3, 0), (0, 0)))
    assert parent_error(clip, shifted_three, 48000, 200)[0] > 0.1
    assert hear_crop(parent[6720:19190], 48000, 200, 1000, 60, 200) == (None, None)


def test_hear_pairs_exclude_overlapping_copied_or_different_metadata_device():
    root = {
        "participant": "p00",
        "device": "d00",
        "session": 1,
        "track_unique": True,
        "parent_pcm_sha256": "a",
        "start_relative_seconds": 0,
        "native_seconds": 100,
        "index": 0,
    }
    later = {
        **root,
        "session": 2,
        "start_relative_seconds": 120,
        "parent_pcm_sha256": "b",
        "index": 1,
    }
    packs = [root, later]
    packs += [
        {**later, **override, "index": i}
        for i, override in enumerate(
            (
                {"start_relative_seconds": 40},
                {"parent_pcm_sha256": "a"},
                {"device": "d01"},
                {"track_unique": False},
            ),
            2,
        )
    ]
    pairs, rows, _ = hear_pairs(packs, [])
    assert [(p["calibration"], p["query"]) for p in pairs] == [(0, 1)]
    assert all(not r["pairs"] for r in rows)


def test_locked_nested_support_ignores_query_order_and_rejects_pcm_overlap():
    train = [
        {"id": f"s:{c}{i}", "letter": c, "group": "a", "crop_pcm_sha256": f"s:{c}{i}"}
        for c in KEYS
        for i in range(12)
    ]
    query = [
        {"id": f"q:{c}{i}", "letter": c, "group": "b", "crop_pcm_sha256": f"q:{c}{i}"}
        for c in KEYS
        for i in range(5)
    ]
    rankings = {str(d): ranks(train, KEYS, 10, 0, d, 1) for d in DRAWS}
    for d in DRAWS:
        assert rankings[str(d)] == ranks(list(reversed(train)), KEYS, 10, 0, d, 1)
        assert set(prefix(rankings[str(d)], KEYS, 2)) < set(prefix(rankings[str(d)], KEYS, 5))
    ids = {i for rank in rankings.values() for i in prefix(rank, KEYS, 10)}
    train = [r for r in train if r["id"] in ids]
    plan = [{"dataset": "hear", "calibration": "a", "query_group": "b"}]
    payload = plan, {"a": rankings}, {"b": [r["id"] for r in query]}, {"hear": KEYS}
    check_roles(train + query, *payload)
    query[0]["crop_pcm_sha256"] = train[0]["crop_pcm_sha256"]
    with pytest.raises(ResearchError):
        check_roles(train + query, *payload)
    labels = np.repeat(np.arange(12), 10)
    y = permute(labels, 0, DRAWS[0])
    assert not np.array_equal(labels, y)
    np.testing.assert_array_equal(np.bincount(labels), np.bincount(y))


def test_missing_test_class_remains_output_and_macro_excludes_absent_class():
    labels = np.array([0, 0, 1])
    scores = np.array([[4, 1, 0], [0, 1, 2], [0, 1, 3]])
    result = scores_metrics(labels, scores, ("a", "b", "c"))
    assert result["classes_present"] == ["a", "b"]
    assert result["confusion"] == [[1, 0, 1], [0, 0, 1], [0, 0, 0]]
    assert result["accuracy"] == 1 / 3 and result["balanced_accuracy_present"] == 0.25


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg diperlukan untuk µ-law")
def test_mulaw_encodes_examples_independently_at_same_length():
    x = np.random.default_rng(3).normal(0, 0.02, (2, 1280)).astype(np.float32)
    batch = mulaw8k(x)
    np.testing.assert_array_equal(batch, np.concatenate([mulaw8k(v[None]) for v in x]))
    assert x.shape == batch.shape and not np.array_equal(x, batch)
