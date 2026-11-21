import numpy as np
import pytest

from ketik_audio.common import ResearchError
from ketik_audio.skaid_pilot_common import KEYS
from ketik_audio.skaid_pilot_data import resample_and_pad, split_plan


def fake_candidates():
    records = []
    for subject in range(17):
        token = f"subject{subject:02d}"
        for phase in ("Phase 1", "Phase 2"):
            for label, key in enumerate(KEYS):
                count = 10 + label % 7 if phase == "Phase 1" else 8
                for row in range(count):
                    records.append(
                        {
                            "id": f"{token}:{phase}:{key}:{row}",
                            "session_token": token,
                            "phase": phase,
                            "row": label * 30 + row,
                            "key": key,
                            "label": label,
                            "audio": f"{token}/{phase}.m4a",
                        }
                    )
    return records


def test_budget_nested_supports_balanced_fixed_queries_and_disjoint_recordings():
    records, plan = split_plan(fake_candidates())
    assert len(plan["plans"]) == 255
    assert len({r["id"] for r in records}) == len(records)
    y = np.array([r["label"] for r in records])
    for subject in range(17):
        rows = [p for p in plan["plans"] if p["subject"] == subject]
        assert len({tuple(p["query"]) for p in rows}) == 1
        for p in rows:
            assert np.array_equal(np.bincount(y[p["support"]], minlength=12), np.full(12, p["k"]))
            assert np.array_equal(np.bincount(y[p["query"]], minlength=12), np.full(12, 5))
            assert {records[i]["phase"] for i in p["support"]} == {"Phase 1"}
            assert {records[i]["phase"] for i in p["query"]} == {"Phase 2"}
            assert not set(p["support"]) & set(p["query"])
            larger = next(v for v in rows if v["draw"] == p["draw"] and v["k"] == 10)
            assert set(p["support"]) <= set(larger["support"])


def test_split_rejects_insufficient_queries_duplicate_identity_and_shared_capture():
    records = fake_candidates()
    duplicate = records + [records[0]]
    with pytest.raises(ResearchError):
        split_plan(duplicate)
    sparse = [
        r
        for r in records
        if not (
            r["session_token"] == "subject00"
            and r["phase"] == "Phase 2"
            and r["key"] == KEYS[0]
            and r["row"] >= 4
        )
    ]
    with pytest.raises(ResearchError):
        split_plan(sparse)
    overlap = [
        {**r, "audio": "same.m4a"} if r["session_token"] == "subject00" else r for r in records
    ]
    with pytest.raises(ResearchError):
        split_plan(overlap)


def test_native_impulse_alignment_and_padding_preserve_reference_press_at_50ms():
    native = np.zeros((3840, 1), dtype=np.float32)
    native[720] = 1  # Press at15ms in the80ms native crop.
    wave, padded = resample_and_pad(native)
    assert wave.shape == (1280,) and padded.shape == (6400,)
    assert np.argmax(np.abs(wave)) == 240
    assert np.argmax(np.abs(padded)) == 800
    assert np.array_equal(padded[560:1840], wave)
    assert np.all(padded[:560] == 0) and np.all(padded[1840:] == 0)
    with pytest.raises(ResearchError):
        resample_and_pad(native[:-1])
