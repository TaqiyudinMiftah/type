from collections import Counter
from copy import deepcopy

import pytest

from research.run_loq_first_session import check_query
from research.run_loq_transfer import (
    CLASSES,
    SCHEMA,
    check_pair,
    permits_exploration,
    select_cohort,
)


def example(session, role, day, study="study-a", microphone="mic-a"):
    return {
        "schema": SCHEMA,
        "session_id": session,
        "participant_id": "participant",
        "study_id": study,
        "role": role,
        "local_day": day,
        "files": {"audio.wav": session + "-audio", "events.jsonl": session + "-events"},
        "study": {"setup": {"laptop": "Lenovo LOQ", "keyboard": "built-in"}},
        "track_settings": {"microphone_label": "Realtek", "device_id_hash": microphone},
    }


def test_first_session_can_be_scored_without_support_and_rejects_wrong_inputs():
    source = example("source", "calibration", "2026-11-21")
    query = example("first-test", "test1", "2026-11-22")
    check_query(source, query)
    with pytest.raises(ValueError, match="hari setelah kalibrasi"):
        check_query(source, {**query, "local_day": "2026-11-21"})
    with pytest.raises(ValueError, match="audio dan log ketukan"):
        check_query(source, {**query, "files": source["files"]})
    with pytest.raises(ValueError, match="mikrofon yang sama"):
        check_query(source, example("other", "test1", "2026-11-22", microphone="mic-b"))
    with pytest.raises(ValueError, match="paket Uji pertama"):
        check_query(source, example("second-test", "test2", "2026-11-23"))


def test_prospective_pair_reserves_distinct_recordings_and_same_target_microphone():
    source = example("source", "calibration", "2026-11-21")
    support = example("support", "test1", "2026-11-22")
    query = example("query", "test2", "2026-11-23")
    check_pair(source, support, query, "same-mic")
    reused = deepcopy(query)
    reused["files"]["audio.wav"] = support["files"]["audio.wav"]
    with pytest.raises(ValueError, match="ulang berkas"):
        check_pair(source, support, reused, "same-mic")

    # A new microphone can be captured on the day of the original calibration.
    new_support = example("new-support", "calibration", "2026-11-21", "study-b", "mic-b")
    new_query = example("new-query", "test1", "2026-11-22", "study-b", "mic-b")
    check_pair(source, new_support, new_query, "new-mic")
    new_query["track_settings"]["device_id_hash"] = "mic-c"
    with pytest.raises(ValueError, match="mikrofon target"):
        check_pair(source, new_support, new_query, "new-mic")


def test_available_budgets_preserve_full_class_coverage():
    support = Counter({key: 7 for key in CLASSES})
    query = Counter({key: 5 for key in CLASSES})
    assert select_cohort(support, query, historical=False) == (list(CLASSES), (1, 2, 3, 5))
    query["q"] = 4
    assert select_cohort(support, query, historical=False) == ([], ())


def test_exploration_rejects_affine_with_failed_validation_or_too_little_support():
    provisional = {
        "affine": {"slope": 1, "intercept": 0},
        "issues": ["insufficient_training_support", "weak_or_ambiguous_calibration"],
        "training_inliers_by_phase": {"sync_start": 3, "sync_end": 2},
        "validation_markers_by_phase": {"sync_start": 4, "sync_end": 3},
        "max_abs_validation_residual_ms": 12.5,
    }
    assert permits_exploration(provisional)
    assert not permits_exploration({**provisional, "max_abs_validation_residual_ms": 21})
    assert not permits_exploration({**provisional, "issues": ["capture_incomplete"]})
    assert not permits_exploration({**provisional, "training_inliers_by_phase": {"sync_end": 1}})
