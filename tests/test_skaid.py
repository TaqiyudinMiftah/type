import csv
import io

import numpy as np
import pytest
import soundfile as sf

from ketik_audio.common import ResearchError
from ketik_audio.skaid import (
    audit_log,
    decode,
    envelopes,
    integer_timestamp,
    pcm_fingerprint,
    probe,
    recording_identity,
)


def log_blob(rows):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["Session ID", "Timestamp (ms)", "Event", "Key", "Phase"])
    for stamp, event, key, phase in rows:
        writer.writerow(["qt_session_1746047498719", stamp, event, key, phase])
    return stream.getvalue().encode()


def test_decimal_timestamps_preserve_exact_ms_and_report_collapsed_clock():
    assert integer_timestamp("1.746047591212E12") == (1746047591212, 0)
    assert integer_timestamp("1.75E12") == (1750000000000, 10)
    detail, _ = audit_log(
        log_blob(
            [
                ("1.75E12", "press", "a", "Phase 1"),
                ("1.75E12", "release", "a", "Phase 1"),
            ]
        ),
        "Keystroke Logs/qt_1746047498719.csv",
    )
    assert detail["timestamp_unique_count"] == 1
    assert detail["minimum_positive_timestamp_step_ms"] is None
    assert detail["nonpositive_hold_durations"] == 1
    assert detail["session_tokens"] == detail["filename_tokens"] == ["qt_1746047498719"]
    with pytest.raises(ResearchError):
        integer_timestamp("3.5")
    with pytest.raises(ResearchError):
        integer_timestamp("nan")


def test_log_state_preserves_repeats_missing_events_and_cross_phase_holds():
    detail, events = audit_log(
        log_blob(
            [
                (1000, "press", "H", "Phase 1"),
                (1010, "press", "h", "Phase 1"),
                (1090, "release", "h", "Phase 2"),
                (1100, "release", "s", "Phase 2"),
                (1110, "press", "a", "Phase 2"),
                (1110, "press", "a", "Phase 2"),
            ]
        ),
        "Keystroke Logs/qt_1746047498719.csv",
    )
    assert len(events) == 6
    assert detail["matched_initial_press_release"] == 1
    assert len(detail["repeated_press_while_held"]) == 2
    assert detail["unclosed_initial_press_rows"] == [6]
    assert detail["unmatched_release_rows"] == [5]
    assert detail["duplicate_event_tuple_rows"] == [7]
    assert detail["cross_phase_hold_pairs"] == 1
    assert detail["phases"]["Phase 1"]["press_counts"] == {"h": 2}


def test_recording_pairing_requires_explicit_matching_tokens_and_phase():
    prefix = "Participant Recordings/qt_1746047498719/"
    assert recording_identity(prefix + "qt_1746047498719_email_1.m4a")["phase"] == "Phase 1"
    assert recording_identity(prefix + "qt_1746047498719_free_form.wav")["phase"] == "Phase 2"
    assert recording_identity(prefix + "other.m4a") is None
    with pytest.raises(ResearchError, match="berbeda"):
        recording_identity(prefix + "qt_1746047498720_email_1.m4a")
    with pytest.raises(ResearchError):
        recording_identity("../qt_1746047498719_email_1.m4a")


def test_native_envelope_accounts_for_partial_bins_and_antiphase_stereo():
    audio = np.tile([0.5, -0.5], (500, 1)).astype(np.float32)
    edges, rms, peaks = envelopes(audio, 22050)
    assert edges.tolist() == [0, 220, 441, 500]
    assert np.array_equal(rms, [0.5, 0.5, 0.5])
    assert np.array_equal(peaks, rms)
    assert pcm_fingerprint(audio, 22050) != pcm_fingerprint(audio, 44100)
    assert pcm_fingerprint(audio, 22050) != pcm_fingerprint(audio[:, :1], 22050)


def test_ffmpeg_native_decode_preserves_pcm_rate_channels_and_frames(tmp_path):
    path = tmp_path / "known.wav"
    audio = (np.arange(2002).reshape(1001, 2) - 1000).astype(np.float32) / 32768
    sf.write(path, audio, 22050, subtype="PCM_16")
    details, stream = probe(path)
    decoded, rate = decode(path, stream)
    assert int(stream["channels"]) == 2
    assert rate == 22050
    assert np.array_equal(decoded, audio)
    assert details["streams"][0]["codec_name"] == "pcm_s16le"
