from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import zipfile

import numpy as np
import pytest
import soundfile as sf

from ketik_audio.cli import main
from ketik_audio.common import LABELS, ResearchError
from ketik_audio.web_collection import import_session


def encode(value):
    return (json.dumps(value, allow_nan=False) + "\n").encode()


@pytest.fixture(scope="module")
def payload():
    rate, first = 16_000, 4096
    audio = np.zeros(36 * rate, dtype=np.float32)
    events, markers = [], []
    steps = [(3.1 + i * 0.83, " ", "sync_start") for i in range(8)]
    steps += [(12, "a", "typing"), (13, "b", "typing")]
    steps += [(26 + i * 0.83, " ", "sync_end") for i in range(8)]
    for i, (seconds, key, phase) in enumerate(steps):
        at = round(seconds * rate)
        event = {
            "seq": i,
            "type": "keydown",
            "phase": phase,
            "time_ms": 1000 + seconds * 1000,
            "handler_ms": 1000 + seconds * 1000 + 0.1,
            "is_trusted": False,
            "key": key,
            "code": "Space" if key == " " else "Key" + key.upper(),
            "repeat": False,
            "is_composing": False,
            "shift": False,
            "ctrl": False,
            "alt": False,
            "meta": False,
        }
        peak = at + 680  # 40 ms input delay plus center of one 5 ms energy window.
        audio[peak - 5 : peak + 6] = 0.4
        if key == " ":
            event["calibration"] = True
            markers.append(
                {
                    "event_seq": i,
                    "phase": phase,
                    "clock_frame": at,
                    "peak_frame": peak,
                    "reason": None,
                }
            )
        events.append(event)
    anchors = [
        {"before_ms": 1000 + 500 * i, "after_ms": 1000 + 500 * i, "context_s": first / rate + i / 2}
        for i in range(73)
    ]
    timing = {
        "method": "calibration-affine-energy-v1",
        "status": "checked",
        "first_frame": first,
        "sample_rate": rate,
        "anchors": anchors,
        "calibration": markers,
        "affine": {"slope": 1.0, "intercept": 680.0},
        "residuals_ms": [0] * 8,
        "quiet_rms": 0.0,
        "issues": [],
    }
    study = {
        "schema": "ketik-web-study/v1",
        "study_id": "22222222-2222-4222-8222-222222222222",
        "participant_id": "11111111-1111-4111-8111-111111111111",
        "timezone": "Asia/Jakarta",
        "setup": {"laptop": "fixture", "keyboard": "fixture"},
        "days": {"calibration": "2026-11-20", "test1": None, "test2": None},
    }
    meta = {
        "schema": "ketik-web-session/v1",
        "session_id": "33333333-3333-4333-8333-333333333333",
        "study_id": study["study_id"],
        "participant_id": study["participant_id"],
        "study": study,
        "role": "calibration",
        "timezone": study["timezone"],
        "local_day": "2026-11-20",
        "started_at": "2026-11-20T01:00:00Z",
        "finished_at": "2026-11-20T01:00:36Z",
        "status": "complete",
        "synthetic": True,
        "sample_rate": rate,
        "frames": len(audio),
        "channels": 1,
        "format": "IEEE_FLOAT_32_LE",
        "first_frame": first,
        "consent": {"version": "local-audio-key-events-v1"},
        "capture_issues": [],
        "timing_status": "checked",
        "event_count": len(events),
        "audio_blocks": [{"seq": 0, "first_frame": first, "frames": len(audio)}],
        "recorded_counts": {c: int(c in "ab") for c in LABELS},
        "prompts": [{"text": "aaaaaaaa"}],
    }
    wav = io.BytesIO()
    sf.write(wav, audio, rate, format="WAV", subtype="FLOAT")
    return {
        "audio.wav": wav.getvalue(),
        "events.jsonl": b"".join(encode(e) for e in events),
        "timing.json": encode(timing),
        "session.json": encode(meta),
    }


def package(tmp_path, files):
    data = copy.deepcopy(files)
    meta = json.loads(data["session.json"])
    meta["files"] = {
        name: hashlib.sha256(value).hexdigest()
        for name, value in data.items()
        if name != "session.json"
    }
    data["session.json"] = encode(meta)
    out = tmp_path / "session.zip"
    with zipfile.ZipFile(out, "w") as archive:
        for name, value in data.items():
            archive.writestr(name, value)
    return out


def test_roundtrip_independent_clock_fixture_and_actual_labels(tmp_path, payload):
    result = import_session(package(tmp_path, payload), tmp_path / "out")
    assert result["status"] == "checked" and result["synthetic"]
    assert result["timing"]["matches_browser_report"]
    assert abs(result["timing"]["affine"]["intercept"] - 680) < 1e-5
    assert result["coverage"]["recorded_counts"]["a"] == 1
    assert result["coverage"]["recorded_counts"]["b"] == 1
    assert len(result["coverage"]["candidate_events"]) == 2
    assert not result["classifier_fit_performed"]
    assert (tmp_path / "out/audio.wav").read_bytes() == payload["audio.wav"]


def test_no_overwrite(tmp_path, payload):
    source = package(tmp_path, payload)
    import_session(source, tmp_path / "out")
    with pytest.raises(ResearchError, match="Tujuan sudah ada"):
        import_session(source, tmp_path / "out")


def test_declared_timing_is_recomputed_not_trusted(tmp_path, payload):
    files = dict(payload)
    timing = json.loads(files["timing.json"])
    timing["affine"]["intercept"] += 1000
    files["timing.json"] = encode(timing)
    result = import_session(package(tmp_path, files), tmp_path / "out")
    assert result["status"] == "needs_review"
    assert "browser_timing_mismatch" in result["timing"]["issues"]
    assert not result["coverage"]["candidate_events"]


def test_hash_tampering_rejected_before_output(tmp_path, payload):
    source = package(tmp_path, payload)
    with zipfile.ZipFile(source) as archive:
        data = {n: archive.read(n) for n in archive.namelist()}
    data["events.jsonl"] += b" "
    with zipfile.ZipFile(source, "w") as archive:
        for name, value in data.items():
            archive.writestr(name, value)
    with pytest.raises(ResearchError, match="Hash berbeda"):
        import_session(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("extra", ["../escape.txt", "/absolute.txt", "extra.json"])
def test_unexpected_archive_paths_rejected(tmp_path, payload, extra):
    data = {**payload, extra: b"no"}
    source = package(tmp_path, data)
    with pytest.raises(ResearchError, match="empat berkas"):
        import_session(source, tmp_path / "out")


def test_same_day_testing_is_not_accepted(tmp_path, payload):
    files = dict(payload)
    meta = json.loads(files["session.json"])
    meta["role"] = "test1"
    meta["study"]["days"]["test1"] = meta["local_day"]
    files["session.json"] = encode(meta)
    with pytest.raises(ResearchError, match="Hari studi"):
        import_session(package(tmp_path, files), tmp_path / "out")


def test_partial_session_is_preserved_and_flagged(tmp_path, payload):
    files = dict(payload)
    meta = json.loads(files["session.json"])
    meta["status"] = "interrupted"
    meta["capture_issues"] = ["window_blurred"]
    files["session.json"] = encode(meta)
    result = import_session(package(tmp_path, files), tmp_path / "out")
    assert result["status"] == "needs_review" and result["capture_status"] == "interrupted"
    assert "window_blurred" in result["timing"]["issues"]


def test_nonfinite_audio_is_rejected(tmp_path, payload):
    files = dict(payload)
    samples, rate = sf.read(io.BytesIO(files["audio.wav"]), dtype="float32")
    samples[500] = np.nan
    stream = io.BytesIO()
    sf.write(stream, samples, rate, format="WAV", subtype="FLOAT")
    files["audio.wav"] = stream.getvalue()
    with pytest.raises(ResearchError, match="tidak finite"):
        import_session(package(tmp_path, files), tmp_path / "out")


def test_dispatch_delay_and_cli_import(tmp_path, payload, capsys):
    files = dict(payload)
    rows = [json.loads(line) for line in files["events.jsonl"].splitlines()]
    rows[0]["handler_ms"] += 30
    files["events.jsonl"] = b"".join(encode(row) for row in rows)
    source = package(tmp_path, files)
    assert main(["web-import", str(source), "--output", str(tmp_path / "out")]) == 0
    assert "needs_review" in capsys.readouterr().out
    report = json.loads((tmp_path / "out/review.json").read_bytes())
    assert "event_dispatch_lag" in report["timing"]["issues"]


def test_robust_cli_keeps_original_report_and_raw_bytes(tmp_path, payload, capsys):
    source = package(tmp_path, payload)
    out = tmp_path / "robust"
    assert (
        main(["web-align", str(source), "--output", str(out), "--evaluation-role", "development"])
        == 0
    )
    text = capsys.readouterr().out
    assert "status=checked" in text and "DATA SINTETIS" in text
    original = json.loads((out / "review.json").read_bytes())
    robust = json.loads((out / "alignment-v2.json").read_bytes())
    assert original["schema"] == "ketik-web-import/v1"
    assert original["status"] == robust["original_v1_status"] == "checked"
    assert robust["method"] == "calibration-affine-consensus-v2"
    assert robust["evaluation_role"] == "development"
    assert not robust["validation_used_for_fit_or_model_selection"]
    assert not robust["classifier_fit_performed"]
    assert len(robust["coverage"]["candidate_events"]) == 2
    assert (out / "audio.wav").read_bytes() == payload["audio.wav"]
    with pytest.raises(ResearchError, match="Tujuan sudah ada"):
        from ketik_audio.web_alignment import align_package

        align_package(source, out)


def test_robust_audit_cannot_promote_an_interrupted_capture(tmp_path, payload):
    from ketik_audio.web_alignment import align_package

    files = dict(payload)
    meta = json.loads(files["session.json"])
    meta["status"] = "interrupted"
    files["session.json"] = encode(meta)
    result = align_package(package(tmp_path, files), tmp_path / "robust")
    assert result["status"] == "needs_review"
    assert "capture_incomplete" in result["issues"]
    assert not result["coverage"]["candidate_events"]


def quota_files(payload, complete=True, duration=36):
    """Construct v2 events independently, retaining the known v1 acoustic markers."""
    from collections import Counter

    from ketik_audio.web_quota import CLASSES, config

    files = dict(payload)
    meta = json.loads(files["session.json"])
    old_events = [json.loads(line) for line in files["events.jsonl"].splitlines()]
    start = [e for e in old_events if e["phase"] == "sync_start"]
    end = [e for e in old_events if e["phase"] == "sync_end"]
    typed = []
    keys = "a " + "bcdefghijklmnopqrstuvwxyz" if complete else "a "
    for key in keys:
        for _ in range(30 if complete and key in "a " else 25 if complete else 7):
            at = 13000 + len(typed) * 12
            typed.append(
                {
                    **old_events[8],
                    "key": key,
                    "code": "Space" if key == " " else "Key" + key.upper(),
                    "time_ms": at,
                    "handler_ms": at + 0.1,
                    "collection_mode": "topup",
                }
            )
    chosen, raw, sequences = Counter(), Counter(), []
    events = start + typed + end
    for i, row in enumerate(events):
        row["seq"] = i
        label = "space" if row["key"] == " " else row["key"].lower()
        row["quota_selected"] = row["phase"] == "typing" and chosen[label] < 25
        if row["phase"] == "typing":
            raw[label] += 1
        if row["quota_selected"]:
            chosen[label] += 1
            sequences.append(i)
    timing = json.loads(files["timing.json"])
    for marker, event in zip(timing["calibration"], start + end, strict=True):
        marker["event_seq"] = event["seq"]
    meta["schema"] = "ketik-web-session/v2"
    meta["study"]["schema"] = "ketik-web-study/v2"
    meta["study"]["collection"] = config()
    meta["track_settings"] = {"microphone_label": "Fake quota fixture"}
    meta["preflight"] = {
        "method": "key-signal-v1",
        "status": "passed",
        "passed": 8,
        "total": 8,
        "quiet_rms": 0.0,
        "sample_rate": meta["sample_rate"],
        "microphone_label": "Fake quota fixture",
        "checked_at": "2026-11-20T00:59:59Z",
        "markers": [
            {**m, "peak_rms": 0.02, "peak_abs": 0.4, "threshold": 1e-4}
            for m in timing["calibration"][:8]
        ],
    }
    meta["collection"] = {
        **config(),
        "raw_counts": {k: raw[k] for k in CLASSES},
        "selected_counts": {k: chosen[k] for k in CLASSES},
        "selected_sequences": sequences,
        "selected_total": len(sequences),
        "complete": complete,
    }
    meta["recorded_counts"] = meta["collection"]["raw_counts"]
    meta["event_count"] = len(events)
    if not complete:
        meta["status"] = "interrupted"
        meta["study"]["days"]["calibration"] = None
    if duration != 36:
        audio, rate = sf.read(io.BytesIO(files["audio.wav"]), dtype="float32")
        samples = np.zeros(duration * rate, dtype=np.float32)
        samples[: len(audio)] = audio
        wav = io.BytesIO()
        sf.write(wav, samples, rate, format="WAV", subtype="FLOAT")
        files["audio.wav"] = wav.getvalue()
        meta["frames"] = len(samples)
        meta["audio_blocks"][0]["frames"] = len(samples)
        minutes, seconds = divmod(duration, 60)
        meta["finished_at"] = f"2026-11-20T01:{minutes:02d}:{seconds:02d}Z"
    files["session.json"] = encode(meta)
    files["events.jsonl"] = b"".join(encode(e) for e in events)
    files["timing.json"] = encode(timing)
    return files


def test_quota_roundtrip_has_27_balanced_classes_and_space(tmp_path, payload):
    result = import_session(package(tmp_path, quota_files(payload)), tmp_path / "v2")
    assert result["schema"] == "ketik-web-import/v2"
    assert result["status"] == "checked"
    assert result["collection"]["selected_total"] == 675
    assert len(result["collection"]["selected_counts"]) == 27
    assert all(v == 25 for v in result["collection"]["selected_counts"].values())
    assert result["collection"]["raw_counts"]["space"] == 30
    assert result["collection"]["raw_counts"]["a"] == 30
    assert result["coverage"]["candidate_scope"] == "quota_selected"
    assert "space" in result["coverage"]["candidate_counts"]
    assert result["timing"]["matches_browser_report"]


def test_partial_quota_is_preserved_without_claiming_675(tmp_path, payload):
    result = import_session(
        package(tmp_path, quota_files(payload, complete=False)), tmp_path / "v2"
    )
    assert result["status"] == "needs_review"
    assert result["collection"]["selected_total"] == 14
    assert not result["collection"]["complete"]


@pytest.mark.parametrize(
    "mutation", ["selected_event", "calibration_space", "manifest", "false_complete"]
)
def test_quota_contradictions_rejected(tmp_path, payload, mutation):
    files = quota_files(payload, complete=mutation != "false_complete")
    meta = json.loads(files["session.json"])
    events = [json.loads(line) for line in files["events.jsonl"].splitlines()]
    if mutation == "selected_event":
        next(e for e in events if e["quota_selected"])["quota_selected"] = False
    elif mutation == "calibration_space":
        events[0]["quota_selected"] = True
    elif mutation == "manifest":
        meta["collection"]["selected_counts"]["a"] = 26
    else:
        meta["status"] = "complete"
        meta["study"]["days"]["calibration"] = meta["local_day"]
    files["session.json"] = encode(meta)
    files["events.jsonl"] = b"".join(encode(e) for e in events)
    with pytest.raises(ResearchError, match="kuota"):
        import_session(package(tmp_path, files), tmp_path / "v2")
    assert not (tmp_path / "v2").exists()


@pytest.mark.parametrize("mutation", ["not_passed", "weak", "clipping", "different_microphone"])
def test_preflight_readiness_metadata_must_be_consistent(tmp_path, payload, mutation):
    files = quota_files(payload)
    meta = json.loads(files["session.json"])
    if mutation == "not_passed":
        meta["preflight"]["status"] = "needs_review"
    elif mutation == "weak":
        meta["preflight"]["markers"][0]["peak_rms"] = 1e-5
    elif mutation == "clipping":
        meta["preflight"]["markers"][0]["peak_abs"] = 1.2
    else:
        meta["track_settings"]["microphone_label"] = "Different microphone"
    files["session.json"] = encode(meta)
    with pytest.raises(ResearchError, match="[Pp]emeriksaan|[Mm]ikrofon"):
        import_session(package(tmp_path, files), tmp_path / "v2")


def test_v2_longer_than_old_5_minute_guard_imports(tmp_path, payload):
    result = import_session(package(tmp_path, quota_files(payload, duration=362)), tmp_path / "v2")
    assert result["duration_seconds"] == 362
    assert result["collection"]["complete"]


def test_v2_pcm_byte_bound_is_enforced_before_audio_read(tmp_path, payload):
    from ketik_audio.web_quota import MAX_AUDIO_BYTES

    files = quota_files(payload)
    meta = json.loads(files["session.json"])
    meta["frames"] = MAX_AUDIO_BYTES // 4 + 1
    files["session.json"] = encode(meta)
    with pytest.raises(ResearchError, match="Jumlah sampel"):
        import_session(package(tmp_path, files), tmp_path / "v2")


def test_robust_alignment_handles_space_quota_and_keeps_original_v2_review(tmp_path, payload):
    from ketik_audio.web_alignment import align_package

    result = align_package(package(tmp_path, quota_files(payload)), tmp_path / "v2", "development")
    assert result["collection"]["selected_counts"]["space"] == 25
    assert result["coverage"]["candidate_scope"] == "quota_selected"
    assert json.loads((tmp_path / "v2/review.json").read_bytes())["schema"] == "ketik-web-import/v2"


def adaptive_quota_files(payload):
    files = quota_files(payload)
    meta = json.loads(files["session.json"])
    preflight = meta["preflight"]
    preflight.update(
        method="key-signal-v2", quiet_rms=0.005, quiet_p95_rms=0.0052,
        quiet_raw_rms=0.005, baseline_estimator="median-5ms-ac-rms",
        min_snr_db=6, floor_rms=1e-6,
    )
    for marker in preflight["markers"]:
        marker.update(
            peak_rms=0.015, threshold=0.005 * 10 ** (6 / 20),
            snr_db=20 * math.log10(3), timing_ambiguous=True,
        )
    meta["study"]["setup"]["device_name"] = "Lenovo ThinkPad T14"
    files["session.json"] = encode(meta)
    return files


def test_adaptive_readiness_accepts_relative_snr_and_keeps_timing_independent(tmp_path, payload):
    result = import_session(package(tmp_path, adaptive_quota_files(payload)), tmp_path / "adaptive")
    assert result["preflight"]["method"] == "key-signal-v2"
    assert result["collection"]["selected_total"] == 675
    assert result["timing"]["matches_browser_report"]


@pytest.mark.parametrize("mutation", ["threshold", "snr", "quiet", "parameters", "weak"])
def test_adaptive_readiness_contradictions_rejected(tmp_path, payload, mutation):
    files = adaptive_quota_files(payload)
    meta = json.loads(files["session.json"])
    check = meta["preflight"]
    if mutation == "threshold":
        check["markers"][0]["threshold"] = 1e-6
    elif mutation == "snr":
        check["markers"][0]["snr_db"] = 25
    elif mutation == "quiet":
        check["quiet_p95_rms"] = 0.001
    elif mutation == "parameters":
        check["min_snr_db"] = 0
    else:
        check["markers"][0]["peak_rms"] = 0.005
    files["session.json"] = encode(meta)
    with pytest.raises(ResearchError, match="[Pp]emeriksaan"):
        import_session(package(tmp_path, files), tmp_path / "adaptive")


def test_additional_modifier_counts_are_recomputed_separately_from_quota(tmp_path, payload):
    from ketik_audio.web_quota import additional_counts

    files = adaptive_quota_files(payload)
    meta = json.loads(files["session.json"])
    events = [json.loads(line) for line in files["events.jsonl"].splitlines()]
    template = next(e for e in events if e["phase"] == "typing")
    extras = [
        {**template, "key": "Control", "code": "ControlLeft", "ctrl": True},
        {**template, "key": "Control", "code": "ControlRight", "ctrl": True},
        {**template, "key": "Backspace", "code": "Backspace"},
    ]
    first_end = next(i for i, e in enumerate(events) if e["phase"] == "sync_end")
    for i, event in enumerate(extras):
        event.update(
            time_ms=24000 + i * 20, handler_ms=24000 + i * 20 + 0.1, quota_selected=False
        )
    events[first_end:first_end] = extras
    for i, event in enumerate(events):
        event["seq"] = i
    meta["additional_keys"] = additional_counts(events)
    meta["event_count"] = len(events)
    timing = json.loads(files["timing.json"])
    for marker in timing["calibration"][8:]:
        marker["event_seq"] += 3
    files["session.json"] = encode(meta)
    files["events.jsonl"] = b"".join(encode(e) for e in events)
    files["timing.json"] = encode(timing)
    result = import_session(package(tmp_path, files), tmp_path / "extra")
    assert result["collection"]["selected_total"] == 675
    assert result["additional_keys"]["counts"]["ControlLeft"] == 1
    assert result["additional_keys"]["counts"]["ControlRight"] == 1
    assert result["additional_keys"]["counts"]["Backspace"] == 1
    meta["additional_keys"]["counts"]["ControlLeft"] = 25
    files["session.json"] = encode(meta)
    with pytest.raises(ResearchError, match="tombol tambahan"):
        import_session(package(tmp_path, files), tmp_path / "wrong-extra")


def context_quota_files(payload, *, embedded=False):
    files = adaptive_quota_files(payload)
    meta = json.loads(files["session.json"])
    check = meta["preflight"]
    check.update(method="key-signal-v3", first_frame=meta["first_frame"], frames=meta["frames"])
    for marker in check["markers"]:
        marker.update(
            frame_source="audio-context-snapshot",
            event_audio_context_s=(marker["clock_frame"] + check["first_frame"])
            / check["sample_rate"],
        )
    if embedded:
        check.update(source="session-sync-start", frames=12 * meta["sample_rate"])
        events = [json.loads(line) for line in files["events.jsonl"].splitlines()]
        for marker in check["markers"]:
            events[marker["event_seq"]]["audio_context_s"] = marker["event_audio_context_s"]
        files["events.jsonl"] = b"".join(encode(event) for event in events)
    files["session.json"] = encode(meta)
    return files


@pytest.mark.parametrize("embedded", [False, True])
def test_context_clock_readiness_roundtrip_keeps_older_timing_audit(tmp_path, payload, embedded):
    files = context_quota_files(payload, embedded=embedded)
    result = import_session(package(tmp_path, files), tmp_path / "context")
    assert result["preflight"]["method"] == "key-signal-v3"
    assert ("audio.wav" in result["preflight"]["evidence"]) == embedded
    assert result["timing"]["matches_browser_report"]
    assert result["collection"]["selected_total"] == 675


@pytest.mark.parametrize("mutation", ["context", "range", "source"])
def test_context_snapshot_contradictions_are_rejected(tmp_path, payload, mutation):
    files = context_quota_files(payload)
    meta = json.loads(files["session.json"])
    check = meta["preflight"]
    if mutation == "context":
        check["markers"][0]["event_audio_context_s"] += 1
    elif mutation == "range":
        check["frames"] = 1
    else:
        check["markers"][0]["frame_source"] = "unverified"
    files["session.json"] = encode(meta)
    with pytest.raises(ResearchError, match="[Pp]emeriksaan|AudioContext"):
        import_session(package(tmp_path, files), tmp_path / "context")


def test_embedded_check_must_reference_the_recorded_start_taps(tmp_path, payload):
    files = context_quota_files(payload, embedded=True)
    events = [json.loads(line) for line in files["events.jsonl"].splitlines()]
    events[0]["audio_context_s"] += 0.05
    files["events.jsonl"] = b"".join(encode(event) for event in events)
    with pytest.raises(ResearchError, match="event awal sesi"):
        import_session(package(tmp_path, files), tmp_path / "context")
