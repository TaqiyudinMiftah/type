from __future__ import annotations

import csv
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from ketik_audio.common import ResearchError, read_json
from ketik_audio.dataset import load_session
from ketik_audio.recording import make_prompts, record


def fake_hardware(monkeypatch, sequence, overflow=False):
    clock = [3.0]
    state = {"terminal_active": False}
    keys = iter(sequence)

    class FakeKeys:
        def __enter__(self):
            state["terminal_active"] = True
            return self

        def read(self, timeout=0.1):
            if timeout == 0:
                return None
            clock[0] += 1
            key = next(keys)
            if overflow and clock[0] == 5:
                state["capture"].overflows = 1
            if key == "interrupt":
                raise KeyboardInterrupt
            return key

        def __exit__(self, *args):
            state["terminal_active"] = False

    class FakeCapture:
        def __init__(self, path, device):
            self.path = path
            self.first_adc = 0.0  # A valid origin; must not be treated as missing.
            self.frames = 1_000_000
            self.rate = 8000
            self.device_name = "mock microphone"
            self.overflows = 0
            self.clipped_samples = 0
            self.error = None
            self.stream = self
            state["capture"] = self

        @property
        def time(self):
            return clock[0]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            sf.write(self.path, np.zeros(self.frames, dtype=np.float32), self.rate, subtype="FLOAT")

    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(query_devices=lambda *args: {}, PortAudioError=RuntimeError),
    )
    monkeypatch.setattr("ketik_audio.recording.TerminalKeys", FakeKeys)
    monkeypatch.setattr("ketik_audio.recording.AudioCapture", FakeCapture)
    monkeypatch.setattr("ketik_audio.recording.time.sleep", lambda seconds: None)
    return state


def test_recording_pause_wrong_key_and_zero_clock_origin(monkeypatch, tmp_path):
    prompts = make_prompts(1, 43)
    wrong = "a" if prompts[1] != "a" else "b"
    sequence = [prompts[0], " ", "\n", wrong, *prompts[1:]]
    state = fake_hardware(monkeypatch, sequence)
    metadata = record(tmp_path, 1, per_key=1)
    assert metadata["status"] == "complete"
    assert metadata["accepted_events"] == 26
    assert len(metadata["ignored_intervals_samples"]) == 1
    assert not state["terminal_active"]
    session = load_session(tmp_path, 1)
    assert session.times[0] == 4.0
    assert len(session.labels) == 26
    assert len(session.truth_labels) == 27  # Actual wrong key remains in sequence truth.
    assert len(session.ignored_intervals) == 1


def test_recording_interrupt_and_overflow_preserve_incomplete_session(monkeypatch, tmp_path):
    state = fake_hardware(monkeypatch, ["interrupt"])
    result = record(tmp_path, 1, per_key=1)
    assert result["status"] == "aborted"
    assert not state["terminal_active"]
    assert (tmp_path / "raw/session-01/audio.wav").is_file()
    prompts = make_prompts(1, 43)
    state = fake_hardware(monkeypatch, prompts, overflow=True)
    with pytest.raises(ResearchError, match="Buffer audio hilang"):
        record(tmp_path, 1, per_key=1, retry=True)
    metadata = read_json(tmp_path / "raw/session-01/session.json")
    assert metadata["status"] == "invalid" and metadata["input_overflows"] == 1
    assert not state["terminal_active"]
    assert len(list((tmp_path / "failed").glob("session-01-*"))) == 1
    with (tmp_path / "raw/session-01/events.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows and all(int(row["sample_index"]) > 0 for row in rows)
