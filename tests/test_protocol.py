from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from ketik_audio.audio import crop, detect, extract
from ketik_audio.cli import main
from ketik_audio.common import LABELS, FeatureConfig, ResearchError, role_for_session
from ketik_audio.devices import configure_rx6000, is_rx6000, select_device
from ketik_audio.metrics import detection, edit_distance
from ketik_audio.recording import TerminalKeys, make_prompts, sample_index


def test_balanced_randomized_collection_and_fixed_roles():
    prompts = make_prompts(25, 42)
    assert Counter(prompts) == dict.fromkeys(LABELS, 25)
    assert prompts != make_prompts(25, 43)
    assert [role_for_session(n) for n in range(1, 7)] == ["train"] * 4 + ["validation", "test"]
    with pytest.raises(ResearchError):
        role_for_session(7)


def test_labels_use_microphone_clock_not_wall_clock():
    # ADC clock can have an arbitrary origin and precede callback delivery.
    assert sample_index(903.125, 900.0, 48_000) == 150_000


def test_detection_and_cer_account_for_missing_and_extra_keys():
    measured = detection([1.0, 2.0, 3.0], [1.03, 2.95, 4.0], 0.075)
    assert measured["matched"] == 2
    assert measured["missed"] == measured["extra"] == 1
    assert measured["f1"] == pytest.approx(2 / 3)
    assert edit_distance("abc", "axbcx") == 2
    assert edit_distance("abc", "") == 3
    assert edit_distance("", "abc") == 3


def test_empty_and_boundary_audio_features_are_finite():
    config = FeatureConfig()
    assert len(detect(np.zeros(44_100, dtype=np.float32), config)) == 0
    assert len(detect(np.zeros(0, dtype=np.float32), config)) == 0
    for seconds in [0.0, 0.5, 1.0]:
        segment = crop(np.zeros(44_100, dtype=np.float32), seconds, config)
        mel, mfcc = extract(segment, config)
        assert mel.shape == (64, 48)
        assert mfcc.shape == (120,)
        assert np.isfinite(mel).all() and np.isfinite(mfcc).all()


def test_gpu_selection_never_silently_ignores_explicit_gpu(monkeypatch):
    monkeypatch.setattr(
        "ketik_audio.devices.gpu_probe",
        lambda: {"available": False, "reason": "GPU tidak kompatibel", "backend": "rocm"},
    )
    selected, metadata = select_device("auto")
    assert selected == "cpu" and metadata["fallback_reason"] == "GPU tidak kompatibel"
    with pytest.raises(ResearchError, match="tidak kompatibel"):
        select_device("gpu")
    monkeypatch.setattr(
        "ketik_audio.devices.gpu_probe",
        lambda: {"available": True, "reason": None, "backend": "rocm", "name": "AMD"},
    )
    assert select_device("gpu")[0] == "cuda"


def test_terminal_settings_restored_on_exception():
    import os
    import termios

    master, slave = os.openpty()
    before = termios.tcgetattr(slave)
    try:
        with pytest.raises(RuntimeError), TerminalKeys(slave):
            assert termios.tcgetattr(slave) != before
            raise RuntimeError("interrupted")
        assert termios.tcgetattr(slave) == before
    finally:
        os.close(master)
        os.close(slave)


def test_cli_reports_missing_data_without_traceback(tmp_path, capsys):
    assert main(["prepare", "--data", str(tmp_path / "missing")]) == 2
    assert "Gagal:" in capsys.readouterr().err


def test_doctor_does_not_call_a_detected_but_failed_gpu_ready(monkeypatch, capsys):
    monkeypatch.setattr(
        "ketik_audio.devices.doctor",
        lambda: {
            "recording_ready": True,
            "training_ready": True,
            "audio": {"inputs": [], "reason": None},
            "gpu": {
                "available": False,
                "name": "AMD Radeon RX 6600 LE",
                "backend": "rocm",
                "reason": "Tes GPU melewati batas waktu",
            },
        },
    )
    assert main(["doctor"]) == 0  # Recording and CPU training remain usable.
    output = capsys.readouterr().out
    assert "GPU: belum siap" in output and "melewati batas waktu" in output


def test_rx6000_selection_overrides_old_igpu_mask(monkeypatch):
    import os

    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")
    monkeypatch.delenv("KETIK_ROCM_DEVICE", raising=False)
    configure_rx6000()
    assert os.environ["ROCR_VISIBLE_DEVICES"] == "0"
    assert os.environ["HIP_VISIBLE_DEVICES"] == "0"


@pytest.mark.parametrize(
    "name,expected",
    [
        ("AMD Radeon RX 6600 LE", True),
        ("AMD Radeon RX 6800", True),
        ("AMD Radeon 780M Graphics", False),
        ("AMD Radeon RX 7800 XT", False),
    ],
)
def test_only_rx6000_is_accepted(name, expected):
    assert is_rx6000(name) is expected
