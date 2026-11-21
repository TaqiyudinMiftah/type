from __future__ import annotations

import shutil

import numpy as np
import pytest
import soundfile as sf

from ketik_audio.cli import main
from ketik_audio.common import ResearchError, read_json, write_json
from ketik_audio.dataset import load_prepared, load_session, prepare
from ketik_audio.demo import synthetic_sessions
from ketik_audio.evaluation import evaluate, predict
from ketik_audio.models import Predictor, train_cnn, train_svm
from ketik_audio.reporting import report


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("experiment")
    data = root / "data"
    synthetic_sessions(data)
    # A corrupt/unavailable test session must not affect preparation or tuning.
    (data / "raw/session-06/session.json").rename(data / "test-metadata.json")
    prepared = root / "processed"
    prepare(data, prepared)
    (data / "test-metadata.json").rename(data / "raw/session-06/session.json")
    return root, data, prepared


def test_split_preprocessing_and_checksum_protect_against_leakage(experiment, tmp_path):
    _, _, prepared = experiment
    arrays, manifest = load_prepared(prepared)
    assert set(arrays["sessions"]) == {1, 2, 3, 4, 5}
    assert manifest["split"]["test"] == [6]
    assert manifest["detector_selection"]["source_session"] == 5
    assert manifest["counts"] == {"train": 208, "validation": 52}
    copy = tmp_path / "copy"
    shutil.copytree(prepared, copy)
    with (copy / "features.npz").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ResearchError, match="berubah"):
        load_prepared(copy)


def test_incomplete_or_lost_audio_is_rejected(experiment, tmp_path):
    _, data, _ = experiment
    destination = tmp_path / "data/raw/session-01"
    shutil.copytree(data / "raw/session-01", destination)
    metadata = read_json(destination / "session.json")
    metadata["input_overflows"] = 1
    write_json(destination / "session.json", metadata)
    with pytest.raises(ResearchError, match="audio hilang"):
        load_session(tmp_path / "data", 1)
    metadata["input_overflows"] = 0
    metadata["status"] = "aborted"
    write_json(destination / "session.json", metadata)
    with pytest.raises(ResearchError, match="belum lengkap"):
        load_session(tmp_path / "data", 1)


def test_svm_cnn_prediction_evaluation_and_report(experiment, tmp_path):
    _, data, prepared = experiment
    svm, cnn = tmp_path / "svm", tmp_path / "cnn"
    assert main(["train", "svm", "--dataset", str(prepared), "--output", str(svm)]) == 0
    cnn_result = train_cnn(prepared, cnn, "cpu", epochs=2)
    arrays, _ = load_prepared(prepared)
    train = arrays["sessions"] <= 4
    metadata = read_json(cnn / "model.json")
    np.testing.assert_allclose(
        metadata["normalization_mean"], arrays["mel"][train].mean(axis=(0, 2)), rtol=1e-5
    )
    assert len(cnn_result["seeds"]) == 3
    assert metadata["seed"] in [42, 43, 44]
    svm_result = evaluate(svm, data)
    cnn_evaluation = evaluate(cnn, data)
    assert svm_result["classification"]["accuracy"] > 0.8
    assert svm_result["synthetic"] and svm_result["target_met"] is None
    assert svm_result["detection"]["f1"] > 0.8
    assert len(cnn_evaluation["seeds"]) == 3
    assert "macro_f1" in cnn_evaluation["seed_statistics"]
    assert 0 <= cnn_evaluation["classification"]["macro_f1"] <= 1
    loaded = Predictor(cnn)
    scores = loaded.scores(arrays["mel"][:2], arrays["mfcc"][:2])
    assert scores.shape == (2, 26)
    np.testing.assert_allclose(scores.sum(axis=1), 1, atol=1e-6)
    # Prediction needs the waveform and frozen model, never keyboard labels.
    wav = tmp_path / "unlabelled.wav"
    shutil.copy(data / "raw/session-06/audio.wav", wav)
    predicted = predict(svm, wav)
    assert len(predicted["events"]) >= 50
    silent = tmp_path / "silence.wav"
    sf.write(silent, np.zeros(16_000), 16_000)
    assert predict(svm, silent)["sequence"] == ""
    controls = tmp_path / "controls"
    controls.mkdir()
    shutil.copy(wav, controls / "audio.wav")
    sidecar = read_json(data / "raw/session-06/session.json")
    sidecar["ignored_intervals_samples"] = [[0, sidecar["frames"]]]
    write_json(controls / "session.json", sidecar)
    assert predict(svm, controls / "audio.wav")["sequence"] == ""
    # Inference used the control sidecar without requiring events.csv.
    duplicate = tmp_path / "duplicate/raw/session-06"
    shutil.copytree(data / "raw/session-01", duplicate)
    duplicate_meta = read_json(duplicate / "session.json")
    duplicate_meta.update(session=6, role="test")
    write_json(duplicate / "session.json", duplicate_meta)
    with pytest.raises(ResearchError, match="identik"):
        evaluate(svm, tmp_path / "duplicate", tmp_path / "duplicate-evaluation")
    output = report([svm, cnn], tmp_path / "report")
    text = output.read_text()
    assert "DATA SINTETIS" in text and "tidak mengukur" in text
    assert len(list(output.parent.glob("*.png"))) == 3
    with pytest.raises(ResearchError, match="sudah ada"):
        train_svm(prepared, svm)
    with pytest.raises(ResearchError, match="sudah ada"):
        evaluate(svm, data)
    altered = tmp_path / "altered"
    shutil.copytree(svm, altered)
    with (altered / "svm.joblib").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ResearchError, match="checksum"):
        Predictor(altered)
