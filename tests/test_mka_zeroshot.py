import numpy as np
import pytest

from ketik_audio.common import ResearchError
from ketik_audio.mka_zeroshot import crop_layout, grouped_mean, metrics, resample_mono


@pytest.mark.parametrize("peak,expected", [(0, 0), (5000, 4200), (15999, 9600)])
def test_peak_crop_keeps_context_and_clamps_edges_without_padding(peak, expected):
    wave = np.zeros(16000, dtype=np.float32)
    wave[peak] = -1
    result = crop_layout(wave)
    assert result["peak_frame"] == peak
    assert result["peak_start"] == expected
    assert len(wave[expected : expected + 6400]) == 6400
    assert result["peak_clamped"] == (expected != peak - 800)
    assert result["tiled_starts"] == [0, 3200, 6400, 9600]


def test_tiled_windows_cover_shorter_clip_and_tied_peak_is_first():
    wave = np.zeros(14364, dtype=np.float32)
    wave[[4000, 8000]] = 1
    result = crop_layout(wave)
    assert result["tiled_starts"] == [0, 3200, 6400, 7964]
    assert result["peak_frame"] == 4000
    assert crop_layout(wave[:6400])["tiled_starts"] == [0]
    with pytest.raises(ResearchError):
        crop_layout(wave[:6399])
    wave[0] = np.nan
    with pytest.raises(ResearchError):
        crop_layout(wave)


def test_channel_average_is_preserved_and_resampling_preserves_event_time():
    mono = np.zeros(44100, dtype=np.float32)
    mono[22050] = 1
    stereo = np.column_stack((mono, -0.5 * mono))
    result = resample_mono(stereo, 44100)
    assert len(result) == 16000
    assert np.abs(result).argmax() == 8000
    assert result.dtype == np.float32
    assert np.array_equal(result, resample_mono(stereo.mean(1)[:, None], 44100))
    same = resample_mono(stereo[:16000], 16000)
    assert np.array_equal(same, stereo[:16000].mean(1))


def test_group_means_are_per_clip_then_per_parent_not_per_window():
    scores = np.array([[0, 1], [2, 7], [4, 2], [6, 8]], dtype=float)
    owners = np.array([1, 0, 1, 1])
    clips = grouped_mean(scores, owners, 2)
    assert np.array_equal(clips, [[2, 7], [10 / 3, 11 / 3]])
    parent = grouped_mean(clips, np.zeros(2, dtype=int), 1)
    assert np.allclose(parent, [[8 / 3, 16 / 3]])
    assert not np.allclose(parent, scores.mean(0))
    with pytest.raises(ResearchError, match="tanpa skor"):
        grouped_mean(scores, owners, 3)


def test_clustered_metrics_exclude_iid_interval_and_expose_constant_prediction():
    y = np.arange(36)
    scores = np.zeros((36, 36))
    scores[:, 2] = 1
    result = metrics(y, scores)
    assert result["accuracy"] == pytest.approx(1 / 36)
    assert result["correct"] == 1
    assert result["predicted_class_count"] == 1
    assert result["most_frequent_prediction_count"] == 36
    assert "wilson95" not in result


def test_scoring_freezes_batchnorm_and_uses_passed_source_statistics():
    import torch

    from ketik_audio.fewshot import _score
    from ketik_audio.resnet_fewshot import state_digest

    model = torch.nn.Sequential(
        torch.nn.BatchNorm2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 2)
    ).requires_grad_(False)
    before = state_digest(model.state_dict())
    mel = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    mean, std = torch.tensor(3.0), torch.tensor(2.0)
    scores = _score(model, mel, mean, std, "cpu")
    with torch.inference_mode():
        expected = model(((torch.from_numpy(mel) - mean) / std)[:, None]).numpy()
    assert np.array_equal(scores, expected)
    assert before == state_digest(model.state_dict())
    assert all(not m.training for m in model.modules())
