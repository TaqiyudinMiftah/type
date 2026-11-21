import numpy as np
import pytest

from ketik_audio.common import ResearchError
from ketik_audio.skaid_pilot_features import mfcc_features


def test_mfcc_is_clip_local_with_valid_short_frame_deltas():
    rng = np.random.default_rng(84)
    quiet = rng.normal(0, 1e-5, 1280).astype(np.float32)
    loud = rng.normal(0, 1.0, 1280).astype(np.float32)
    batch = mfcc_features(np.stack((quiet, loud)))
    assert batch.shape == (2, 78) and np.isfinite(batch).all()
    assert np.array_equal(batch[0], mfcc_features(quiet[None])[0])
    assert np.array_equal(batch[1], mfcc_features(loud[None])[0])
    assert not np.allclose(batch[0], batch[1])


def test_mfcc_rejects_wrong_crop_length_or_nonfinite_audio():
    with pytest.raises(ResearchError):
        mfcc_features(np.zeros((1, 6400), np.float32))
    with pytest.raises(ResearchError):
        mfcc_features(np.full((1, 1280), np.nan, np.float32))
