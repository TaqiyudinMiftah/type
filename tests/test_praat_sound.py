import numpy as np
import pytest

from ketik_audio.common import ResearchError
from ketik_audio.praat_sound import read_praat_sound

HEADER = b'File type = "ooTextFile"\nObject class = "Sound 2"\nnx = 2\nny = 2\ndx = 0.0000625\n'
SAMPLES = b"z [1] [1] = 0.25\nz [1] [2] = -0.5\nz [2] [1] = 0.125\nz [2] [2] = -0.25\n"


def test_praat_sound_preserves_channel_order_and_rate():
    audio, rate = read_praat_sound(HEADER + SAMPLES)
    assert rate == 16000
    np.testing.assert_array_equal(audio, [[0.25, 0.125], [-0.5, -0.25]])
    assert audio.dtype == np.float64


@pytest.mark.parametrize(
    "samples", [SAMPLES.splitlines(keepends=True)[0], SAMPLES + b"z [1] [1] = 0\n"]
)
def test_praat_sound_rejects_missing_or_duplicate_samples(samples):
    with pytest.raises(ResearchError):
        read_praat_sound(HEADER + samples)
