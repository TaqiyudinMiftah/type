import io

import numpy as np
import pytest
import soundfile as sf

from ketik_audio.common import ResearchError
from ketik_audio.mka import (
    audio_details,
    count_budget,
    locate_excerpt,
    locate_multichannel_excerpt,
    parse_member,
)


def wav(samples, rate=16000, format_name="WAV"):
    stream = io.BytesIO()
    sf.write(stream, samples, rate, format=format_name, subtype="PCM_16")
    return stream.getvalue()


def test_pcm_duplicates_ignore_container_but_preserve_rate_and_channels():
    samples = np.arange(100, dtype=float) / 32768
    first = audio_details(wav(samples))
    alternate = audio_details(wav(samples, format_name="WAVEX"))
    assert first["format"] != alternate["format"]
    assert first["pcm_sha256"] == alternate["pcm_sha256"]
    assert first["pcm_sha256"] != audio_details(wav(samples, rate=8000))["pcm_sha256"]
    assert (
        first["pcm_sha256"] != audio_details(wav(np.column_stack((samples, samples))))["pcm_sha256"]
    )


def test_excerpt_location_recovers_offset_gain_and_rejects_unrelated_audio():
    rng = np.random.default_rng(8)
    raw = rng.normal(size=5000)
    clip = -0.4 * raw[173:673] + 0.03
    found = locate_excerpt(raw, clip)
    assert found["matched"]
    assert found["start_frame"] == 173
    assert found["stop_frame"] == 673
    assert found["correlation"] == pytest.approx(-1)
    assert found["affine_gain"] == pytest.approx(-0.4)
    assert found["affine_offset"] == pytest.approx(0.03)
    assert found["normalized_residual_rms"] < 1e-12
    assert not locate_excerpt(raw, rng.normal(size=500))["matched"]
    with pytest.raises(ResearchError, match="konstan"):
        locate_excerpt(raw, np.zeros(500))


def test_clip_budget_deduplicates_and_does_not_count_aggregate_copies():
    rows = []
    for key in "ab":
        for i in range(6):
            rows.append(
                {
                    "scope": "canonical",
                    "platform": "hp",
                    "role": "segment",
                    "label": key,
                    "audio": {"pcm_sha256": f"{key}-{i}", "usable_waveform": True},
                }
            )
    rows += [{**r, "scope": "aggregate", "platform": "All Dataset"} for r in rows]
    budget = count_budget(rows, keys="ab")["hp"]
    assert budget["minimum_unique_supplied_clips_per_key"] == 6
    assert budget["count_only_feasible_K"] == {"1": [1, 2, 3, 5], "5": [1], "10": []}
    rows[5]["audio"] = rows[4]["audio"]
    budget = count_budget(rows, keys="ab")["hp"]
    assert budget["minimum_unique_supplied_clips_per_key"] == 5
    assert budget["count_only_feasible_K"] == {"1": [1, 2, 3], "5": [], "10": []}


def test_stereo_excerpt_uses_same_channel_when_local_variances_change():
    rng = np.random.default_rng(20)
    raw = rng.normal(size=(5000, 2))
    raw[:, 0] *= 50
    raw[173:673, 0] /= 100
    clip = raw[173:673].copy()
    assert np.var(raw, axis=0).argmax() == 0
    assert np.var(clip, axis=0).argmax() == 1
    found = locate_multichannel_excerpt(raw, clip)
    assert found["matched"] and found["all_channels_pcm_equal"]
    assert found["start_frame"] == 173
    assert found["raw_channel"] == found["clip_channel"] == 1


def test_source_groups_come_from_raw_paths_not_excerpt_numbers():
    raw = parse_member("MKA datasets/hp/Raw data/0.wav")
    clip = parse_member("MKA datasets/hp/Sound Segment(wav)/0/01.wav")
    aggregate = parse_member("MKA datasets/All Dataset/Raw Data/0/0hp.wav")
    assert (raw["label"], clip["label"], aggregate["label"]) == ("0", "0", "0")
    assert raw["scope"] == clip["scope"] == "canonical"
    assert aggregate["scope"] == "aggregate"
    with pytest.raises(ResearchError):
        parse_member("../untrusted.wav")
