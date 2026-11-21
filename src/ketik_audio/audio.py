"""Identical audio processing for preparation, evaluation, and prediction."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

from .common import FeatureConfig, ResearchError, read_json


def protocol_intervals(metadata: dict) -> list[tuple[float, float]]:
    rate = metadata.get("sample_rate", 0)
    intervals = metadata.get("ignored_intervals_samples", [])
    if not intervals:
        return []
    if rate <= 0:
        raise ResearchError("Sample rate metadata jeda tidak valid.")
    result = []
    for start, end in intervals:
        if not 0 <= start <= end <= metadata["frames"]:
            raise ResearchError("Batas jeda protokol berada di luar rekaman.")
        result.append((start / rate, end / rate))
    return result


def mask_protocol(
    audio: np.ndarray, intervals: list[tuple[float, float]], config: FeatureConfig
) -> np.ndarray:
    if not intervals:
        return audio
    audio = audio.copy()
    for start, end in intervals:
        audio[round(start * config.sample_rate) : round(end * config.sample_rate)] = 0
    return audio


def without_controls(times: np.ndarray, intervals: list[tuple[float, float]]) -> np.ndarray:
    keep = np.ones(len(times), dtype=bool)
    for start, end in intervals:
        keep &= (times < start) | (times > end)
    return keep


def prediction_audio(path: Path, config: FeatureConfig) -> np.ndarray:
    """Optional protocol sidecar contains control timings, never typed letters."""
    audio = load_audio(path, config)
    sidecar = path.with_name("session.json")
    if path.name == "audio.wav" and sidecar.exists():
        metadata = read_json(sidecar)
        audio = mask_protocol(audio, protocol_intervals(metadata), config)
    return audio


def load_audio(path: Path, config: FeatureConfig) -> np.ndarray:
    try:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as exc:
        raise ResearchError(f"Tidak dapat membaca audio {path}: {exc}") from exc
    if not len(audio) or not np.isfinite(audio).all():
        raise ResearchError(f"Audio kosong atau berisi nilai tidak valid: {path}")
    mono = audio.mean(axis=1)
    if rate != config.sample_rate:
        mono = librosa.resample(mono, orig_sr=rate, target_sr=config.sample_rate)
    return np.asarray(mono, dtype=np.float32)


def crop(audio: np.ndarray, seconds: float, config: FeatureConfig) -> np.ndarray:
    length = round(config.segment_ms * config.sample_rate / 1000)
    start = round(seconds * config.sample_rate) - round(config.pre_ms * config.sample_rate / 1000)
    end = start + length
    result = np.zeros(length, dtype=np.float32)
    left, right = max(0, start), min(len(audio), end)
    if right > left:
        result[left - start : right - start] = audio[left:right]
    return result


def extract(segment: np.ndarray, config: FeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    power = librosa.feature.melspectrogram(
        y=segment,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        center=False,
        power=2.0,
    )
    mel = librosa.power_to_db(power, ref=1.0).astype(np.float32)
    mfcc = librosa.feature.mfcc(S=mel, n_mfcc=config.n_mfcc)
    components = np.concatenate(
        [mfcc, librosa.feature.delta(mfcc), librosa.feature.delta(mfcc, order=2)], axis=0
    )
    summary = np.concatenate([components.mean(axis=1), components.std(axis=1)]).astype(np.float32)
    if not np.isfinite(mel).all() or not np.isfinite(summary).all():
        raise ResearchError("Ekstraksi fitur menghasilkan nilai tidak valid.")
    return mel, summary


def detect(audio: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """Audio-only onset detector. Key markers never enter this function."""
    if len(audio) < config.n_fft or np.max(np.abs(audio)) < 1e-6:
        return np.empty(0, dtype=np.float64)
    flux = librosa.onset.onset_strength(
        y=audio,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
    )
    median = float(np.median(flux))
    deviation = float(np.median(np.abs(flux - median))) * 1.4826
    height = max(1.0, median + config.detector_k * max(deviation, 0.1))
    distance = max(1, round(config.refractory_ms * config.sample_rate / (1000 * config.hop_length)))
    peaks, _ = find_peaks(flux, height=height, distance=distance)
    times = librosa.frames_to_time(peaks, sr=config.sample_rate, hop_length=config.hop_length)
    # Silence/quiet-room rejection uses a robust recording-level noise estimate.
    rms = librosa.feature.rms(y=audio, frame_length=config.n_fft, hop_length=config.hop_length)[0]
    floor = max(float(np.quantile(rms, 0.2)) * 3, 1e-5)
    accepted = []
    for seconds in times:
        center = round(seconds * config.sample_rate)
        window = audio[max(0, center - config.n_fft) : min(len(audio), center + config.n_fft)]
        if len(window) and float(np.sqrt(np.mean(window**2))) > floor:
            accepted.append(float(seconds))
    return np.asarray(accepted, dtype=np.float64)


def features_at(audio: np.ndarray, times: np.ndarray, config: FeatureConfig):
    if not len(times):
        frames = (
            1
            + (round(config.segment_ms * config.sample_rate / 1000) - config.n_fft)
            // config.hop_length
        )
        return (
            np.empty((0, config.n_mels, frames), dtype=np.float32),
            np.empty((0, config.n_mfcc * 6), dtype=np.float32),
        )
    pairs = [extract(crop(audio, float(seconds), config), config) for seconds in times]
    return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
