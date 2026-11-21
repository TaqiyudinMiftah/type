"""Frozen acoustic features for the locked, short-crop SKAID pilot."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .common import (
    ResearchError,
    environment,
    file_hash,
    fresh_directory,
    now,
    read_json,
    write_json,
)
from .devices import configure_rx6000, select_device
from .skaid_pilot_common import (
    CACHE,
    CHECKPOINT,
    DATA,
    ENCODERS,
    PROTOCOL,
    ROOT,
    SEEDS,
    SOURCE,
    source_hashes,
    verify,
)


def signature():
    verify(DATA, read_json(DATA / "manifest.json"))
    paths = [
        Path(__file__),
        PROTOCOL,
        ROOT / "src/ketik_audio/skaid_pilot_common.py",
        DATA / "manifest.json",
        DATA / "segments.npz",
        DATA / "records.json",
        DATA / "plan.json",
        ROOT / "src/ketik_audio/mka_lopo.py",
    ]
    return {**source_hashes(), **{str(p.relative_to(ROOT)): file_hash(p) for p in paths}}


def mfcc_features(waves):
    import librosa

    if waves.ndim != 2 or waves.shape[1] != 1280 or not np.isfinite(waves).all():
        raise ResearchError("MFCC memerlukan matriks80ms/16kHz yang finite.")
    result = []
    for wave in waves:
        mel = librosa.feature.melspectrogram(
            y=wave,
            sr=16000,
            n_fft=256,
            win_length=256,
            hop_length=80,
            center=False,
            window="hann",
            power=2,
            n_mels=32,
            fmin=20,
            fmax=7600,
            htk=False,
            norm="slaney",
        )
        log = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=80)
        cepstra = librosa.feature.mfcc(S=log, n_mfcc=13, dct_type=2, norm="ortho", lifter=0)
        if cepstra.shape != (13, 13):
            raise ResearchError("Frame MFCC berbeda dari protokol.")
        channels = np.concatenate(
            [
                cepstra,
                librosa.feature.delta(cepstra, order=1, width=9, mode="nearest"),
                librosa.feature.delta(cepstra, order=2, width=9, mode="nearest"),
            ]
        )
        result.append(np.concatenate((channels.mean(axis=1), channels.std(axis=1))))
    features = np.asarray(result, dtype=np.float32)
    if features.shape != (len(waves), 78) or not np.isfinite(features).all():
        raise ResearchError("MFCC78 tidak sesuai/finite.")
    return features


def cache(output):
    configure_rx6000()
    device, hardware = select_device("gpu")
    import torch

    from .fewshot import _beats, frontend
    from .frozen import extract, frozen_encoder
    from .mka_lopo import extract_beats
    from .resnet_fewshot import cpu_state, initialize_model, state_digest

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sig = signature()
    fresh_directory(output)
    write_json(
        output / "lock.json",
        {
            "created_at": now(),
            "input_sha256": sig,
            "environment": environment(),
            "hardware": hardware,
        },
    )
    (output / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    with np.load(DATA / "segments.npz", allow_pickle=False) as z:
        wave80, wave400 = z["wave80"], z["wave400"]
    arrays, encoders, controls = {}, [], {}
    started = time.perf_counter()
    arrays["mfcc"] = mfcc_features(wave80)
    encoders.append(
        {
            "encoder": "mfcc",
            "dimension": 78,
            "device": "cpu",
            "seconds": time.perf_counter() - started,
        }
    )
    images = []
    with torch.inference_mode():
        for offset in range(0, len(wave400), 64):
            images.append(
                frontend(torch.from_numpy(wave400[offset : offset + 64]).to(device)).cpu().numpy()
            )
    arrays["mel64"] = np.concatenate(images)
    del images
    model, _ = _beats(CHECKPOINT, device)
    before = state_digest(model.state_dict())
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    arrays["beats"] = extract_beats(model, wave400, device, progress=True)
    with np.load(ROOT / "data/external/harrison-fewshot/segments.npz", allow_pickle=False) as old:
        controls["beats_indices"] = np.arange(36, dtype=np.int64) * 25
        source_waves = old["zoom_wave"][controls["beats_indices"]]
    actual = extract_beats(model, source_waves, device)
    with np.load(
        ROOT / "data/external/harrison-fewshot-features/features.npz", allow_pickle=False
    ) as old:
        expected = old["zoom_beats"][controls["beats_indices"]]
    if not np.allclose(actual, expected, atol=1e-3, rtol=1e-3):
        raise ResearchError("BEATs tidak cocok dengan kontrol sumber.")
    controls["beats"] = actual
    torch.cuda.synchronize()
    after = state_digest(model.state_dict())
    if before != after:
        raise ResearchError("State BEATs berubah.")
    encoders.append(
        {
            "encoder": "beats",
            "dimension": 768,
            "device": device,
            "hardware": hardware,
            "parameters": sum(p.numel() for p in model.parameters()),
            "before": before,
            "after": after,
            "seconds": time.perf_counter() - started,
            "source_control_max_abs_error": float(np.max(np.abs(actual - expected))),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
    )
    del model
    torch.cuda.empty_cache()
    for kind in ("pretrained", "random"):
        for seed in SEEDS:
            folder = SOURCE / f"sources/s{seed}"
            state = (
                torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
                if kind == "pretrained"
                else None
            )
            model = initialize_model(seed, state).requires_grad_(False).eval().to(device)
            name = f"{kind}_s{seed}"
            if kind == "random":
                torch.save(cpu_state(model), output / f"{name}.pt")
            before = state_digest(model.state_dict())
            encoder = frozen_encoder(model)
            with np.load(folder / "normalization.npz", allow_pickle=False) as stats:
                mean, std = stats["mean"], stats["std"]
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            arrays[name] = extract(encoder, arrays["mel64"], mean, std, device)
            control_error = None
            if kind == "pretrained":
                with torch.inference_mode():
                    arrays[f"source_logits_s{seed}"] = (
                        model[-1](torch.from_numpy(arrays[name]).to(device)).cpu().numpy()
                    )
                with np.load(folder / "scores.npz", allow_pickle=False) as old:
                    indices, expected = old["indices"], old["scores"]
                with np.load(
                    ROOT / "data/external/harrison-fewshot-features/features.npz",
                    allow_pickle=False,
                ) as old:
                    source_images = old["zoom_mel"][indices]
                vectors = extract(encoder, source_images, mean, std, device)
                with torch.inference_mode():
                    actual = model[-1](torch.from_numpy(vectors).to(device)).cpu().numpy()
                if not np.allclose(actual, expected, atol=1e-3, rtol=1e-3) or not np.array_equal(
                    actual.argmax(1), expected.argmax(1)
                ):
                    raise ResearchError("ResNet tidak cocok dengan logits sumber.")
                controls[f"resnet_s{seed}"] = actual
                control_error = float(np.max(np.abs(actual - expected)))
            torch.cuda.synchronize()
            after = state_digest(model.state_dict())
            if before != after:
                raise ResearchError("State ResNet berubah saat ekstraksi target.")
            encoders.append(
                {
                    "encoder": name,
                    "dimension": 128,
                    "device": device,
                    "before": before,
                    "after": after,
                    "seconds": time.perf_counter() - started,
                    "source_control_max_abs_error": control_error,
                    "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
            )
            print(f"Extracted {name}: {len(wave400)} frozen embeddings", flush=True)
            del encoder, model
            torch.cuda.empty_cache()
    if any(
        arrays[k].shape[0] != len(wave400) or not np.isfinite(arrays[k]).all() for k in ENCODERS
    ):
        raise ResearchError("Fitur encoder tidak sesuai/finite.")
    np.savez_compressed(output / "features.npz", **arrays)
    np.savez_compressed(output / "controls.npz", **controls)
    if signature() != sig:
        raise ResearchError("Masukan/kode berubah selama cache.")
    result = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "clips": len(wave400),
        "encoders": encoders,
        "no_target_statistics_fit": True,
        "all_neural_states_unchanged": True,
        "hardware": hardware,
        "environment": environment(),
    }
    result["artifact_sha256"] = {
        p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()
    }
    write_json(output / "manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CACHE)
    args = parser.parse_args()
    try:
        cache(args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
