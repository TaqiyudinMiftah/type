"""Rebuild every MKA crop from the ZIP and replay all frozen source models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.fewshot import _score
from ketik_audio.mka import KEYS
from ketik_audio.mka_zeroshot import (
    ARCHIVE,
    MANIFEST,
    POLICIES,
    ROOT,
    SEEDS,
    SOURCE,
    array_digest,
    controls,
    measurements,
    mel_cpu,
    signature,
)
from ketik_audio.resnet_fewshot import initialize_model, load_data, state_digest


def rebuild(records, saved):
    """Reconstruct crop coordinates independently of crop_layout/extract_inputs."""
    manifest = read_json(MANIFEST)
    index, tile_index, clamped = 0, 0, 0
    with zipfile.ZipFile(ARCHIVE) as archive:
        for group_index, group in enumerate(manifest["groups"]):
            for clip in group["supplied_unique_clips"]:
                row = records[index]
                for key, value in clip.items():
                    if row[key] != value:
                        raise ResearchError("Provenance potongan berbeda dari manifest.")
                if (
                    row["clip_index"] != index
                    or row["group_index"] != group_index
                    or row["parent_group_id"] != group["parent_group_id"]
                    or row["raw_path"] != group["raw_path"]
                    or row["key"] != group["key"]
                    or row["platform"] != group["platform_folder"]
                    or row["label"] != KEYS.index(group["key"])
                    or saved["y"][index] != row["label"]
                    or saved["clip_group"][index] != group_index
                ):
                    raise ResearchError("Identitas/label/induk cache tidak cocok.")
                blob = archive.read(clip["path"])
                if hashlib.sha256(blob).hexdigest() != clip["sha256"]:
                    raise ResearchError("SHA anggota ZIP berubah.")
                audio, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
                if len(audio) != row["stop_frame"] - row["start_frame"]:
                    raise ResearchError("Durasi native/provenance tidak cocok.")
                divisor = math.gcd(rate, 16000)
                mono = resample_poly(audio.mean(axis=1), 16000 // divisor, rate // divisor)
                if array_digest(mono) != row["mono_sha256"]:
                    raise ResearchError("Waveform mono/resampling berbeda.")
                peak = int(np.argmax(np.abs(mono)))
                start = int(np.clip(peak - 800, 0, len(mono) - 6400))
                if (
                    row["resampled_frames"] != len(mono)
                    or row["peak_frame"] != peak
                    or row["peak_start"] != start
                    or row["peak_clamped"] != (start != peak - 800)
                    or not np.array_equal(saved["peak_wave"][index], mono[start : start + 6400])
                ):
                    raise ResearchError("Crop puncak berbeda.")
                expected = sorted(set([*range(0, len(mono) - 6399, 3200), len(mono) - 6400]))
                if row["tiled_starts"] != expected:
                    raise ResearchError("Grid jendela berbeda.")
                for start in expected:
                    if saved["tiled_owner"][tile_index] != index or not np.array_equal(
                        saved["tiled_wave"][tile_index], mono[start : start + 6400]
                    ):
                        raise ResearchError("Crop tiled/pemilik berbeda.")
                    tile_index += 1
                clamped += row["peak_clamped"]
                index += 1
    if index != 1296 or tile_index != len(saved["tiled_wave"]):
        raise ResearchError("Jumlah crop berbeda.")
    return {"clips_rebuilt": index, "tiled_windows_rebuilt": tile_index, "clamped": clamped}


def run(root, output):
    torch.set_num_threads(4)
    started = time.perf_counter()
    result = read_json(root / "results.json")
    config = read_json(root / "lock.json")["config"]
    if result["status"] != "complete" or result["config"] != config:
        raise ResearchError("Run belum selesai/konfigurasi berbeda.")
    if signature() != config["input_sha256"]:
        raise ResearchError("Input/code/protokol berubah.")
    for name, digest in result["artifact_sha256"].items():
        if file_hash(root / name) != digest:
            raise ResearchError(f"Artefak berubah: {name}")
    fresh_directory(output)
    records = read_json(root / "records.json")["records"]
    with np.load(root / "inputs.npz", allow_pickle=False) as data:
        saved = {name: data[name] for name in data.files}
    reconstruction = rebuild(records, saved)
    for policy in POLICIES:
        actual = mel_cpu(saved[f"{policy}_wave"])
        if not np.array_equal(actual, saved[f"{policy}_mel"]):
            raise ResearchError("Frontend CPU berbeda dari cache.")
    print(f"Rebuilt {len(records)} clips and verified all frontend tensors", flush=True)
    with np.load(root / "scores.npz", allow_pickle=False) as data:
        old_scores = {name: data[name] for name in data.files}
    data, mel = load_data(
        ROOT / "data/external/harrison-fewshot", ROOT / "data/external/harrison-fewshot-features"
    )
    replayed, comparisons, source_checks, classifications = {}, [], [], []
    for seed in SEEDS:
        folder = SOURCE / f"sources/s{seed}"
        weights = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        model = initialize_model(seed, weights).requires_grad_(False).eval()
        before = state_digest(model.state_dict())
        with np.load(folder / "normalization.npz", allow_pickle=False) as z:
            mean, std = (torch.from_numpy(z[k]) for k in ("mean", "std"))
        source_checks.extend(controls(seed, model, mean, std, data, mel, output))
        for policy in POLICIES:
            windows = _score(model, saved[f"{policy}_mel"], mean, std, "cpu")
            expected = old_scores[f"s{seed}_{policy}_window"]
            if not np.array_equal(windows, expected):
                raise ResearchError("Replay logit jendela CPU berbeda.")
            if policy == "peak":
                clips = windows
            else:
                clips = np.stack(
                    [
                        windows[saved["tiled_owner"] == i].astype(np.float64).mean(0)
                        for i in range(len(records))
                    ]
                )
            if not np.array_equal(clips, old_scores[f"s{seed}_{policy}_clip"]):
                raise ResearchError("Agregasi logit per potongan berbeda.")
            replayed[f"s{seed}_{policy}_clip"] = clips
            parents = np.stack(
                [clips[saved["clip_group"] == i].astype(np.float64).mean(0) for i in range(216)]
            )
            for unit, rows, values in (
                ("clip", records, clips),
                ("parent", records[::6], parents),
            ):
                for row, predicted in zip(rows, values.argmax(1), strict=True):
                    classifications.append(
                        {
                            "seed": seed,
                            "policy": policy,
                            "unit": unit,
                            "platform": row["platform"],
                            "parent_group_id": row["parent_group_id"],
                            "clip_path": row["path"] if unit == "clip" else "",
                            "true_key": row["key"],
                            "predicted_key": KEYS[int(predicted)],
                            "correct": int(predicted == row["label"]),
                        }
                    )
            comparisons.append({"seed": seed, "policy": policy, "maximum_logit_difference": 0.0})
        if before != state_digest(model.state_dict()):
            raise ResearchError("State berubah pada audit.")
        state = next(r for r in result["state_checks"] if r["seed"] == seed)
        if state["before"] != before or state["after"] != before:
            raise ResearchError("Digest state run tidak cocok.")
        print(f"Audited both policies and source controls for seed {seed}", flush=True)
    summary = measurements(records, replayed)
    if summary != {name: result[name] for name in ("conditions", "groups")}:
        raise ResearchError("Metrik/summary berbeda.")
    if (
        source_checks != result["source_controls"]
        or reconstruction["clamped"] != result["peak_clamped"]
    ):
        raise ResearchError("Kontrol/crop count berbeda.")
    with (output / "classifications.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(classifications[0]))
        writer.writeheader()
        writer.writerows(classifications)
    audit = {
        "created_at": now(),
        "status": "passed",
        "code_sha256": file_hash(Path(__file__)),
        "run_results_sha256": file_hash(root / "results.json"),
        "input_lock_sha256": file_hash(root / "lock.json"),
        **reconstruction,
        "model_replays": comparisons,
        "source_controls": source_checks,
        "classification_rows": len(classifications),
        "metrics_match": True,
        "all_weights_and_buffers_unchanged": True,
        "gpu_used": False,
        "seconds": time.perf_counter() - started,
        "artifact_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir())},
    }
    if signature() != config["input_sha256"]:
        raise ResearchError("Input berubah selama audit.")
    write_json(output / "audit.json", audit)
    print("Full zero-shot replay passed", flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/mka-zeroshot")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/mka-zeroshot-audit")
    args = parser.parse_args()
    try:
        run(args.run, args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
