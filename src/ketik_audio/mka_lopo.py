"""Frozen embeddings and matched linear probes across disjoint MKA folders."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

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
from .fewshot import BEATS_SHA256, ROOT, _beats
from .mka import KEYS, PLATFORMS
from .mka_zeroshot import grouped_mean, metrics
from .mka_zeroshot import signature as parent_signature
from .resnet_fewshot import initialize_model, state_digest

PROTOCOL = ROOT / "research/MKA_LOPO_PROTOCOL.md"
PARENT = ROOT / "runs/mka-zeroshot"
SOURCE = ROOT / "runs/resnet-fewshot"
CHECKPOINT = ROOT / "data/external/beats/BEATs_iter3_plus_AS2M.pt"
CACHE = ROOT / "data/external/mka-lopo-features"
SEEDS = (42, 43, 44)
DRAWS = (101, 102, 103)
KS = (1, 2, 3, 5, 6)
ENCODERS = ("beats", "resnet_s42", "resnet_s43", "resnet_s44")
PARENT_SHA = "5c9027ea875bc1080fa209d87a55e0eb3a2200be01ba95ca7996fa0220529c47"


def signature():
    previous = parent_signature()
    result = read_json(PARENT / "results.json")
    if file_hash(PARENT / "results.json") != PARENT_SHA:
        raise ResearchError("Hasil induk berubah.")
    if result["config"]["input_sha256"] != previous:
        raise ResearchError("Input studi induk berubah.")
    for name, digest in result["artifact_sha256"].items():
        if file_hash(PARENT / name) != digest:
            raise ResearchError(f"Artefak induk berubah: {name}")
    vendor = ROOT / "research/vendor/beats"
    provenance = read_json(vendor / "provenance.json")
    paths = [
        Path(__file__),
        PROTOCOL,
        CHECKPOINT,
        ROOT / "src/ketik_audio/frozen.py",
        ROOT / "src/ketik_audio/devices.py",
        ROOT / "src/ketik_audio/mka_zeroshot.py",
        ROOT / "research/audit_mka_zeroshot.py",
        ROOT / "reports/mka-zeroshot-audit/audit.json",
        ROOT / "data/external/beats/provenance.json",
        vendor / "provenance.json",
        PARENT / "results.json",
        PARENT / "inputs.npz",
        PARENT / "records.json",
    ]
    paths.extend(vendor / name for name in provenance["files"])
    current = {str(path.relative_to(ROOT)): file_hash(path) for path in paths}
    if current[str(CHECKPOINT.relative_to(ROOT))] != BEATS_SHA256:
        raise ResearchError("Checkpoint BEATs berubah.")
    for name, row in provenance["files"].items():
        if current[str((vendor / name).relative_to(ROOT))] != row["sha256"]:
            raise ResearchError("Vendor BEATs berubah.")
    return {**previous, **current}


def verified_records():
    records = read_json(PARENT / "records.json")["records"]
    expected = {(p, key) for p in PLATFORMS for key in KEYS}
    groups = {}
    for index, row in enumerate(records):
        if row["clip_index"] != index or row["label"] != KEYS.index(row["key"]):
            raise ResearchError("Indeks/label potongan tidak sesuai.")
        groups.setdefault((row["platform"], row["key"]), []).append(row)
    if len(records) != 1296 or set(groups) != expected:
        raise ResearchError("Cakupan platform/kelas tidak lengkap.")
    parents = []
    for rows in groups.values():
        identities = {r["parent_group_id"] for r in rows}
        if len(rows) != 6 or len(identities) != 1:
            raise ResearchError("Kelompok harus berisi enam potongan dari satu WAV.")
        parents.append(next(iter(identities)))
    if len(set(parents)) != 216 or len({r["pcm_sha256"] for r in records}) != 1296:
        raise ResearchError("Duplikasi induk/potongan lintas kelompok.")
    return records


def permutations(records, draw):
    rng = np.random.default_rng(draw)
    groups = {}
    for index, row in enumerate(records):
        groups.setdefault(row["parent_group_id"], []).append(index)
    return {group: rng.permutation(indices) for group, indices in groups.items()}


def split_plan(records):
    ranks = {draw: permutations(records, draw) for draw in DRAWS}
    plan = []
    for heldout in PLATFORMS:
        test = np.asarray([i for i, r in enumerate(records) if r["platform"] == heldout])
        test_parents = {records[i]["parent_group_id"] for i in test}
        for k in KS:
            for draw in (None,) if k == 6 else DRAWS:
                selected = []
                if k == 6:
                    selected = [i for i, row in enumerate(records) if row["platform"] != heldout]
                else:
                    for indices in ranks[draw].values():
                        if records[int(indices[0])]["platform"] != heldout:
                            selected.extend(indices[:k].tolist())
                support = np.asarray(sorted(selected), dtype=np.int64)
                support_parents = {records[i]["parent_group_id"] for i in support}
                if (
                    len(support) != 180 * k
                    or len(test) != 216
                    or len(support_parents) != 180
                    or len(test_parents) != 36
                    or support_parents & test_parents
                    or set(support) & set(test)
                ):
                    raise ResearchError("Split/anggaran/identitas induk tidak valid.")
                plan.append(
                    {
                        "id": f"{heldout}-k{k}-d{draw if draw is not None else 'all'}",
                        "heldout": heldout,
                        "k": k,
                        "draw": draw,
                        "support": support.tolist(),
                        "test": test.tolist(),
                        "support_parents": sorted(support_parents),
                        "test_parents": sorted(test_parents),
                    }
                )
    return plan


def permuted_parent_labels(records, support):
    groups = {}
    for offset, index in enumerate(support):
        groups.setdefault(records[index]["parent_group_id"], []).append(offset)
    labels = np.asarray([records[support[indices[0]]]["label"] for indices in groups.values()])
    shuffled = np.random.default_rng(997).permutation(labels)
    result = np.empty(len(support), dtype=np.int64)
    for indices, label in zip(groups.values(), shuffled, strict=True):
        result[indices] = label
    if not np.array_equal(np.bincount(result, minlength=36), np.full(36, 30)):
        raise ResearchError("Kontrol negatif tidak seimbang.")
    return result


def fit_head(features, labels, support, test, fit_labels=None):
    """No learned preprocessing; the only fitted object is the source-folder head."""
    x = normalize(np.asarray(features, dtype=np.float64), norm="l2")
    y = labels[support] if fit_labels is None else fit_labels
    if set(y.tolist()) != set(range(36)):
        raise ResearchError("Kepala harus memiliki seluruh 36 kelas.")
    head = RidgeClassifier(alpha=1.0, solver="cholesky", fit_intercept=True, class_weight=None)
    with threadpool_limits(limits=4):
        head.fit(x[support], y)
        scores = head.decision_function(x[test])
        support_scores = head.decision_function(x[support])
    if not np.array_equal(head.classes_, np.arange(36)):
        raise ResearchError("Urutan kelas kepala berubah.")
    return head, scores, support_scores


def parent_scores(records, indices, scores):
    unique = list(dict.fromkeys(records[int(i)]["parent_group_id"] for i in indices))
    mapping = {parent: index for index, parent in enumerate(unique)}
    owners = np.asarray([mapping[records[int(i)]["parent_group_id"]] for i in indices])
    labels = np.asarray(
        [
            next(
                records[int(i)]["label"] for i in indices if records[int(i)]["parent_group_id"] == p
            )
            for p in unique
        ]
    )
    return labels, grouped_mean(scores, owners, len(unique)), unique


def extract_beats(model, waves, device, progress=False):
    import torch

    values = []
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ResearchError("BEATs belum dibekukan.")
    with torch.inference_mode():
        for offset in range(0, len(waves), 16):
            tokens, mask = model.extract_features(
                torch.from_numpy(waves[offset : offset + 16]).to(device)
            )
            if (
                mask is not None
                or tokens.shape[1:] != (16, 768)
                or not torch.isfinite(tokens).all()
            ):
                raise ResearchError("Fitur BEATs tidak sesuai/finite.")
            values.append(tokens.mean(1).cpu().numpy())
            if progress and (offset % 128 == 0 or offset + 16 >= len(waves)):
                print(f"BEATs frozen {min(offset + 16, len(waves))}/{len(waves)}", flush=True)
    return np.concatenate(values)


def cache_features(output):
    configure_rx6000()
    device, hardware = select_device("gpu")
    import torch

    from .frozen import extract, frozen_encoder

    torch.set_num_threads(4)
    sig = signature()
    fresh_directory(output)
    write_json(output / "lock.json", {"created_at": now(), "input_sha256": sig})
    records = verified_records()
    with np.load(PARENT / "inputs.npz", allow_pickle=False) as old:
        wave, mel = old["peak_wave"], old["peak_mel"]
    if wave.shape != (1296, 6400) or mel.shape != (1296, 64, 64):
        raise ResearchError("Bentuk input crop induk berubah.")
    arrays, encoders = {}, []
    model, _ = _beats(CHECKPOINT, device)
    before = state_digest(model.state_dict())
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    arrays["beats"] = extract_beats(model, wave, device, progress=True)
    controls = np.arange(36, dtype=np.int64) * 25
    with np.load(ROOT / "data/external/harrison-fewshot/segments.npz", allow_pickle=False) as data:
        control_wave = data["zoom_wave"][controls]
    vectors = extract_beats(model, control_wave, device)
    with np.load(
        ROOT / "data/external/harrison-fewshot-features/features.npz", allow_pickle=False
    ) as old:
        expected = old["zoom_beats"][controls]
    if not np.allclose(vectors, expected, atol=1e-3, rtol=1e-3):
        raise ResearchError("Kontrol ekstraksi BEATs berbeda dari cache sumber lama.")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    after = state_digest(model.state_dict())
    if before != after:
        raise ResearchError("State BEATs berubah.")
    encoders.append(
        {
            "encoder": "beats",
            "dimension": 768,
            "parameters": sum(p.numel() for p in model.parameters()),
            "before": before,
            "after": after,
            "extraction_and_control_seconds": elapsed,
            "source_control_max_abs_difference": float(np.abs(vectors - expected).max()),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "hardware": hardware,
        }
    )
    np.savez_compressed(output / "controls.npz", indices=controls, beats=vectors)
    del model
    torch.cuda.empty_cache()
    with np.load(PARENT / "scores.npz", allow_pickle=False) as old:
        for seed in SEEDS:
            folder = SOURCE / f"sources/s{seed}"
            model = (
                initialize_model(
                    seed, torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
                )
                .requires_grad_(False)
                .eval()
            )
            before = state_digest(model.state_dict())
            encoder = frozen_encoder(model)
            with np.load(folder / "normalization.npz", allow_pickle=False) as stats:
                mean, std = stats["mean"], stats["std"]
            started = time.perf_counter()
            arrays[f"resnet_s{seed}"] = extract(encoder, mel, mean, std, "cpu")
            with torch.inference_mode():
                logits = model[-1](torch.from_numpy(arrays[f"resnet_s{seed}"])).numpy()
            expected = old[f"s{seed}_peak_clip"]
            if not np.allclose(logits, expected, atol=1e-3, rtol=1e-3) or not np.array_equal(
                logits.argmax(1), expected.argmax(1)
            ):
                raise ResearchError("Representasi ResNet gagal merekonstruksi head sumber.")
            after = state_digest(model.state_dict())
            if before != after:
                raise ResearchError("State ResNet berubah.")
            encoders.append(
                {
                    "encoder": f"resnet_s{seed}",
                    "dimension": 128,
                    "parameters_with_source_head": sum(p.numel() for p in model.parameters()),
                    "before": before,
                    "after": after,
                    "extraction_and_control_seconds": time.perf_counter() - started,
                    "source_control_max_abs_difference": float(np.abs(logits - expected).max()),
                    "source_predictions_match": True,
                    "device": "cpu",
                }
            )
            print(f"Extracted frozen ResNet source seed {seed}", flush=True)
    np.savez_compressed(output / "features.npz", **arrays)
    result = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "environment": environment(),
        "records": len(records),
        "encoders": encoders,
        "features_sha256": file_hash(output / "features.npz"),
        "artifact_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir())},
    }
    if signature() != sig:
        raise ResearchError("Input/kode berubah selama ekstraksi.")
    write_json(output / "manifest.json", result)
    print(f"Frozen cache complete: {output}", flush=True)
    return result


def load_cache(cache):
    manifest = read_json(cache / "manifest.json")
    if manifest["status"] != "complete" or manifest["input_sha256"] != signature():
        raise ResearchError("Cache belum selesai atau signature berubah.")
    for name, digest in manifest["artifact_sha256"].items():
        if file_hash(cache / name) != digest:
            raise ResearchError("Artefak cache berubah.")
    with np.load(cache / "features.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if set(arrays) != set(ENCODERS):
        raise ResearchError("Representasi tidak lengkap.")
    for key, values in arrays.items():
        expected = 768 if key == "beats" else 128
        if values.shape != (1296, expected) or not np.isfinite(values).all():
            raise ResearchError("Embedding tidak sesuai/finite.")
    return manifest, arrays


def run(cache, output):
    manifest, features = load_cache(cache)
    config = {
        "input_sha256": manifest["input_sha256"],
        "cache_manifest_sha256": file_hash(cache / "manifest.json"),
        "cache_features_sha256": manifest["features_sha256"],
        "encoders": list(ENCODERS),
        "k": list(KS),
        "draws": list(DRAWS),
        "head": {
            "alpha": 1.0,
            "solver": "cholesky",
            "fit_intercept": True,
            "class_weight": None,
            "feature_dtype": "float64",
            "row_norm": "l2",
        },
        "heldout_labels_for_fit": 0,
        "heldout_labels_for_selection": 0,
        "backbone_updates": 0,
        "head_device": "cpu",
    }
    if (output / "results.json").exists():
        result = read_json(output / "results.json")
        if result["config"] != config:
            raise ResearchError("Run selesai memiliki lock berbeda.")
        for name, digest in result["artifact_sha256"].items():
            if file_hash(output / name) != digest:
                raise ResearchError("Artefak run berubah.")
        return result
    records = verified_records()
    plan = split_plan(records)
    fresh_directory(output)
    (output / "conditions").mkdir()
    (output / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    write_json(output / "split-plan.json", {"splits": plan})
    write_json(
        output / "lock.json",
        {
            "created_at": now(),
            "config": config,
            "split_plan_sha256": file_hash(output / "split-plan.json"),
        },
    )
    labels = np.asarray([r["label"] for r in records], dtype=np.int64)
    conditions = []
    started = time.perf_counter()
    for split in plan:
        support, test = (np.asarray(split[key], dtype=np.int64) for key in ("support", "test"))
        for negative in (False, True) if split["k"] == 6 else (False,):
            fit_y = permuted_parent_labels(records, support) if negative else labels[support]
            for encoder in ENCODERS:
                fit_start = time.perf_counter()
                head, scores, fitted = fit_head(features[encoder], labels, support, test, fit_y)
                name = f"{split['id']}-{encoder}{'-permuted' if negative else ''}"
                path = output / "conditions" / f"{name}.npz"
                np.savez_compressed(
                    path,
                    coef=head.coef_,
                    intercept=head.intercept_,
                    classes=head.classes_,
                    support=support,
                    test=test,
                    fit_y=fit_y,
                    test_y=labels[test],
                    scores=scores,
                    support_scores=fitted,
                )
                parent_y, grouped, _ = parent_scores(records, test, scores)
                row = {
                    "id": name,
                    "heldout": split["heldout"],
                    "k": split["k"],
                    "draw": split["draw"],
                    "encoder": encoder,
                    "family": "beats" if encoder == "beats" else "resnet",
                    "source_seed": None if encoder == "beats" else int(encoder[-2:]),
                    "permuted": negative,
                    "support_count": len(support),
                    "support_parents": len(split["support_parents"]),
                    "support_accuracy": float((fitted.argmax(1) == fit_y).mean()),
                    "test": metrics(labels[test], scores),
                    "parent": metrics(parent_y, grouped),
                    "seconds_including_save": time.perf_counter() - fit_start,
                    "artifact": str(path.relative_to(output)),
                    "sha256": file_hash(path),
                }
                conditions.append(row)
        if split["k"] == 6:
            print(
                f"Completed heldout {split['heldout']}: all K/draws and negative controls",
                flush=True,
            )
    if len(conditions) != 336:
        raise ResearchError("Jumlah fit tidak sesuai protokol.")
    result = {
        "created_at": now(),
        "status": "complete",
        "config": config,
        "environment": environment(),
        "conditions": conditions,
        "seconds": time.perf_counter() - started,
        "artifact_sha256": {
            str(p.relative_to(output)): file_hash(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    if signature() != manifest["input_sha256"]:
        raise ResearchError("Input/kode berubah selama fit.")
    write_json(output / "results.json", result)
    print(f"Completed {len(conditions)} fixed linear heads", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cache_parser = sub.add_parser("cache")
    cache_parser.add_argument("--output", type=Path, default=CACHE)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--cache", type=Path, default=CACHE)
    run_parser.add_argument("--output", type=Path, default=ROOT / "runs/mka-lopo")
    args = parser.parse_args()
    try:
        cache_features(args.output) if args.command == "cache" else run(args.cache, args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
