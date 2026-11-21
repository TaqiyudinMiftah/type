"""Source-folder-only alpha selection with a locked, separate outer evaluation."""

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
from .mka import PLATFORMS
from .mka_lopo import (
    CACHE,
    ENCODERS,
    ROOT,
    load_cache,
    parent_scores,
    permuted_parent_labels,
    split_plan,
    verified_records,
)
from .mka_zeroshot import metrics

PROTOCOL = ROOT / "research/MKA_NESTED_PROTOCOL.md"
BASELINE = ROOT / "runs/mka-lopo"
PREVIOUS_AUDIT = ROOT / "reports/mka-lopo-audit"
ALPHAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)
PINS = {
    "runs/mka-lopo/results.json": (
        "dc7120d828390de9f1fc61ec342eb73613e1026366715f4d448ef16e4d062f7e"
    ),
    "runs/mka-lopo/split-plan.json": (
        "10de280d09d2ef4dc0fccbf01d0436529aaeb2ddf690e188192b899f09aa077d"
    ),
    "data/external/mka-lopo-features/manifest.json": (
        "c4e7ee56ccd04ecd6fac5b4dd167551a701fabcd56011b0e981327359394792f"
    ),
    "data/external/mka-lopo-features/features.npz": (
        "a0b14258bfbf84f77d2ff32304fa5b12769e4a681e74289f262f6c17384403cd"
    ),
    "reports/mka-lopo-audit/audit.json": (
        "538c3bc5f513db6e8d48940d3e213a8a9475bff83a7d3303892fe7c14056851b"
    ),
    "reports/mka-lopo-audit/cpu-features.npz": (
        "081ad8e41ab6636f140f8c8e0b0cd4745d4081ba7c8ecbb30a9258b343677f44"
    ),
}


def inputs():
    cache, features = load_cache(CACHE)
    extra = {name: file_hash(ROOT / name) for name in PINS}
    if extra != PINS:
        raise ResearchError("Artefak studi LOPO yang dipatok berubah.")
    baseline = read_json(BASELINE / "results.json")
    audit = read_json(PREVIOUS_AUDIT / "audit.json")
    if (
        baseline["status"] != "complete"
        or audit["status"] != "passed"
        or audit["run_results_sha256"] != PINS["runs/mka-lopo/results.json"]
        or baseline["config"]["input_sha256"] != cache["input_sha256"]
    ):
        raise ResearchError("Baseline/cache/audit tidak sesuai.")
    for folder, record in ((BASELINE, baseline), (PREVIOUS_AUDIT, audit)):
        for name, digest in record["artifact_sha256"].items():
            if file_hash(folder / name) != digest:
                raise ResearchError(f"Artefak lama berubah: {folder / name}")
    for path in (
        Path(__file__),
        PROTOCOL,
        ROOT / "research/audit_mka_lopo.py",
        ROOT / "src/ketik_audio/mka_lopo_analysis.py",
    ):
        extra[str(path.relative_to(ROOT))] = file_hash(path)
    signature = {**cache["input_sha256"], **extra}
    return signature, features, verified_records(), baseline


def nested_plan(records):
    result = []
    for outer in split_plan(records):
        if outer["k"] != 6:
            continue
        support, test = outer["support"], outer["test"]
        inner = []
        for folder in PLATFORMS:
            if folder == outer["heldout"]:
                continue
            train_local = [
                i for i, index in enumerate(support) if records[index]["platform"] != folder
            ]
            validation_local = [
                i for i, index in enumerate(support) if records[index]["platform"] == folder
            ]
            train = [support[i] for i in train_local]
            validation = [support[i] for i in validation_local]
            roles = (train, validation, test)
            groups = [{records[i]["parent_group_id"] for i in role} for role in roles]
            pcm = [{records[i]["pcm_sha256"] for i in role} for role in roles]
            if (
                [len(role) for role in roles] != [864, 216, 216]
                or [len(group) for group in groups] != [144, 36, 36]
                or any(
                    groups[i] & groups[j] or pcm[i] & pcm[j] for i, j in ((0, 1), (0, 2), (1, 2))
                )
                or set(train) | set(validation) != set(support)
            ):
                raise ResearchError("Kebocoran/ukuran inner split tidak sesuai.")
            inner.append(
                {
                    "validation_folder": folder,
                    "train": train,
                    "validation": validation,
                    "train_local": train_local,
                    "validation_local": validation_local,
                }
            )
        result.append(
            {"heldout": outer["heldout"], "support": support, "test": test, "inner": inner}
        )
    return result


def choose_alpha(correct):
    correct = np.asarray(correct)
    if correct.shape != (len(ALPHAS),) or correct.dtype.kind not in "iu" or (correct < 0).any():
        raise ResearchError("Seleksi memerlukan jumlah benar integer untuk seluruh grid.")
    return max(range(len(ALPHAS)), key=lambda index: (int(correct[index]), ALPHAS[index]))


def ridge(x, labels, alpha):
    if (
        not np.isfinite(alpha)
        or alpha <= 0
        or not np.isfinite(x).all()
        or x.ndim != 2
        or len(x) != len(labels)
        or set(labels.tolist()) != set(range(36))
    ):
        raise ResearchError("Masukan ridge/kelas/alpha tidak valid.")
    model = RidgeClassifier(alpha=alpha, solver="cholesky", fit_intercept=True, class_weight=None)
    model.fit(x, labels)
    if not np.array_equal(model.classes_, np.arange(36)):
        raise ResearchError("Urutan kelas ridge berubah.")
    return model


def inner_select(x, labels, folds):
    """Only outer-training rows/labels are accepted; no outer-test arguments exist."""
    x = normalize(np.asarray(x, dtype=np.float64))
    if len(x) != 1080 or len(labels) != 1080 or len(folds) != 5:
        raise ResearchError("Selektor harus menerima 1.080 baris dari lima folder latih.")
    validation_rows = np.concatenate([fold["validation_local"] for fold in folds])
    if not np.array_equal(np.sort(validation_rows), np.arange(1080)):
        raise ResearchError("Setiap baris outer-train harus menjadi validasi tepat sekali.")
    dimension = x.shape[1]
    coefficients = np.empty((len(ALPHAS), 5, 36, dimension), dtype=np.float64)
    intercepts = np.empty((len(ALPHAS), 5, 36), dtype=np.float64)
    scores = np.empty((len(ALPHAS), 5, 216, 36), dtype=np.float64)
    correct = np.empty((len(ALPHAS), 5), dtype=np.int64)
    fitted = np.empty((len(ALPHAS), 5), dtype=np.float64)
    with threadpool_limits(limits=4):
        for column, alpha in enumerate(ALPHAS):
            for fold, split in enumerate(folds):
                train, val = (
                    np.asarray(split[name]) for name in ("train_local", "validation_local")
                )
                if (
                    len(train) != 864
                    or len(val) != 216
                    or set(train) & set(val)
                    or set(train) | set(val) != set(range(1080))
                ):
                    raise ResearchError("Inner fit/validasi bukan partisi outer-train.")
                model = ridge(x[train], labels[train], alpha)
                values = model.decision_function(x[val])
                coefficients[column, fold], intercepts[column, fold] = model.coef_, model.intercept_
                scores[column, fold] = values
                correct[column, fold] = int((values.argmax(1) == labels[val]).sum())
                fitted[column, fold] = float((model.predict(x[train]) == labels[train]).mean())
    selected = choose_alpha(correct.sum(1))
    return {
        "coef": coefficients,
        "intercept": intercepts,
        "validation_scores": scores,
        "correct": correct,
        "support_accuracy": fitted,
        "selected_index": np.array(selected),
    }


def task_id(heldout, encoder, permuted):
    return f"{heldout}-{encoder}-{'permuted' if permuted else 'real'}"


def baseline_lookup(record):
    rows = [r for r in record["conditions"] if r["k"] == 6]
    result = {(r["heldout"], r["encoder"], r["permuted"]): r for r in rows}
    if len(rows) != 48 or len(result) != 48:
        raise ResearchError("Baseline K6 harus memiliki 48 kondisi unik.")
    return result


def select(output):
    signature, features, records, baseline = inputs()
    plan = nested_plan(records)
    labels = np.asarray([r["label"] for r in records])
    previous = baseline_lookup(baseline)
    config = {
        "input_sha256": signature,
        "alphas": list(ALPHAS),
        "outer_folds": 6,
        "inner_folds": 5,
        "k": 6,
        "outer_fit_labels": 1080,
        "inner_fit_labels": 864,
        "inner_validation_labels": 216,
        "outer_test_labels": 216,
        "selection": "maximum total correct; ties largest alpha",
        "outer_labels_for_selection": 0,
        "backbone_updates": 0,
        "gpu_used": False,
    }
    fresh_directory(output)
    (output / "inner").mkdir()
    (output / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    write_json(output / "split-plan.json", {"outer_splits": plan})
    write_json(
        output / "lock.json",
        {
            "created_at": now(),
            "config": config,
            "split_plan_sha256": file_hash(output / "split-plan.json"),
        },
    )
    started = time.perf_counter()
    tasks = []
    for outer in plan:
        support = np.asarray(outer["support"])
        for encoder in ENCODERS:
            for permuted in (False, True):
                identity = task_id(outer["heldout"], encoder, permuted)
                fit_y = permuted_parent_labels(records, support) if permuted else labels[support]
                prior = previous[outer["heldout"], encoder, permuted]
                with np.load(BASELINE / prior["artifact"], allow_pickle=False) as old:
                    if not np.array_equal(old["support"], support) or not np.array_equal(
                        old["fit_y"], fit_y
                    ):
                        raise ResearchError("Dukungan/label berbeda dari baseline alpha1.")
                arrays = inner_select(features[encoder][support], fit_y, outer["inner"])
                path = output / "inner" / f"{identity}.npz"
                np.savez_compressed(
                    path,
                    alphas=np.asarray(ALPHAS),
                    support=support,
                    fit_y=fit_y,
                    validation_indices=np.asarray([s["validation"] for s in outer["inner"]]),
                    train_indices=np.asarray([s["train"] for s in outer["inner"]]),
                    **arrays,
                )
                chosen = int(arrays["selected_index"])
                tasks.append(
                    {
                        "id": identity,
                        "heldout": outer["heldout"],
                        "encoder": encoder,
                        "permuted": permuted,
                        "selected_index": chosen,
                        "selected_alpha": ALPHAS[chosen],
                        "inner_correct": arrays["correct"].tolist(),
                        "inner_accuracy": (arrays["correct"].sum(1) / 1080).tolist(),
                        "inner_support_accuracy": arrays["support_accuracy"].tolist(),
                        "artifact": str(path.relative_to(output)),
                        "sha256": file_hash(path),
                    }
                )
                print(f"Selected {len(tasks)}/48 {identity}: alpha={ALPHAS[chosen]:g}", flush=True)
    selection = {
        "created_at": now(),
        "status": "complete",
        "config": config,
        "tasks": tasks,
        "inner_fits": 2160,
        "seconds": time.perf_counter() - started,
    }
    write_json(output / "selections.json", selection)
    if inputs()[0] != signature:
        raise ResearchError("Input/kode berubah selama seleksi.")
    write_json(
        output / "selections-locked.json",
        {
            "created_at": now(),
            "artifact_sha256": {
                str(p.relative_to(output)): file_hash(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
            "outer_predictions_computed": 0,
        },
    )
    print("All 48 alpha choices locked; no new outer predictions computed", flush=True)
    return selection


def verify_selection(root, signature):
    locked = read_json(root / "selections-locked.json")
    selection = read_json(root / "selections.json")
    config = read_json(root / "lock.json")["config"]
    if (
        selection["status"] != "complete"
        or selection["config"] != config
        or config["input_sha256"] != signature
        or locked["outer_predictions_computed"] != 0
    ):
        raise ResearchError("Lock seleksi berbeda/tidak lengkap.")
    for name, digest in locked["artifact_sha256"].items():
        if file_hash(root / name) != digest:
            raise ResearchError(f"Artefak seleksi berubah: {name}")
    expected = {(p, e, n) for p in PLATFORMS for e in ENCODERS for n in (False, True)}
    actual = {(t["heldout"], t["encoder"], t["permuted"]) for t in selection["tasks"]}
    if len(selection["tasks"]) != 48 or actual != expected:
        raise ResearchError("Seleksi tidak memuat seluruh 48 tugas unik.")
    return selection


def evaluate(root):
    signature, features, records, baseline = inputs()
    selection = verify_selection(root, signature)
    selection_sha = file_hash(root / "selections-locked.json")
    if (root / "results.json").exists():
        result = read_json(root / "results.json")
        if result["selection_lock_sha256"] != selection_sha:
            raise ResearchError("Run selesai memiliki lock seleksi berbeda.")
        for name, digest in result["artifact_sha256"].items():
            if file_hash(root / name) != digest:
                raise ResearchError("Artefak evaluasi berubah.")
        return result
    fresh_directory(root / "conditions")
    write_json(
        root / "evaluation-start.json",
        {"created_at": now(), "selection_lock_sha256": selection_sha},
    )
    plan = {p["heldout"]: p for p in nested_plan(records)}
    previous = baseline_lookup(baseline)
    labels = np.asarray([r["label"] for r in records])
    normalized = {key: normalize(value.astype(np.float64)) for key, value in features.items()}
    conditions = []
    started = time.perf_counter()
    with threadpool_limits(limits=4):
        for task in selection["tasks"]:
            outer = plan[task["heldout"]]
            support, test = (np.asarray(outer[k]) for k in ("support", "test"))
            x = normalized[task["encoder"]]
            with np.load(root / task["artifact"], allow_pickle=False) as saved:
                fit_y = saved["fit_y"]
            model = ridge(x[support], fit_y, task["selected_alpha"])
            scores, fitted = model.decision_function(x[test]), model.decision_function(x[support])
            path = root / "conditions" / f"{task['id']}.npz"
            np.savez_compressed(
                path,
                coef=model.coef_,
                intercept=model.intercept_,
                classes=model.classes_,
                support=support,
                test=test,
                fit_y=fit_y,
                test_y=labels[test],
                scores=scores,
                support_scores=fitted,
            )
            prior = previous[task["heldout"], task["encoder"], task["permuted"]]
            with np.load(BASELINE / prior["artifact"], allow_pickle=False) as old:
                baseline_scores = x[test] @ old["coef"].T + old["intercept"]
                if not np.allclose(
                    baseline_scores, old["scores"], atol=1e-9, rtol=1e-9
                ) or not np.array_equal(baseline_scores.argmax(1), old["scores"].argmax(1)):
                    raise ResearchError("Rekonstruksi prediksi baseline alpha1 berbeda.")
            parent_y, grouped, _ = parent_scores(records, test, scores)
            conditions.append(
                {
                    "id": task["id"],
                    "heldout": task["heldout"],
                    "encoder": task["encoder"],
                    "family": "beats" if task["encoder"] == "beats" else "resnet",
                    "source_seed": None
                    if task["encoder"] == "beats"
                    else int(task["encoder"][-2:]),
                    "permuted": task["permuted"],
                    "k": 6,
                    "draw": None,
                    "selected_alpha": task["selected_alpha"],
                    "inner_selected_accuracy": task["inner_accuracy"][task["selected_index"]],
                    "support_count": 1080,
                    "support_accuracy": float((fitted.argmax(1) == fit_y).mean()),
                    "test": metrics(labels[test], scores),
                    "parent": metrics(parent_y, grouped),
                    "baseline_id": prior["id"],
                    "baseline_test": prior["test"],
                    "baseline_parent": prior["parent"],
                    "difference_accuracy": float((scores.argmax(1) == labels[test]).mean())
                    - prior["test"]["accuracy"],
                    "artifact": str(path.relative_to(root)),
                    "sha256": file_hash(path),
                }
            )
    verify_selection(root, signature)
    if inputs()[0] != signature:
        raise ResearchError("Input berubah selama evaluasi.")
    result = {
        "created_at": now(),
        "status": "complete",
        "config": selection["config"],
        "selection_lock_sha256": selection_sha,
        "conditions": conditions,
        "environment": environment(),
        "outer_refits": 48,
        "seconds": time.perf_counter() - started,
        "artifact_sha256": {
            str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()
        },
    }
    write_json(root / "results.json", result)
    print("Completed all 48 selected outer models; baseline alpha1 preserved", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pick = sub.add_parser("select")
    pick.add_argument("--output", type=Path, default=ROOT / "runs/mka-nested")
    score = sub.add_parser("evaluate")
    score.add_argument("--run", type=Path, default=ROOT / "runs/mka-nested")
    args = parser.parse_args()
    try:
        select(args.output) if args.command == "select" else evaluate(args.run)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
