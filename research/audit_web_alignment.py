"""Run the fixed synthetic controls and retain separate real-pilot reanalyses."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from web_alignment_synthetic import make_case

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, write_json
from ketik_audio.web_alignment import align_package, consensus_timing
from ketik_audio.web_collection import replay_timing

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research/WEB_ALIGNMENT_V2_PROTOCOL.json"


def synthetic_result(case, seed, rate, positives):
    fixture = make_case(case, seed, rate)
    declared = {
        "method": "calibration-affine-energy-v1",
        "first_frame": fixture["first"],
        "sample_rate": rate,
        "anchors": fixture["anchors"],
    }
    old, _ = replay_timing(
        fixture["audio"],
        rate,
        fixture["first"],
        fixture["events"],
        declared,
        fixture["capture_issues"],
    )
    old["issues"] = [i for i in old["issues"] if i != "browser_timing_mismatch"]
    old["status"] = "needs_review" if old["issues"] else "checked"
    new = consensus_timing(old, rate)
    errors = (
        [
            abs(
                new["affine"]["slope"] * row["clock_frame"]
                + new["affine"]["intercept"]
                - row["physical_frame"]
            )
            / rate
            * 1000
            for row in fixture["truth"]
        ]
        if new["affine"]
        else []
    )
    expected = "checked" if case in positives or case == "hidden_middle_bend" else "needs_review"
    ok = new["status"] == expected
    if case in positives:
        ok &= bool(errors) and max(errors) < 10
    elif case == "hidden_middle_bend":
        ok &= bool(errors) and max(errors) > 60
    return {
        "case": case,
        "seed": seed,
        "rate": rate,
        "expected_status": expected,
        "v1_status": old["status"],
        "v2_status": new["status"],
        "v2_issues": new["issues"],
        "max_known_mapping_error_ms": max(errors) if errors else None,
        "max_boundary_validation_residual_ms": new["max_abs_validation_residual_ms"],
        "excluded_training_peaks": new["excluded_training_peaks"],
        "passed_expected_behavior": bool(ok),
    }


def figures(output, real):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    for entry in real:
        source = output / entry["directory"]
        report = json.loads((source / "alignment-v2.json").read_text())
        old = json.loads((source / "review.json").read_text())["timing"]
        if report["affine"] is None:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
        ax = axes[0]
        markers = report["calibration"]
        for split, color, shape, label in [
            ("fit", "#1c7a79", "o", "Penanda fit"),
            ("validation", "#bc6d34", "s", "Penanda validasi terpisah"),
        ]:
            subset = [m for m in markers if m["split"] == split and m["reason"] is None]
            ax.scatter(
                [m["clock_frame"] / entry["rate"] for m in subset],
                [(m["peak_frame"] - m["clock_frame"]) / entry["rate"] * 1000 for m in subset],
                marker=shape,
                color=color,
                label=label,
            )
        usable = [m for m in markers if m["reason"] is None]
        timeline = np.array(
            [min(m["clock_frame"] for m in usable), max(m["clock_frame"] for m in usable)]
        )
        for affine, color, label in [
            (old["affine"], "#907092", "Fit v1"),
            (report["affine"], "#317d4a", "Hipotesis utama v2 (belum lulus)"),
        ]:
            if affine:
                ax.plot(
                    timeline / entry["rate"],
                    ((affine["slope"] - 1) * timeline + affine["intercept"]) / entry["rate"] * 1000,
                    color=color,
                    label=label,
                )
        ax.set(xlabel="Waktu menurut clock audio (detik)", ylabel="Offset puncak audio (ms)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
        ax.set_title(
            "Hanya penanda fit yang memilih hipotesis; sinyal lemah tetap dikeluarkan",
            loc="left",
            fontsize=11,
        )
        ax = axes[1]
        for values, shift, color, label in [
            (old["residuals_ms"], -0.18, "#907092", "v1"),
            (report["residuals_ms"], 0.18, "#317d4a", "v2"),
        ]:
            ax.bar(np.arange(len(values)) + shift, values, width=0.35, color=color, label=label)
        ax.axhline(20, color="#b24d47", ls="--", lw=0.8)
        ax.axhline(-20, color="#b24d47", ls="--", lw=0.8)
        ax.axhline(0, color="#777", lw=0.5)
        ax.set(
            xlabel="Urutan penanda validasi yang memiliki sinyal", ylabel="Residual validasi (ms)"
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.15)
        fig.suptitle(
            f"Alignment {entry['session']} — status tetap {report['status']}",
            fontsize=14,
            fontweight="bold",
        )
        fig.savefig(source / "alignment-comparison.png", dpi=160)
        plt.close(fig)


def run(output, packages, protocol_path=PROTOCOL):
    protocol = json.loads(protocol_path.read_text())
    frozen = {
        str(p.relative_to(ROOT)): file_hash(p)
        for p in [
            protocol_path,
            Path(__file__),
            ROOT / "research/web_alignment_synthetic.py",
            ROOT / "src/ketik_audio/web_alignment.py",
            ROOT / "src/ketik_audio/web_collection.py",
        ]
    }
    fresh_directory(output)
    write_json(output / "lock.json", {"created_at": now(), "source_sha256": frozen})
    spec = protocol["synthetic_evaluation"]
    positives = set(spec["positive_cases"])
    results = [
        synthetic_result(case, seed, rate, positives)
        for rate in spec["rates"]
        for seed in range(spec["seed_range"][0], spec["seed_range"][1] + 1)
        for case in spec["cases"]
    ]
    blind_spot = [
        synthetic_result("hidden_middle_bend", seed, rate, positives)
        for rate in spec["rates"]
        for seed in range(spec["seed_range"][0], spec["seed_range"][1] + 1)
    ]
    write_json(
        output / "synthetic-cases.json", {"cases": results, "boundary_blind_spot": blind_spot}
    )
    development = {d["zip_sha256"]: d["session"] for d in protocol["development_sources"]}
    real = []
    for package in packages:
        digest = file_hash(package)
        session = development.get(digest, digest[:8])
        destination = output / session
        role = "development" if digest in development else "unclassified"
        report = align_package(package, destination, role)
        meta = json.loads((destination / "session.json").read_text())
        real.append(
            {
                "session": session,
                "source_sha256": digest,
                "directory": session,
                "evaluation_role": role,
                "rate": meta["sample_rate"],
                "v1_status": report["original_v1_status"],
                "v2_status": report["status"],
                "issues": report["issues"],
                "excluded_training_peaks": report["excluded_training_peaks"],
                "training_inliers_by_phase": report["training_inliers_by_phase"],
                "validation_markers_by_phase": report["validation_markers_by_phase"],
                "v1_max_residual_ms": report["original_v1_max_residual_ms"],
                "v2_max_residual_ms": report["max_abs_validation_residual_ms"],
                "ambiguous_training_consensus": report["ambiguous_training_consensus"],
                "verified_candidates": sum(report["coverage"]["candidate_counts"].values()),
            }
        )
    figures(output, real)
    good = [r for r in results if r["case"] in positives]
    negative = [r for r in results if r["case"] not in positives]
    summary = {
        "schema": "ketik-web-alignment-experiment/v2",
        "created_at": now(),
        "protocol_sha256": frozen[str(protocol_path.relative_to(ROOT))],
        "synthetic_total": len(results),
        "expected_behavior_passed": sum(r["passed_expected_behavior"] for r in results),
        "positive_cases": len(good),
        "positive_checked": sum(r["v2_status"] == "checked" for r in good),
        "max_known_mapping_error_positive_ms": max(r["max_known_mapping_error_ms"] for r in good),
        "negative_cases": len(negative),
        "negative_erroneously_checked": sum(r["v2_status"] == "checked" for r in negative),
        "case_counts": dict(Counter(r["case"] for r in results)),
        "boundary_blind_spot": {
            "cases": len(blind_spot),
            "observed": all(r["passed_expected_behavior"] for r in blind_spot),
            "max_interior_error_ms": max(r["max_known_mapping_error_ms"] for r in blind_spot),
            "interpretation": (
                "Consistent boundary markers cannot exclude "
                "a hidden nonlinear timing bend between them."
            ),
        },
        "real_pilots": real,
        "classifier_fit_performed": False,
        "gpu_used": False,
        "limitations": [
            "Generated controls verify specified failures, not arbitrary recording conditions.",
            "Both real pilots were inspected during development; neither is independent evidence.",
            "Waveform peaks and browser events do not certify mechanical onset.",
        ],
    }
    for name, digest in frozen.items():
        if file_hash(ROOT / name) != digest:
            raise ResearchError(f"Sumber berubah selama evaluasi: {name}")
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not all(r["passed_expected_behavior"] for r in results + blind_spot):
        raise ResearchError("Ada perilaku kontrol sintetis yang tidak sesuai protokol.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package", type=Path, action="append", default=[])
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    run(args.output, args.package, args.protocol.resolve())
