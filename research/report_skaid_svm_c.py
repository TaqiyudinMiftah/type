"""Report the locked C expansion, paired effects, and source boundary diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_skaid_selection import budget_ranges, comparisons, save
from report_skaid_svm import LABELS, NAMES, csv_file, group, pct

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.skaid_pilot_analysis import FAMILIES, analyse
from ketik_audio.skaid_pilot_common import KS, ROOT, verify
from ketik_audio.skaid_svm_c import OLD_REPORT, artifacts
from ketik_audio.skaid_svm_c import OUTPUT as RUN
from ketik_audio.skaid_svm_c_core import NEW_CS, RECIPES
from ketik_audio.skaid_svm_core import CS, POLICIES

AUDIT = ROOT / "reports/skaid-svm-c-audit"
NUMERIC = ROOT / "reports/skaid-svm-c-numeric"
OUTPUT = ROOT / "reports/skaid-svm-c"
SNAPSHOT = ROOT / "research/results/skaid-svm-c-2026-11-14.json"
COLORS = {"linear": "#326ba7", "rbf": "#167957", "ridge": "#707070"}


def source_profiles(choices):
    rows = []
    for p in POLICIES:
        for family in FAMILIES:
            denominator = 16 * 60 * (3 if family in ("mfcc", "beats") else 9)
            for k, permuted in [*((k, False) for k in KS), (10, True)]:
                folds = [
                    c
                    for c in choices
                    if c["policy"] == p
                    and c["family"] == family
                    and c["k"] == k
                    and c["permuted"] == permuted
                ]
                if len(folds) != 17:
                    raise ResearchError("Profil skor sumber tidak lengkap.")
                for c_value in (*CS, *NEW_CS):
                    indices = [
                        i for i, r in enumerate(RECIPES) if r["kernel"] == p and r["c"] == c_value
                    ]
                    best = [max(c["source_correct_by_recipe"][i] for i in indices) for c in folds]
                    rows.append(
                        {
                            "policy": p,
                            "family": family,
                            "k": k,
                            "permuted": permuted,
                            "c": c_value,
                            "source_max_mean_accuracy": float(np.mean(best) / denominator),
                            "source_max_min_accuracy": min(best) / denominator,
                            "source_max_max_accuracy": max(best) / denominator,
                            "selected_folds": sum(c["recipe"]["c"] == c_value for c in folds),
                        }
                    )
    return rows


def plot_curves(old, expanded, output):
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 7.7), sharex=True, sharey=True, layout="constrained"
    )
    for ax, family in zip(axes.flat, FAMILIES, strict=True):
        ax.plot(
            KS,
            [100 * group(old["ridge"], family, k)["accuracy"]["mean"] for k in KS],
            color=COLORS["ridge"],
            marker="o",
            linewidth=1.5,
            linestyle=":",
            label="Ridge",
        )
        for p in POLICIES:
            for analysis, capped in ((old[p], True), (expanded[p], False)):
                values = [group(analysis, family, k)["accuracy"]["mean"] for k in KS]
                ax.plot(
                    KS,
                    np.array(values) * 100,
                    marker="o",
                    markersize=4,
                    color=COLORS[p],
                    linestyle="--" if capped else "-",
                    alpha=0.55 if capped else 1,
                    label=f"{LABELS[p]} {'≤100' if capped else '≤100.000'}",
                )
        ax.axhline(100 / 12, color="0.75", linestyle=":", linewidth=1)
        ax.set(title=NAMES[family], ylim=(0, 100), ylabel="Akurasi query (%)", xticks=KS)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("K contoh per huruf pada ID target")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    fig.suptitle("SKAID: perluasan C pada split dan budget yang sama\nMean 17 ID, 12 kelas")
    save(fig, output, "kurva-grid-c")


def plot_deltas(paired, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, layout="constrained")
    keys = ("minus_old", "minus_ridge")
    titles = ("Dibanding SVM grid lama", "Dibanding Ridge terpilih")
    for ax, key, title in zip(axes, keys, titles, strict=True):
        for pi, p in enumerate(POLICIES):
            rows = [r["selected_minus_baseline"] for r in paired[f"{p}_{key}"] if r["k"] == 10]
            means = np.array([r["mean"] for r in rows]) * 100
            cis = np.array([r["ci95_id_bootstrap"] for r in rows]).T * 100
            ax.errorbar(
                means,
                np.arange(4) + pi * 0.21 - 0.105,
                xerr=np.stack((means - cis[0], cis[1] - means)),
                fmt="o",
                color=COLORS[p],
                capsize=3,
                label=LABELS[p],
            )
        ax.axvline(0, color="0.6", linestyle="--")
        ax.set(
            title=title,
            xlabel="Delta akurasi (poin persentase)",
            yticks=range(4),
            yticklabels=[NAMES[f] for f in FAMILIES],
        )
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="best")
    axes[0].invert_yaxis()
    fig.suptitle("Efek perluasan C pada K=10\nCI95% bootstrap berpasangan, bersyarat pada 17 ID")
    save(fig, output, "delta-k10")


def plot_source_profiles(profiles, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5), sharey=True, layout="constrained")
    colors = ("#177b58", "#ad6718", "#2d63a5", "#9466a8")
    for ax, policy in zip(axes, POLICIES, strict=True):
        for family, color in zip(FAMILIES, colors, strict=True):
            rows = [
                r
                for r in profiles
                if r["policy"] == policy
                and r["family"] == family
                and r["k"] == 10
                and not r["permuted"]
            ]
            ax.plot(
                [r["c"] for r in rows],
                [100 * r["source_max_mean_accuracy"] for r in rows],
                marker="o",
                markersize=4,
                color=color,
                label=NAMES[family],
            )
        ax.axvline(100, color="0.5", linestyle="--", linewidth=1, label="Batas grid lama")
        ax.set(xscale="log", title=LABELS[policy], xlabel="C (skala log)", ylim=(0, 100))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Akurasi terbaik di ID sumber (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    fig.suptitle("Profil C pada validasi sumber K=10\nSetiap titik memilih normalisasi/g di sumber")
    save(fig, output, "profil-c-sumber-k10")


def count_text(n):
    return f"{n:,}".replace(",", ".")


def delta_text(row):
    delta = row["selected_minus_baseline"]
    lo, hi = delta["ci95_id_bootstrap"]
    return f"{pct(delta['mean'])} [{pct(lo)}; {pct(hi)}]"


def write_report(output, old, expanded, paired, costs, stats, validation, audit, numeric):
    lines = [
        "# SKAID: apakah memperbesar C memperbaiki SVM?",
        "",
        "C maksimum naik dari 100 ke 100.000, sedangkan 17 ID, 12 huruf, crop, "
        "fitur, query dan biaya label tetap. Setiap pilihan hanya memakai 16 ID lain. "
        "Eksperimen eksploratif ini memakai ulang cohort dan query yang sudah diperiksa.",
        "",
        "## Akurasi seluruh K",
        "",
        "| Fitur | K | Ridge | Linear lama | Linear baru | RBF lama | RBF baru |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        for k in KS:
            cols = (old["ridge"], old["linear"], expanded["linear"], old["rbf"], expanded["rbf"])
            values = [pct(group(a, family, k)["accuracy"]["mean"]) + "%" for a in cols]
            lines.append(f"| {NAMES[family]} | {k} | " + " | ".join(values) + " |")
    lines += [
        "",
        "![Kurva](kurva-grid-c.png)",
        "",
        "Mean antar ID setelah rata-rata draw dan model dalam ID, bukan ensemble. "
        "Tidak ada rerata 80% atau 90% pada grid K yang diuji.",
        "",
        "## Delta K10 berpasangan",
        "",
        "| Fitur | Kebijakan | Akurasi baru [CI95%] | Delta vs SVM lama (pp) | "
        "Delta vs Ridge (pp) | ID naik/tetap/turun vs SVM lama |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        for p in POLICIES:
            acc = group(expanded[p], family, 10)["accuracy"]
            d, r = (
                next(x for x in paired[f"{p}_{kind}"] if x["family"] == family and x["k"] == 10)
                for kind in ("minus_old", "minus_ridge")
            )
            lo, hi = acc["ci95_id_bootstrap"]
            lines.append(
                f"| {NAMES[family]} | {LABELS[p]} | {pct(acc['mean'])}% "
                f"[{pct(lo)}; {pct(hi)}] | {delta_text(d)} | {delta_text(r)} | "
                f"{d['ids_improved']}/{d['ids_unchanged']}/{d['ids_worsened']} |"
            )
    lines += [
        "",
        "![Selisih K10](delta-k10.png)",
        "",
        "CI memakai 10.000 bootstrap berpasangan dengan ID sebagai unit. "
        "Fold berbagi sumber, sehingga interval deskriptif dan bersyarat pada cohort/prosedur.",
        "",
        "## Dukungan dan kontrol",
        "",
        "| Fitur | Kebijakan | Fit dukungan K10 normal | Query K10 kontrol |",
        "|---|---|---:|---:|",
    ]
    for f in FAMILIES:
        for p in POLICIES:
            n, c = (group(expanded[p], f, 10, v) for v in (False, True))
            lines.append(
                f"| {NAMES[f]} | {LABELS[p]} | {pct(n['support_fit_label_accuracy'])}% "
                f"| {pct(c['accuracy']['mean'])}% |"
            )
    lines += [
        "",
        "Peluang uniform 8,33%. Kontrol memilih parameter dari tabel label dukungan "
        "yang diacak, terpisah dari tabel normal.",
        "",
        "## Batas grid dan biaya",
        "",
        "| Kebijakan | Pilihan normal berubah | C baru dipilih | C tertinggi lama | "
        "C tertinggi baru |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {LABELS[p]} | {stats[p]['normal']['changed']}/340 | "
            f"{stats[p]['normal']['new_c']} | {stats[p]['normal']['old_cap_before']} | "
            f"{stats[p]['normal']['new_cap']} |"
            for p in POLICIES
        ],
        "",
        "![Profil C](profil-c-sumber-k10.png)",
        "",
        "Grafik profil memakai maksimum dari alternatif normalisasi/g di setiap C pada "
        "16 ID sumber, lalu dirata-ratakan antar fold. Nilai ini adalah data pengembangan "
        "yang dipakai untuk seleksi, bukan akurasi held ID atau estimasi test independen.",
        "",
        "| K | Label target | Label sumber unik per fold |",
        "|---:|---:|---:|",
        *[
            f"| {b['k']} | {b['target_labels_per_trial']} | "
            f"{count_text(b['source_labels_unique_min'])}–"
            f"{count_text(b['source_labels_unique_max'])} |"
            for b in costs
        ],
        "",
        "Tidak ada label sumber unik tambahan dibanding SVM awal. C baru memakai ulang "
        "union dukungan tiga draw dan 960 query pada 16 ID lain. ResNet pralatih tetap "
        "memiliki biaya 720 label Harrison. Budget classifier target tidak mencakup "
        "log, filter, query evaluasi, maupun supervisi pralatih generik.",
        "",
        "## Audit",
        "",
        f"{count_text(validation['new_candidates'])} kandidat baru konvergen. "
        f"{count_text(validation['old_candidates_reused'])} kandidat dan audit lama dipakai ulang. "
        f"{count_text(validation['anchor_pairs_identical'])} pasangan anchor identik. "
        f"{validation['candidates_with_any_iteration_above_old_cap']} kandidat melewati cap "
        f"lama, maksimum {count_text(validation['maximum_binary_iterations'])} iterasi biner.",
        "",
        "| Pemeriksaan | Jumlah |",
        "|---|---:|",
        *[
            f"| {name} | {count_text(audit[key])} |"
            for name, key in (
                (
                    "Kandidat baru di luar toleransi skor",
                    "new_candidate_heads_outside_score_tolerance",
                ),
                (
                    "Perubahan prediksi kernel langsung (baru)",
                    "new_candidate_direct_prediction_changes",
                ),
                ("Pilihan yang berubah secara diagnostik", "diagnostic_direct_choices_changed"),
                ("Kepala final di-replay dari dual", "final_heads_replayed"),
                ("Prediksi dual berubah", "final_dual_prediction_changes"),
                ("Prediksi cache CPU normal berubah", "normal_cpu_prediction_changes"),
                ("Prediksi cache CPU kontrol berubah", "control_cpu_prediction_changes"),
            )
        ],
        "",
        "Pilihan asli tidak berubah karena audit. Skor OvR bukan probabilitas. "
        "Temuan numerik 65 kandidat lama tetap tercatat pada laporan SVM awal; "
        "lihat juga audit kandidat baru untuk jumlah benar dan selisih skor.",
        "",
        f"Diagnosis mereproduksi {numeric['new_candidate_heads_diagnosed']} outlier baru. "
        "Seluruhnya SVM linear, dengan 300 dari fitur acak, tiga BEATs dan tiga ResNet "
        "pralatih. Tidak ada outlier baru yang terpilih untuk ID-nya sendiri. "
        "Kasus selisih terbesar memperlihatkan perbedaan intercept antara solusi "
        "dengan koefisien dual yang praktis sama dan sangat dekat batas C. "
        "Ini diagnosis sensitivitas, tanpa koreksi model.",
        "",
        "Keempat perubahan cache CPU berasal dari ResNet acak RBF. Dua pada "
        "kondisi normal (satu perubahan akurasi) dan dua pada kontrol "
        "(satu perubahan akurasi). MFCC, BEATs dan ResNet pralatih tetap sama prediksinya.",
        "",
        "## Batas interpretasi",
        "",
        "Hasil masih 12 huruf dengan segmentasi berbasis log. Metadata belum memverifikasi "
        "mikrofon, keyboard atau sesi fisik baru. Studi ini tidak menetapkan optimum universal, "
        "SOTA, atau pengenalan teks tanpa acuan. Tidak memperluas rentang C lagi dari hasil luar.",
        "",
        "[Protokol](../../research/SKAID_SVM_C_PROTOCOL.md), "
        "[SVM awal](../skaid-svm/laporan.md), [audit](../skaid-svm-c-audit/audit.json), "
        "[diagnosis numerik](../skaid-svm-c-numeric/summary.json), "
        "[prediksi](../skaid-svm-c-audit/classifications.csv), [per-ID](per-id.csv), "
        "[pilihan](selected-recipes.csv), [profil sumber](source-c-profile.csv), "
        "dan [biaya label](label-budgets.csv).",
        "",
    ]
    (output / "laporan.md").write_text("\n".join(lines), encoding="utf-8")


def run(output, snapshot):
    if snapshot.exists():
        raise ResearchError("Jangan menimpa snapshot C.")
    result, audit = read_json(RUN / "results.json"), read_json(AUDIT / "audit.json")
    numeric = read_json(NUMERIC / "summary.json")
    prior = read_json(OLD_REPORT / "summary.json")
    for base, record in ((RUN, result), (AUDIT, audit), (NUMERIC, numeric), (OLD_REPORT, prior)):
        verify(base, record)
    paths = [
        Path(__file__),
        ROOT / "research/report_skaid_svm.py",
        ROOT / "research/report_skaid_selection.py",
        ROOT / "research/review_skaid_pilot_report.py",
        ROOT / "src/ketik_audio/skaid_pilot_analysis.py",
        RUN / "results.json",
        AUDIT / "audit.json",
        NUMERIC / "summary.json",
        OLD_REPORT / "summary.json",
    ]
    sig = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    fresh_directory(output)
    write_json(output / "lock.json", {"created_at": now(), "input_sha256": sig})
    trials = read_json(RUN / "trials.json")
    old = prior["analyses"]
    expanded = {
        p: analyse([t for t in trials["trials"] if t["policy"] == p], trials["zero_shot"])
        for p in POLICIES
    }
    paired = {
        f"{p}_minus_{name}": comparisons(old[base], expanded[p])
        for p in POLICIES
        for name, base in (("old", p), ("ridge", "ridge"))
    }
    paired["rbf_minus_linear"] = comparisons(expanded["linear"], expanded["rbf"])
    choices = read_json(RUN / "selection.json")["choices"]
    costs = budget_ranges(read_json(RUN / "budgets.json")["budgets"])
    if costs != prior["source_label_budgets"]:
        raise ResearchError("Biaya label berubah.")
    validation = read_json(RUN / "validation.json")
    stats = {}
    for policy in POLICIES:
        stats[policy] = {}
        for permuted, kind in ((False, "normal"), (True, "control")):
            cs = [c for c in choices if c["policy"] == policy and c["permuted"] == permuted]
            stats[policy][kind] = {
                "c_counts": dict(Counter(str(c["recipe"]["c"]) for c in cs)),
                "changed": sum(c["recipe_index"] != c["old_recipe_index"] for c in cs),
                "new_c": sum(c["recipe"]["c"] > 100 for c in cs),
                "old_cap_before": sum(RECIPES[c["old_recipe_index"]]["c"] == 100 for c in cs),
                "new_cap": sum(c["recipe"]["c"] == max(NEW_CS) for c in cs),
            }
    profiles = source_profiles(choices)
    plot_curves(old, expanded, output)
    plot_deltas(paired, output)
    plot_source_profiles(profiles, output)
    csv_file(
        output / "per-id.csv",
        [{"policy": p, **r} for p, a in expanded.items() for r in a["per_id"]],
    )
    csv_file(output / "source-c-profile.csv", profiles)
    csv_file(output / "label-budgets.csv", costs)
    csv_file(
        output / "selected-recipes.csv",
        [
            {
                **{
                    k: c[k]
                    for k in (
                        "id",
                        "policy",
                        "family",
                        "held",
                        "k",
                        "permuted",
                        "old_recipe_index",
                        "recipe_index",
                        "source_labels_unique_total",
                        "old_source_correct",
                        "selected_source_correct",
                    )
                },
                **c["recipe"],
            }
            for c in choices
        ],
    )
    write_report(output, old, expanded, paired, costs, stats, validation, audit, numeric)
    if any(file_hash(ROOT / p) != digest for p, digest in sig.items()):
        raise ResearchError("Masukan laporan berubah.")
    record = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "old_analyses": old,
        "expanded_analyses": expanded,
        "paired_deltas": paired,
        "choice_statistics": stats,
        "source_label_budgets": costs,
        "source_c_profiles": profiles,
        "validation": {
            k: v for k, v in validation.items() if k not in ("input_sha256", "artifact_sha256")
        },
        "audit": {k: v for k, v in audit.items() if k not in ("input_sha256", "artifact_sha256")},
        "numeric": {
            k: v for k, v in numeric.items() if k not in ("input_sha256", "artifact_sha256")
        },
        "artifact_sha256": artifacts(output),
    }
    write_json(output / "summary.json", record)
    write_json(snapshot, {"report_summary_sha256": file_hash(output / "summary.json"), **record})
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    try:
        run(args.output, args.snapshot)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
