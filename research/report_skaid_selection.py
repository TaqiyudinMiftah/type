"""Report cross-ID recipe selection, paired gains, source label costs and numerical checks."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from review_skaid_pilot_report import readable_text

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.skaid_pilot_analysis import FAMILIES, analyse, estimate
from ketik_audio.skaid_pilot_common import KS, ROOT, verify
from ketik_audio.skaid_pilot_common import RUN as PILOT
from ketik_audio.skaid_selection import OUTPUT as RUN
from ketik_audio.skaid_selection_core import ALPHAS, RECIPES

AUDIT = ROOT / "reports/skaid-selection-audit"
NUMERIC = ROOT / "reports/skaid-selection-numeric"
LABELS = {
    "mfcc": "MFCC",
    "beats": "BEATs beku",
    "pretrained": "ResNet pralatih",
    "random": "ResNet acak",
}
COLORS = {"mfcc": "#177b58", "beats": "#b97818", "pretrained": "#2d63a5", "random": "#9466a8"}


def percent(value):
    return f"{100 * value:.2f}".replace(".", ",")


def ci_text(value):
    lo, hi = value["ci95_id_bootstrap"]
    return f"{percent(lo)}–{percent(hi)}"


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def comparisons(baseline, selected):
    bootstrap = np.random.default_rng(7471).integers(0, 17, size=(10000, 17))
    result = []
    for family in FAMILIES:
        for k in KS:

            def values(analysis, selected_family=family, selected_k=k):
                return np.array(
                    [
                        next(
                            r["accuracy"]
                            for r in analysis["per_id"]
                            if r["family"] == selected_family
                            and r["k"] == selected_k
                            and not r["permuted"]
                            and r["subject"] == i
                        )
                        for i in range(17)
                    ]
                )

            delta = values(selected) - values(baseline)
            result.append(
                {
                    "family": family,
                    "k": k,
                    "selected_minus_baseline": estimate(delta, bootstrap),
                    "ids_improved": int((delta > 1e-10).sum()),
                    "ids_unchanged": int((np.abs(delta) <= 1e-10).sum()),
                    "ids_worsened": int((delta < -1e-10).sum()),
                }
            )
    return result


def budget_ranges(budgets):
    result = []
    for k in KS:
        rows = [r for r in budgets if r["k"] == k]
        totals = [r["source_labels_unique_total"] for r in rows]
        support = [r["source_support_labels_unique"] for r in rows]
        result.append(
            {
                "k": k,
                "target_labels_per_trial": 12 * k,
                "source_labels_unique_min": min(totals),
                "source_labels_unique_max": max(totals),
                "source_labels_unique_mean": float(np.mean(totals)),
                "source_support_unique_min": min(support),
                "source_support_unique_max": max(support),
                "source_query_unique": 960,
                "source_ids_per_fold": 16,
            }
        )
    return result


def plot_curves(baseline, selected, output):
    old = {(r["family"], r["k"]): r["accuracy"] for r in baseline["groups"] if not r["permuted"]}
    new = {(r["family"], r["k"]): r["accuracy"] for r in selected["groups"] if not r["permuted"]}
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 7.5), sharex=True, sharey=True, layout="constrained"
    )
    for ax, family in zip(axes.flat, FAMILIES, strict=True):
        rows = [new[family, k] for k in KS]
        values = np.array([r["mean"] for r in rows]) * 100
        limits = np.array([r["ci95_id_bootstrap"] for r in rows]).T * 100
        ax.plot(
            KS,
            [100 * old[family, k]["mean"] for k in KS],
            color="0.5",
            linestyle="--",
            marker="o",
            label="Baseline alpha1/L2",
        )
        ax.plot(KS, values, color=COLORS[family], marker="o", label="Resep dari 16 ID sumber")
        ax.fill_between(KS, limits[0], limits[1], color=COLORS[family], alpha=0.14)
        ax.axhline(100 / 12, color="0.75", linestyle=":", linewidth=1)
        ax.set(
            title=LABELS[family],
            xlabel="K label per huruf pada ID yang ditahan",
            ylabel="Akurasi query (%)",
            ylim=(0, 100),
        )
        ax.set_xticks(KS)
        ax.grid(axis="y", alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "SKAID: pemilihan resep antar-ID memperbaiki transfer\n"
        "Pita: CI95% bersyarat 17 ID; crop 80 ms dan timestamp acuan",
        fontsize=13,
    )
    save(fig, output, "kurva-baseline-vs-seleksi")


def plot_costs(budgets, output):
    fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
    ks = [b["k"] for b in budgets]
    lo, hi = (
        [b["source_labels_unique_min"] for b in budgets],
        [b["source_labels_unique_max"] for b in budgets],
    )
    ax.fill_between(ks, lo, hi, color="#b97818", alpha=0.25)
    ax.plot(
        ks,
        [b["source_labels_unique_mean"] for b in budgets],
        color="#b97818",
        marker="o",
        label="Label sumber unik untuk memilih resep",
    )
    ax.plot(
        ks,
        [b["target_labels_per_trial"] for b in budgets],
        color="#177b58",
        marker="o",
        label="Label dukungan ID target",
    )
    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set(
        xlabel="K label per huruf pada target",
        ylabel="Jumlah label unik (skala log)",
        title="Biaya pemilihan resep berada di luar budget target",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, output, "biaya-label-sumber")


def plot_choices(choices, output):
    rows, names = [], []
    for family in FAMILIES:
        for k in KS:
            selected = [
                c for c in choices if c["family"] == family and c["k"] == k and not c["permuted"]
            ]
            rows.append(np.bincount([c["recipe_index"] for c in selected], minlength=len(RECIPES)))
            names.append(f"{LABELS[family]} /K{k}")
    values = np.asarray(rows)
    fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
    image = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0, vmax=17)
    for i in range(len(rows)):
        for j in range(len(RECIPES)):
            if values[i, j]:
                ax.text(
                    j,
                    i,
                    str(values[i, j]),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if values[i, j] > 10 else "black",
                )
    ax.axvline(8.5, color="0.4", linewidth=1)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xticks(range(18), [f"{r['alpha']:g}" for r in RECIPES], rotation=60, fontsize=8)
    ax.set(
        xlabel="Alpha: kiri row L2, kanan support z-score lalu L2",
        title="Jumlah fold memilih resep; 17 ID per baris, label normal",
    )
    fig.colorbar(image, ax=ax, label="Jumlah ID")
    save(fig, output, "pilihan-resep")


def numeric_summary(numeric, checks, trials):
    lookup = {r["id"]: r for r in trials}
    enriched = []
    for row in numeric["prediction_changes"]:
        trial = next(
            t
            for t in trials
            if t["encoder"] == row["encoder"] and t["array_index"] == row["array_index"]
        )
        enriched.append(
            {**row, "permuted": trial["permuted"], "k": trial["k"], "draw": trial["draw"]}
        )
    changed_accuracy = [
        r for r in checks if r["primary"]["accuracy"] != r["cpu_features"]["accuracy"]
    ]
    return {
        "prediction_changes": enriched,
        "normal_prediction_changes": sum(not r["permuted"] for r in enriched),
        "control_prediction_changes": sum(r["permuted"] for r in enriched),
        "normal_accuracy_changes": sum(
            not lookup[r["trial_id"]]["permuted"] for r in changed_accuracy
        ),
        "control_accuracy_changes": sum(
            lookup[r["trial_id"]]["permuted"] for r in changed_accuracy
        ),
    }


def run(output, snapshot):
    result = read_json(RUN / "results.json")
    audit = read_json(AUDIT / "audit.json")
    numeric = read_json(NUMERIC / "summary.json")
    for base, record in (
        (RUN, result),
        (AUDIT, audit),
        (NUMERIC, numeric),
        (PILOT, read_json(PILOT / "results.json")),
    ):
        verify(base, record)
    fresh_directory(output)
    paths = [
        Path(__file__),
        ROOT / "research/review_skaid_pilot_report.py",
        ROOT / "src/ketik_audio/skaid_pilot_analysis.py",
        RUN / "results.json",
        RUN / "selection.json",
        RUN / "trials.json",
        RUN / "budgets.json",
        PILOT / "results.json",
        AUDIT / "audit.json",
        NUMERIC / "summary.json",
    ]
    sig = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    write_json(output / "lock.json", {"created_at": now(), "input_sha256": sig})
    trial_data = read_json(RUN / "trials.json")
    old = read_json(PILOT / "trials.json")
    baseline = analyse(old["trials"], old["zero_shot"])
    selected = analyse(trial_data["trials"], trial_data["zero_shot"])
    delta = comparisons(baseline, selected)
    budgets = budget_ranges(read_json(RUN / "budgets.json")["budgets"])
    choices = read_json(RUN / "selection.json")["choices"]
    numerical = numeric_summary(
        numeric, read_json(AUDIT / "final-checks.json")["trials"], trial_data["trials"]
    )
    normal = [c for c in choices if not c["permuted"]]
    choice_stats = {
        "normalizers": dict(Counter(c["recipe"]["normalizer"] for c in normal)),
        "alphas": dict(Counter(str(c["recipe"]["alpha"]) for c in normal)),
        "lower_alpha_boundary_choices": sum(c["recipe"]["alpha"] == ALPHAS[0] for c in normal),
        "upper_alpha_boundary_choices": sum(c["recipe"]["alpha"] == ALPHAS[-1] for c in normal),
    }
    plot_curves(baseline, selected, output)
    plot_costs(budgets, output)
    plot_choices(choices, output)
    for name, rows in (("per-id", selected["per_id"]), ("label-budgets", budgets)):
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    choice_rows = [
        {
            "choice_id": c["id"],
            "held": c["held"],
            "family": c["family"],
            "k": c["k"],
            "permuted": c["permuted"],
            "normalizer": c["recipe"]["normalizer"],
            "alpha": c["recipe"]["alpha"],
            "source_ids": ";".join(map(str, c["source_ids"])),
            "source_correct": c["selected_source_correct"],
            "source_labels_unique_total": c["source_labels_unique_total"],
        }
        for c in choices
    ]
    with (output / "selected-recipes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(choice_rows[0]))
        writer.writeheader()
        writer.writerows(choice_rows)
    groups = {(r["family"], r["k"], r["permuted"]): r for r in selected["groups"]}
    old_groups = {(r["family"], r["k"], r["permuted"]): r for r in baseline["groups"]}
    lines = [
        "# SKAID: pemilihan resep antar-ID",
        "",
        "Tanggal lokal: 9 November 2026.",
        "",
        "**Pada K=10, mean MFCC meningkat dari 52,29% menjadi 75,78%; "
        "BEATs 26,31%→69,80%; ResNet pralatih 43,12%→69,96%.** "
        "Resep dipilih dari 16 ID lain, lalu diterapkan pada dukungan "
        "ID yang ditahan. Kenaikan ini menggunakan label sumber "
        "tambahan, sehingga tidak dianggap perbaikan gratis pada "
        "budget total. Rerata 80%/90% belum tercapai pada grid K1–10.",
        "",
        "## Apa yang dikendalikan",
        "",
        "[Protokol](../../research/SKAID_SELECTION_PROTOCOL.md) ditulis "
        "setelah pilot baseline dibaca dan sebelum kandidat baru "
        "dihitung. Data, crop80 ms,12 huruf,17 ID, tiga draw dan delapan "
        "encoder tetap. Semua pekerjaan memakai CPU pada cache yang "
        "sudah dipatok; tidak ada update backbone atau ekstraksi neural baru.",
        "",
        "Ada18 resep: row L2 atau support z-score lalu L2, masing-masing "
        "pada sembilan alpha1e-6 hingga100. Mean/population-SD scaler "
        "hanya dihitung dari dukungan percobaan, bukan query. Satu "
        "resep dipilih per keluarga/K/held ID dan dipakai bersama "
        "untuk seluruh seed serta draw. Seri memilih alpha terbesar, "
        "lalu row L2. Semua408 pilihan (340 normal +68 kontrol) "
        "dikunci sebelum refit luar.",
        "",
        "Setiap ID berperan sebagai validasi sumber bagi16 fold lain. "
        "Karena itu kandidat pada semua query memang dihitung. "
        "Untuk model ID h, baris h dikeluarkan sebelum fungsi seleksi "
        "menerima kriteria; tidak memilih resep terbaik dari query h. "
        "Ini validasi silang resep pada kohort yang sudah diperiksa, "
        "bukan test baru yang seluruhnya belum pernah dilihat.",
        "",
        "## Seluruh kurva",
        "",
        "Mean antar17 ID setelah rerata draw/model di dalam ID. "
        "Keluarga ResNet memakai3 model×3 draw; MFCC/BEATs3 draw. "
        "Query tetap60 per ID, seimbang12 kelas; peluang uniform8,33%.",
        "",
        "| K / label target | MFCC | BEATs | ResNet pralatih | ResNet acak |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in KS:
        lines.append(
            f"| {k} / {12 * k} | "
            + " | ".join(percent(groups[f, k, False]["accuracy"]["mean"]) + "%" for f in FAMILIES)
            + " |"
        )
    lines += [
        "",
        "![Kurva sebelum/sesudah](kurva-baseline-vs-seleksi.png)",
        "",
        "| Fitur, K10 | Baseline | Terpilih | CI95% antar-ID | "
        "Dukungan fit-label | Kontrol query |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for f in FAMILIES:
        g = groups[f, 10, False]
        lines.append(
            f"| {LABELS[f]} | {percent(old_groups[f, 10, False]['accuracy']['mean'])}% | "
            f"{percent(g['accuracy']['mean'])}% | {ci_text(g['accuracy'])}% | "
            f"{percent(g['support_fit_label_accuracy'])}% | "
            f"{percent(groups[f, 10, True]['accuracy']['mean'])}% |"
        )
    lines += [
        "",
        "Kenaikan mean terjadi pada seluruh17 ID untuk setiap keluarga "
        "pada K10. Semua delta berpasangan dan CI tersimpan di JSON. "
        "Dukungan neural pralatih/BEATs kini hampir100%, sementara "
        "query sekitar70%. Ini menunjukkan bahwa fit dukungan saja "
        "belum menjamin transfer ke rekaman tugas berikutnya.",
        "",
        "MFCC K10:75,78%, CI95%72,16–79,25%. Tidak ada K pada grid "
        "ini yang mencapai mean80%/90%. Top3 MFCC91,27% berbeda "
        "dari akurasi top1; tidak digunakan untuk menetapkan jumlah "
        "label minimum. Hasil tetap bersyarat pada12 kelas dan "
        "timestamp/filter acuan.",
        "",
        "CI memakai10.000 bootstrap ID, seed7471. Fold berbagi data "
        "sumber dan dataset sudah diperiksa; interval bersifat "
        "deskriptif/bersyarat pada17 ID dan kebijakan yang dipilih, "
        "bukan bukti generalisasi populasi mikrofon/sesi baru.",
        "",
        "## Biaya label sumber",
        "",
        "Untuk setiap held ID, validasi sumber memakai16 ID lain. "
        "Hitungan adalah union dukungan tiga draw ditambah960 query "
        "sumber unik. Seed/encoder/resep tidak mengalikan label "
        "karena menggunakan ulang contoh yang sama.",
        "",
        "| K | Label classifier target /percobaan | Label sumber unik /fold | "
        "Query sumber di dalam total |",
        "|---:|---:|---:|---:|",
    ]
    for b in budgets:
        lines.append(
            f"| {b['k']} | {b['target_labels_per_trial']} | "
            f"{b['source_labels_unique_min']}–{b['source_labels_unique_max']} | 960 |"
        )
    lines += [
        "",
        "![Biaya sumber](biaya-label-sumber.png)",
        "",
        "Pada K10,120 label target menggunakan prior pemilihan "
        "resep yang memerlukan5.749–5.807 label sumber unik. Union "
        "untuk seluruh K sama dengan K10 karena dukungan nested. "
        "ResNet pralatih juga memakai ulang720 label Harrison; "
        "generic pretraining BEATs lebih luas dan tidak disebut "
        "0 label total. Label query held ID untuk evaluasi, serta "
        "anotasi timestamp/filter/stratifikasi, berada di luar "
        "budget classifier target seperti pada pilot.",
        "",
        "## Pilihan dan kontrol",
        "",
        "Pilihan normal memakai row L2 pada288 dari340 kasus dan "
        "support z-score pada52. Tidak ada pilihan normal di batas "
        "bawah alpha1e-6; dua memilih batas atas100. Grid tidak "
        "diperluas berdasarkan hasil luar. Kontrol label acak "
        "memilih resepnya sendiri melalui prosedur16 ID dan "
        "18 resep yang sama, lalu memakai label dukungan permutasi "
        "yang sama dengan pilot pada ID yang ditahan.",
        "",
        "![Resep terpilih](pilihan-resep.png)",
        "",
        "## Audit dan stabilitas numerik",
        "",
        "Seluruh44.064 kandidat diperiksa terhadap sklearn;408 "
        "pilihan diperiksa dengan total integer dan tie rule "
        "independen. Mengubah hanya baris metrik held ID tidak "
        "mengubah resepnya. Semua2.448 refit cocok pada replay; "
        "statistik scaler cocok dengan population variance "
        "independen dan anggaran cocok dengan union indeks.",
        "",
        f"Selisih skor kandidat terhadap sklearn maksimal "
        f"{audit['maximum_errors']['candidate_sklearn']:.3g}; "
        "tidak ada prediksi kandidat yang berbeda. Replay final "
        f"maksimum{audit['maximum_errors']['final_replay']:.3g}; "
        "dual ridge independen berbeda maksimal "
        f"{audit['maximum_errors']['final_dual']:.3g}, tanpa "
        "perubahan prediksi. Toleransi parameter/skor "
        "atol1e-6/rtol1e-5 ditetapkan sebelum perhitungan.",
        "",
        "Kontrol memakai fitur CPU lengkap dari audit pilot yang "
        "sudah dipatok, tanpa ekstraksi baru. Ada7 perubahan "
        "prediksi dari146.880 prediksi berulang: satu BEATs normal "
        "(salah→salah) dan enam pada kontrol ResNet acak. Akurasi "
        "top1 seluruh kondisi normal tetap sama. MFCC dan ResNet "
        "pralatih tidak berubah prediksinya. Salah satu kontrol "
        "berubah dari benar ke salah; hasil CPU tidak digunakan "
        "untuk mengganti pilihan atau hasil utama.",
        "",
        "Selisih skor terbesar1,424 terjadi pada kontrol random_s44, "
        "support z-score/alpha1e-5. Mask near-constant tidak berubah; "
        "Gram kondisi ini memiliki condition number sekitar7,1e6. "
        "Pola konsisten dengan amplifikasi perubahan input kecil "
        "oleh sistem berkondisi buruk; ini inferensi diagnostik, "
        "bukan bukti penyebab tunggal. Skor Ridge bukan probabilitas. "
        "[Rincian numerik](../skaid-selection-numeric/summary.json).",
        "",
        "[Audit lengkap](../skaid-selection-audit/audit.json), "
        "[kandidat sumber](../skaid-selection-audit/validation-checks.csv) "
        "dan [prediksi final](../skaid-selection-audit/classifications.csv) "
        "mempertahankan seluruh kondisi termasuk kontrol.",
        "",
        "## Batas dan kelanjutan",
        "",
        "Kebijakan memperbaiki readout pada fitur yang sama, tetapi "
        "tetap memakai supervisi sumber tambahan. Ini tidak "
        "membuktikan seluruh gain berasal dari satu jenis normalisasi "
        "atau bahwa keluarga encoder tertentu adalah SOTA. "
        "Metadata belum membuktikan pergantian mikrofon/keyboard/"
        "sesi fisik. Crop80 ms dapat memuat ekor/noise meskipun "
        "tidak ada event lain pada log, dan onset fisik belum "
        "dianotasi manual. Bukan evaluasi detektor bebas timestamp "
        "atau biaya label end-to-end.",
        "",
        "Karena fit dukungan kini hampir penuh tetapi query masih "
        "terbatas, langkah berikutnya layak membandingkan kepala "
        "nonlinear dengan pemilihan resep source-only yang sama. "
        "Pertahankan ID, query, budget dan baseline; jangan memilih "
        "C/gamma dari query ID yang sedang dievaluasi. Dataset "
        "sendiri nantinya perlu identitas mikrofon/sesi yang jelas "
        "dan jumlah per tombol yang seimbang untuk konfirmasi.",
        "",
        "## Reproduksi",
        "",
        "Semua tahap memakai CPU dan membutuhkan "
        "cache/audit pilot pada path yang dipatok. Gunakan direktori baru.",
        "",
        "```bash",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\",
        "  .venv/bin/python -m ketik_audio.skaid_selection validate",
        ".venv/bin/python -m ketik_audio.skaid_selection select",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\",
        "  .venv/bin/python -m ketik_audio.skaid_selection evaluate",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\",
        "  .venv/bin/python research/audit_skaid_selection.py",
        ".venv/bin/python research/inspect_skaid_selection_numeric.py",
        ".venv/bin/python research/report_skaid_selection.py",
        "```",
        "",
        "[Resep terpilih](selected-recipes.csv), [biaya label](label-budgets.csv), "
        "[hasil per ID](per-id.csv).",
        "",
    ]
    (output / "laporan.md").write_text(readable_text("\n".join(lines)), encoding="utf-8")
    for name, digest in sig.items():
        if file_hash(ROOT / name) != digest:
            raise ResearchError("Masukan/kode laporan berubah.")
    summary = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "baseline": baseline,
        "selected": selected,
        "paired_deltas": delta,
        "source_label_budgets": budgets,
        "choice_statistics": choice_stats,
        "numeric_changes": numerical,
        "candidate_heads": 44064,
        "selected_heads": 2448,
        "choices": 408,
        "new_encoder_extractions": 0,
    }
    summary["artifact_sha256"] = {
        p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()
    }
    write_json(output / "summary.json", summary)
    if snapshot:
        if snapshot.exists():
            raise ResearchError("Snapshot sudah ada.")
        write_json(
            snapshot, {"report_summary_sha256": file_hash(output / "summary.json"), **summary}
        )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/skaid-selection")
    parser.add_argument(
        "--snapshot", type=Path, default=ROOT / "research/results/skaid-selection-2026-11-09.json"
    )
    args = parser.parse_args()
    try:
        run(args.output, args.snapshot)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
