"""Report linear and RBF SVM policies against frozen source-selected Ridge."""

from __future__ import annotations

import argparse
import copy
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_skaid_selection import budget_ranges, comparisons, plot_costs, save

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.skaid_pilot_analysis import FAMILIES, analyse
from ketik_audio.skaid_pilot_common import DATA, ENCODERS, KS, ROOT, verify
from ketik_audio.skaid_pilot_probe import metrics
from ketik_audio.skaid_svm import OUTPUT as RUN
from ketik_audio.skaid_svm import RIDGE
from ketik_audio.skaid_svm_core import POLICIES

AUDIT = ROOT / "reports/skaid-svm-audit"
NUMERIC = ROOT / "reports/skaid-svm-numeric"
BOUNDS = ROOT / "reports/skaid-svm-bounds"
NAMES = {"mfcc": "MFCC", "beats": "BEATs", "pretrained": "ResNet pralatih", "random": "ResNet acak"}
COLORS = {"ridge": "#777777", "linear": "#2866b0", "rbf": "#15815c"}
LABELS = {"ridge": "Ridge terpilih", "linear": "SVM linear", "rbf": "SVM RBF"}


def pct(x):
    return f"{100 * x:.2f}".replace(".", ",")


def group(analysis, family, k, permuted=False):
    return next(
        r
        for r in analysis["groups"]
        if r["family"] == family and r["k"] == k and r["permuted"] == permuted
    )


def csv_file(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plots(analyses, costs, output):
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 7.5), sharex=True, sharey=True, layout="constrained"
    )
    for ax, family in zip(axes.flat, FAMILIES, strict=True):
        for policy, analysis in analyses.items():
            values = [group(analysis, family, k)["accuracy"] for k in KS]
            ax.plot(
                KS,
                [100 * r["mean"] for r in values],
                marker="o",
                color=COLORS[policy],
                label=LABELS[policy],
                linestyle="--" if policy == "ridge" else "-",
            )
            if policy != "ridge":
                bounds = np.array([r["ci95_id_bootstrap"] for r in values]).T * 100
                ax.fill_between(KS, *bounds, color=COLORS[policy], alpha=0.1)
        ax.axhline(100 / 12, color="0.7", linestyle=":", linewidth=1)
        ax.set(
            title=NAMES[family],
            xlabel="K label per huruf pada target",
            ylabel="Akurasi query (%)",
            ylim=(0, 100),
        )
        ax.set_xticks(KS)
        ax.grid(axis="y", alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "SKAID: Ridge terpilih, SVM linear, dan SVM RBF\n"
        "Pita: CI95% bersyarat 17 ID; 12 kelas dan crop 80 ms",
        fontsize=13,
    )
    save(fig, output, "kurva-svm-vs-ridge")
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 7.5), sharex=True, sharey=True, layout="constrained"
    )
    for ax, family in zip(axes.flat, FAMILIES, strict=True):
        for policy, analysis in analyses.items():
            rows = sorted(
                [
                    r
                    for r in analysis["per_id"]
                    if r["family"] == family and r["k"] == 10 and not r["permuted"]
                ],
                key=lambda r: r["subject"],
            )
            ax.plot(
                range(17),
                [100 * r["accuracy"] for r in rows],
                marker="o",
                markersize=4,
                linewidth=1,
                color=COLORS[policy],
                label=LABELS[policy],
            )
        ax.set(
            title=NAMES[family],
            xlabel="Indeks ID pada cohort tetap",
            ylabel="Akurasi query K10 (%)",
            ylim=(0, 100),
        )
        ax.set_xticks(range(17))
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Hasil seluruh 17 ID pada K10\nRata-rata draw/model dalam ID, bukan ensemble", fontsize=13
    )
    save(fig, output, "seluruh-id-k10")
    plot_costs(costs, output)


def run(output):
    result, audit = read_json(RUN / "results.json"), read_json(AUDIT / "audit.json")
    numeric = read_json(NUMERIC / "summary.json")
    bound_diagnostic = read_json(BOUNDS / "summary.json")
    verify(RUN, result)
    verify(AUDIT, audit)
    verify(NUMERIC, numeric)
    verify(BOUNDS, bound_diagnostic)
    paths = [
        Path(__file__),
        ROOT / "research/report_skaid_selection.py",
        ROOT / "research/review_skaid_pilot_report.py",
        ROOT / "src/ketik_audio/skaid_pilot_analysis.py",
        RUN / "results.json",
        AUDIT / "audit.json",
        NUMERIC / "summary.json",
        BOUNDS / "summary.json",
        RIDGE / "results.json",
    ]
    sig = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    fresh_directory(output)
    write_json(output / "lock.json", {"created_at": now(), "input_sha256": sig})
    trials = read_json(RUN / "trials.json")
    old = read_json(RIDGE / "trials.json")
    analyses = {"ridge": analyse(old["trials"], old["zero_shot"])}
    for policy in POLICIES:
        analyses[policy] = analyse(
            [t for t in trials["trials"] if t["policy"] == policy], trials["zero_shot"]
        )
    cpu_trials = copy.deepcopy(trials["trials"])
    with np.load(DATA / "segments.npz", allow_pickle=False) as z:
        labels = z["y"]
    for encoder in ENCODERS:
        with np.load(AUDIT / f"cpu-scores-{encoder}.npz", allow_pickle=False) as z:
            scores = z["query_scores"]
        with np.load(RUN / "final" / f"{encoder}.npz", allow_pickle=False) as z:
            queries = z["query"]
        for t in cpu_trials:
            if t["encoder"] == encoder:
                i = t["array_index"]
                t["query"] = metrics(labels[queries[i]], scores[i])
    cpu_analyses = {
        p: analyse([t for t in cpu_trials if t["policy"] == p], trials["zero_shot"])
        for p in POLICIES
    }
    paired = {f"{p}_minus_ridge": comparisons(analyses["ridge"], analyses[p]) for p in POLICIES}
    paired["rbf_minus_linear"] = comparisons(analyses["linear"], analyses["rbf"])
    costs = budget_ranges(read_json(RUN / "budgets.json")["budgets"])
    choices = read_json(RUN / "selection.json")["choices"]
    statistics = {}
    for policy in POLICIES:
        normal = [c for c in choices if c["policy"] == policy and not c["permuted"]]
        statistics[policy] = {
            "normalizers": dict(Counter(c["recipe"]["normalizer"] for c in normal)),
            "c": dict(Counter(str(c["recipe"]["c"]) for c in normal)),
            "multipliers": dict(Counter(str(c["recipe"]["multiplier"]) for c in normal)),
        }
    plots(analyses, costs, output)
    csv_file(
        output / "per-id.csv",
        [{"policy": p, **r} for p, a in analyses.items() for r in a["per_id"]],
    )
    csv_file(
        output / "selected-recipes.csv",
        [
            {
                "id": c["id"],
                "policy": c["policy"],
                "family": c["family"],
                "held": c["held"],
                "k": c["k"],
                "permuted": c["permuted"],
                "recipe_index": c["recipe_index"],
                **c["recipe"],
                "source_labels_unique_total": c["source_labels_unique_total"],
            }
            for c in choices
        ],
    )
    csv_file(output / "label-budgets.csv", costs)
    text = [
        "# SKAID: perbandingan SVM dan Ridge dengan fitur tetap",
        "",
        "Eksplorasi lanjutan pada 17 ID yang sama, 12 huruf dan crop 80 ms. "
        "Parameter dipilih dari 16 ID lain; query held ID dikeluarkan dari kriteria. "
        "Kedua kebijakan SVM dilaporkan, tanpa memilih kebijakan terbaik dari query target.",
        "",
        "## Kurva seluruh K",
        "",
        "| Fitur | K | Ridge terpilih | SVM linear | SVM RBF |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        for k in KS:
            values = [pct(group(analyses[p], family, k)["accuracy"]["mean"]) for p in analyses]
            text.append(f"| {NAMES[family]} | {k} | {'% | '.join(values)}% |")
    text += [
        "",
        "![Kurva](kurva-svm-vs-ridge.png)",
        "",
        "Mean antar ID setelah rata-rata draw/model. Pita adalah CI bootstrap "
        "deskriptif bersyarat cohort; fold berbagi sumber. "
        "Bukan CI mikrofon/sesi baru atau ensemble.",
        "",
        "## K10 dan delta berpasangan",
        "",
        "| Fitur | Kebijakan | Akurasi | CI95% | Delta vs Ridge [CI95%] (pp) | "
        "ID naik/tetap/turun |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        for p in POLICIES:
            acc = group(analyses[p], family, 10)["accuracy"]
            delta = next(
                r for r in paired[f"{p}_minus_ridge"] if r["family"] == family and r["k"] == 10
            )
            lo, hi = acc["ci95_id_bootstrap"]
            dlo, dhi = delta["selected_minus_baseline"]["ci95_id_bootstrap"]
            text.append(
                f"| {NAMES[family]} | {LABELS[p]} | {pct(acc['mean'])}% | "
                f"{pct(lo)}–{pct(hi)}% | {pct(delta['selected_minus_baseline']['mean'])} "
                f"[{pct(dlo)}; {pct(dhi)}] | "
                f"{delta['ids_improved']}/{delta['ids_unchanged']}/{delta['ids_worsened']} |"
            )
    text += [
        "",
        "![Seluruh ID](seluruh-id-k10.png)",
        "",
        "## Dukungan, kontrol dan CPU",
        "",
        "| Fitur | Kebijakan | Fit dukungan normal K10 | Query kontrol K10 | "
        "Query normal cache CPU K10 |",
        "|---|---|---:|---:|---:|",
    ]
    for family in FAMILIES:
        for p in POLICIES:
            normal, control = group(analyses[p], family, 10), group(analyses[p], family, 10, True)
            cpu_acc = group(cpu_analyses[p], family, 10)["accuracy"]["mean"]
            text.append(
                f"| {NAMES[family]} | {LABELS[p]} | "
                f"{pct(normal['support_fit_label_accuracy'])}% | "
                f"{pct(control['accuracy']['mean'])}% | {pct(cpu_acc)}% |"
            )
    text += [
        "",
        "Peluang uniform 8,33%. Kontrol memilih resep dari tabel permutasi tersendiri "
        "dengan fleksibilitas sama. Replay CPU mempertahankan resep; "
        "tidak memilih ulang parameter.",
        "",
        "## Supervisi dan audit",
        "",
        "| K | Label target per model | Label sumber unik per fold |",
        "|---:|---:|---:|",
    ]
    for b in costs:
        text.append(
            f"| {b['k']} | {b['target_labels_per_trial']} | "
            f"{b['source_labels_unique_min']}–{b['source_labels_unique_max']} |"
        )
    text += [
        "",
        "![Biaya label](biaya-label-sumber.png)",
        "",
        "Budget sumber mencakup union dukungan tiga draw serta 960 query dari 16 ID lain; "
        "dipakai ulang kedua kebijakan. ResNet pralatih juga memakai 720 label Harrison. "
        "Supervisi pralatih generik BEATs dan anotasi log/filter bukan nol label. "
        "Budget target bersyarat pada segmentasi acuan, bukan biaya end-to-end.",
        "",
        f"Audit memeriksa {audit['candidate_heads_checked']:,} kandidat lewat kernel langsung, "
        f"{audit['choices_checked']} pilihan dan {audit['final_heads_replayed']:,} kepala final. "
        "Variasi prediksi kandidat precomputed/direct: "
        f"{audit['candidate_direct_prediction_disagreements']}; kandidat di luar toleransi skor: "
        f"{audit['candidate_heads_outside_score_tolerance']}. Perubahan prediksi replay dual: "
        f"{audit['dual_prediction_disagreements']}; cache fitur CPU: "
        f"{audit['cpu_feature_prediction_disagreements']}.",
        "",
        "Seluruh variasi numerik disimpan, termasuk kandidat ekstrem yang tidak terpilih. "
        "Skor OvR berupa vote ditambah koreksi confidence, bukan probabilitas. "
        "Jika ada perbedaan kernel/CPU, lihat audit dan diagnostik "
        "sebelum menafsirkan kenaikan kecil.",
        "",
        "Diagnosis memeriksa margin pasangan kelas tanpa mengubah parameter. "
        f"Kandidat outlier terpilih untuk ID-nya sendiri: "
        f"{numeric['outlier_heads_selected_for_own_held_id']}. Entri count sumber "
        f"yang berubah dengan kernel langsung: {numeric['source_correct_count_entries_changed']}; "
        "pilihan diagnostik yang akan berubah: "
        f"{len(numeric['diagnostic_source_choices_that_would_change_with_direct_counts'])}. "
        "Ini pemeriksaan sensitivitas; pilihan asli tetap terkunci.",
        "",
        "| Kondisi | Perubahan prediksi CPU | Perubahan prediksi dual | "
        "Kondisi berubah akurasi di CPU |",
        "|---|---:|---:|---:|",
        *[
            f"| {k} | {v['cpu_prediction_changes']} | {v['dual_prediction_changes']} | "
            f"{v['cpu_accuracy_changed_conditions']} |"
            for k, v in numeric["final_numeric_summary"].items()
        ],
        "",
        "[Diagnostik margin](../skaid-svm-numeric/candidate-diagnostics.json) "
        "memuat seluruh outlier numerik; selisih skor yang melebihi toleransi "
        "tetap merupakan temuan audit, meskipun tidak mengubah hasil terpilih.",
        "",
        "Diagnosis lanjutan terhadap intercept memakai status koefisien di batas C. "
        "Pada kasus terbesarnya, koefisien pasangan hanya berbeda 1,42e-14, "
        "tetapi intercept berbeda 0,26548. Satu jalur memiliki satu koefisien "
        "sedikit di bawah C, jalur lain tepat di C. Kedua intercept berada dalam "
        "interval KKT diagnostik yang sama. Ini konsisten dengan sensitivitas "
        "pemilihan intercept ketika tidak ada koefisien interior yang jelas, "
        "bukan sekadar margin biner yang selalu sangat kecil.",
        "",
        "[Sumber libsvm](https://raw.githubusercontent.com/scikit-learn/"
        "scikit-learn/1.9.1/sklearn/svm/src/libsvm/svm.cpp) membedakan status "
        "bound secara eksak dan memakai rumus rho berbeda saat ada koefisien free. "
        "[Kasus dan interval](../skaid-svm-bounds/cases.json) merupakan analisis "
        "post-hoc; tidak melakukan snapping koefisien atau mengubah model.",
        "",
        "Maksimum selisih numerik (nilai absolut, kecuali count di atas):",
        "",
        "| Pembandingan | Maksimum |",
        "|---|---:|",
        *[f"| `{k}` | {v:.9g} |" for k, v in audit["maximum_errors"].items()],
        "",
        "Pilihan pada 340 kondisi normal per kebijakan:",
        "",
        "| Kebijakan | Row L2 | Support z-score | C=0,01 | C=100 | g=0,01 | g=100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {LABELS[p]} | {statistics[p]['normalizers'].get('row_l2', 0)} | "
            f"{statistics[p]['normalizers'].get('support_z_l2', 0)} | "
            f"{statistics[p]['c'].get('0.01', 0)} | {statistics[p]['c'].get('100.0', 0)} | "
            f"{statistics[p]['multipliers'].get('0.01', 0)} | "
            f"{statistics[p]['multipliers'].get('100.0', 0)} |"
            for p in POLICIES
        ],
        "",
        "Batas grid dilaporkan agar hasil yang tersaturasi tidak disalahartikan "
        "sebagai optimum universal. Grid tetap, tidak diperluas dari hasil luar. "
        "Seluruh parameter ada pada CSV pilihan dan summary JSON.",
        "",
        "## Batas interpretasi",
        "",
        "Penelitian ini memakai ulang cohort dan query yang sudah diperiksa, "
        "sehingga bersifat eksploratif. Timestamp/filter masih memakai log, "
        "kosakata hanya 12 huruf; metadata belum membuktikan mikrofon, "
        "keyboard atau sesi fisik berbeda. Hasil bukan detektor bebas log, "
        "reproduksi literal paper, atau penetapan SOTA. Tidak memperluas grid "
        "atau menyaring ID setelah melihat evaluasi luar.",
        "",
        "[Protokol](../../research/SKAID_SVM_PROTOCOL.md), "
        "[audit](../skaid-svm-audit/audit.json), "
        "[prediksi](../skaid-svm-audit/classifications.csv), "
        "[pilihan](selected-recipes.csv), [per-ID](per-id.csv), "
        "dan [biaya label](label-budgets.csv).",
        "",
    ]
    (output / "laporan.md").write_text("\n".join(text), encoding="utf-8")
    if any(file_hash(ROOT / p) != digest for p, digest in sig.items()):
        raise ResearchError("Masukan laporan berubah.")
    record = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "analyses": analyses,
        "cpu_analyses": cpu_analyses,
        "paired_deltas": paired,
        "source_label_budgets": costs,
        "choice_statistics": statistics,
        "audit": {k: v for k, v in audit.items() if k not in ("input_sha256", "artifact_sha256")},
        "numeric": {
            k: v for k, v in numeric.items() if k not in ("input_sha256", "artifact_sha256")
        },
        "bounds_diagnostic": {
            k: v
            for k, v in bound_diagnostic.items()
            if k not in ("input_sha256", "artifact_sha256")
        },
        "candidate_heads": result["candidate_heads"],
        "selected_heads": result["outer_refits"],
        "choices": len(choices),
    }
    record["artifact_sha256"] = {
        str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    write_json(output / "summary.json", record)
    snapshot = ROOT / "research/results/skaid-svm-2026-11-11.json"
    if snapshot.exists():
        raise ResearchError("Jangan menimpa snapshot.")
    write_json(snapshot, {"report_summary_sha256": file_hash(output / "summary.json"), **record})
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/skaid-svm")
    args = parser.parse_args()
    try:
        run(args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
