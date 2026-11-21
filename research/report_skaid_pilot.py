"""Report all fixed-budget SKAID predictors, ID uncertainty, controls and error patterns."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.skaid_pilot_analysis import FAMILIES, analyse
from ketik_audio.skaid_pilot_common import CACHE, DATA, KEYS, KS, PROTOCOL, ROOT, RUN, verify

LABELS = {
    "mfcc": "MFCC",
    "beats": "BEATs beku",
    "pretrained": "ResNet pralatih",
    "random": "ResNet acak",
}
COLORS = {"mfcc": "#177b58", "beats": "#b97818", "pretrained": "#2d63a5", "random": "#9466a8"}
AUDIT = ROOT / "reports/skaid-pilot-audit"


def percent(value):
    return f"{100 * value:.2f}".replace(".", ",")


def interval(row):
    lo, hi = row["ci95_id_bootstrap"]
    return f"{percent(lo)}–{percent(hi)}%"


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def plots(analysis, output):
    groups = {(r["family"], r["k"], r["permuted"]): r for r in analysis["groups"]}
    fig, ax = plt.subplots(figsize=(9, 5.8), layout="constrained")
    for f in FAMILIES:
        rows = [groups[f, k, False]["accuracy"] for k in KS]
        mean = np.array([r["mean"] for r in rows]) * 100
        limits = np.array([r["ci95_id_bootstrap"] for r in rows]).T * 100
        ax.plot(KS, mean, marker="o", color=COLORS[f], label=LABELS[f])
        ax.fill_between(KS, limits[0], limits[1], color=COLORS[f], alpha=0.12)
    ax.axhline(100 / 12, linestyle="--", color="0.45", linewidth=1, label="Peluang uniform 1/12")
    ax.set(
        xlabel="K label unik per huruf pada Phase 1",
        ylabel="Akurasi query Phase 2 (%)",
        title="SKAID: kalibrasi 12 huruf pada rekaman terpisah",
        ylim=(0, 100),
    )
    ax.set_xticks(KS)
    upper = ax.secondary_xaxis("top", functions=(lambda x: 12 * x, lambda x: x / 12))
    upper.set_xticks([12 * k for k in KS])
    upper.set_xlabel("Jumlah label dukungan per ID per percobaan")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["right"]].set_visible(False)
    fig.text(
        0.5,
        0.005,
        "Pita: 95% bootstrap17 ID; crop80 ms, timestamp acuan, query tetap60/ID.",
        ha="center",
        fontsize=9,
    )
    save(fig, output, "kurva-k")

    fig, ax = plt.subplots(figsize=(9, 4.9), layout="constrained")
    x = np.arange(4)
    for offset, field, label, color in (
        (-0.19, "support_fit_label_accuracy", "Dukungan: fit-label", "#8eaeb7"),
        (0.19, "accuracy", "Query: label asli", "#25678c"),
    ):
        values = [
            groups[f, 10, False][field]["mean"]
            if field == "accuracy"
            else groups[f, 10, False][field]
            for f in FAMILIES
        ]
        ax.bar(x + offset, 100 * np.array(values), width=0.34, label=label, color=color)
    ax.scatter(
        x,
        [100 * groups[f, 10, True]["accuracy"]["mean"] for f in FAMILIES],
        marker="x",
        color="#ac423a",
        label="Query: label dukungan diacak",
        s=60,
    )
    ax.axhline(100 / 12, color="0.5", linestyle="--", linewidth=1)
    ax.set_xticks(x, [LABELS[f] for f in FAMILIES])
    ax.set(
        ylabel="Akurasi (%)", ylim=(0, 100), title="K10: kecocokan dukungan dan transfer ke query"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, output, "dukungan-dan-kontrol")

    values = (
        np.array(
            [
                [
                    next(
                        r["accuracy"]
                        for r in analysis["per_id"]
                        if r["subject"] == s
                        and r["family"] == "mfcc"
                        and r["k"] == k
                        and not r["permuted"]
                    )
                    for k in KS
                ]
                for s in range(17)
            ]
        )
        * 100
    )
    fig, ax = plt.subplots(figsize=(7.5, 8), layout="constrained")
    im = ax.imshow(values, vmin=0, vmax=100, cmap="viridis", aspect="auto")
    for i in range(17):
        for j in range(5):
            ax.text(
                j,
                i,
                f"{values[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if values[i, j] < 60 else "black",
            )
    ax.set_xticks(range(5), KS)
    ax.set_yticks(range(17), [f"ID {s:02d}" for s in range(17)])
    ax.set(xlabel="K per huruf", ylabel="ID; rerata tiga draw", title="MFCC: variasi antar-ID")
    fig.colorbar(im, ax=ax, label="Akurasi query (%)")
    save(fig, output, "mfcc-per-id")


def errors(trials, records, labels, output):
    selected = [t for t in trials if t["encoder"] == "mfcc" and t["k"] == 10 and not t["permuted"]]
    matrix = np.zeros((12, 12), dtype=np.int64)
    rows = []
    with np.load(RUN / "mfcc.npz", allow_pickle=False) as z:
        for t in selected:
            i = t["array_index"]
            q, scores = z["query"][i], z["query_scores"][i]
            pred = scores.argmax(1)
            np.add.at(matrix, (labels[q], pred), 1)
            for j in np.flatnonzero(labels[q] != pred):
                index = int(q[j])
                rows.append(
                    {
                        "subject": t["subject"],
                        "draw": t["draw"],
                        "clip_id": records[index]["id"],
                        "true_key": KEYS[int(labels[index])],
                        "prediction": KEYS[int(pred[j])],
                        "query_raw_file": records[index]["audio"],
                    }
                )
    with (output / "errors-mfcc-k10.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    values = matrix / matrix.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    im = ax.imshow(values, cmap="Blues", vmin=0, vmax=100)
    for i in range(12):
        for j in range(12):
            if values[i, j] >= 3 or i == j:
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if values[i, j] >= 60 else "black",
                )
    ax.set_xticks(range(12), KEYS)
    ax.set_yticks(range(12), KEYS)
    ax.set(
        xlabel="Prediksi",
        ylabel="Huruf acuan",
        title="MFCC K10: confusion per baris (%)\n1.020 query unik diulang pada3 draw dukungan",
    )
    fig.colorbar(im, ax=ax, label="Persentase per kelas acuan")
    save(fig, output, "confusion-mfcc-k10")
    pairs = sorted(
        (
            (int(matrix[i, j]), KEYS[i], KEYS[j], float(values[i, j]))
            for i in range(12)
            for j in range(12)
            if i != j
        ),
        reverse=True,
    )
    return {
        "matrix": matrix.tolist(),
        "prediction_rows": int(matrix.sum()),
        "unique_queries": 1020,
        "top_confusions": [
            {"true_key": a, "prediction": b, "count_repeated_queries": n, "row_percent": v}
            for n, a, b, v in pairs[:5]
        ],
    }


def run(output, snapshot):
    result, audit = read_json(RUN / "results.json"), read_json(AUDIT / "audit.json")
    for base, record in (
        (DATA, read_json(DATA / "manifest.json")),
        (CACHE, read_json(CACHE / "manifest.json")),
        (RUN, result),
        (AUDIT, audit),
    ):
        verify(base, record)
    fresh_directory(output)
    paths = [
        Path(__file__),
        ROOT / "src/ketik_audio/skaid_pilot_analysis.py",
        PROTOCOL,
        DATA / "manifest.json",
        CACHE / "manifest.json",
        RUN / "results.json",
        RUN / "trials.json",
        AUDIT / "audit.json",
    ]
    sig = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    write_json(output / "lock.json", {"created_at": now(), "input_sha256": sig})
    trials = read_json(RUN / "trials.json")
    analysis = analyse(trials["trials"], trials["zero_shot"])
    plots(analysis, output)
    with np.load(DATA / "segments.npz", allow_pickle=False) as z:
        labels = z["y"]
    records = read_json(DATA / "records.json")["records"]
    error_analysis = errors(trials["trials"], records, labels, output)
    for name, rows in (("per-id", analysis["per_id"]), ("per-condition", trials["trials"])):
        flat = [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows]
        if name == "per-condition":
            flat = [
                {**v, **{f"query_{k}": value for k, value in r["query"].items()}}
                for v, r in zip(flat, rows, strict=True)
            ]
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    groups = {(r["family"], r["k"], r["permuted"]): r for r in analysis["groups"]}
    lines = [
        "# SKAID: pilot klasifikasi dengan sedikit label",
        "",
        "Tanggal lokal: 9 November 2026.",
        "",
        "**MFCC + Ridge mencapai 52,29% pada K10, lebih tinggi daripada "
        "ResNet pralatih 43,12%, BEATs 26,31% dan ResNet acak 13,44% pada "
        "setup ini.** Semua keluarga membaik ketika label bertambah, "
        "tetapi belum mencapai rerata 80% atau 90% pada grid K1–10. "
        "Ini 12 huruf dengan timestamp/filter acuan, bukan alfabet penuh "
        "atau evaluasi detektor pada rekaman tanpa anotasi.",
        "",
        "## Dataset, budget dan kontrol",
        "",
        "[Protokol](../../research/SKAID_PILOT_PROTOCOL.md) dikunci "
        "sebelum fitur/akurasi SKAID dihitung. Terdapat17 ID,34 M4A, "
        "6.137 crop unik terpilih:5.117 union dukungan tiga draw dan "
        "1.020 query tetap. Tiap ID memakai60 query,5 per huruf, dari "
        "Phase 2. Dukungan Phase 1 memiliki12×K label unik pada "
        "K1/2/3/5/10; tiga draw nested101/102/103. Dukungan dan query "
        "berasal dari file berbeda. Query sama untuk seluruh metode.",
        "",
        "Kelas alfabetis: `a,d,e,h,i,l,n,o,r,s,t,u`. Crop80 ms/pre15 ms "
        "harus lengkap dan tidak memuat press/release tombol lain "
        "menurut log. Audio diresample48→16 kHz setelah crop. MFCC "
        "memakai80 ms asli; neural menerima buffer nol400 ms dengan "
        "audio pada35–115 ms, press acuan50 ms. Frontend neural "
        "berbeda konteks dari sumber400 ms penuh dan dapat "
        "memengaruhi transfer.",
        "",
        "Seluruh backbone beku. ResNet pralatih memakai ketiga "
        "checkpoint Harrison seed42/43/44 dan720 label sumber "
        "(648 fit +72 validasi) yang dipakai ulang. Random ResNet "
        "memakai arsitektur/mean-std sumber sama tanpa pelatihan bobot. "
        "BEATs memakai encoder generik AudioSet yang telah dipatok. "
        "Kepala seluruh fitur sama: L2 per baris, Ridge alpha1, "
        "tanpa scaler/query statistics, target validation atau tuning "
        "hyperparameter dari query. Random encoder bukan scratch "
        "end-to-end yang dioptimasi.",
        "",
        "Ada2.040 fit normal,408 kontrol permutasi label dukungan "
        "pada K10, dan51 evaluasi head sumber K0. Kontrol menjaga "
        "jumlah tiap label dukungan; query tidak diubah. Tiga ResNet "
        "disilangkan dengan tiga draw, lalu akurasi dirata-ratakan; "
        "bukan ensemble logits atau pemilihan seed terbaik.",
        "",
        "## Seluruh kurva label",
        "",
        "Tabel adalah mean antar17 ID setelah rerata draw/model "
        "di dalam ID. Query seimbang: peluang uniform8,33%.",
        "",
        "| K / label per ID | MFCC | BEATs beku | ResNet pralatih | ResNet acak |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in KS:
        values = [percent(groups[f, k, False]["accuracy"]["mean"]) for f in FAMILIES]
        lines.append(f"| {k} / {12 * k} | " + " | ".join(v + "%" for v in values) + " |")
    lines += [
        "",
        "![Kurva label](kurva-k.png)",
        "",
        "## K10 dan ketidakpastian",
        "",
        "| Fitur | Mean query | SD antar ID | 95% CI bootstrap ID | "
        "Dukungan fit-label | Kontrol query |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for f in FAMILIES:
        row, control = groups[f, 10, False], groups[f, 10, True]
        accuracy = row["accuracy"]
        lines.append(
            f"| {LABELS[f]} | {percent(accuracy['mean'])}% | {percent(accuracy['sd_ids'])} pp | "
            f"{interval(accuracy)} | {percent(row['support_fit_label_accuracy'])}% | "
            f"{percent(control['accuracy']['mean'])}% |"
        )
    lines += [
        "",
        "CI menggunakan10.000 resample **ID**, seed7471, bukan "
        "crop, draw atau seed sebagai observasi independen. CI ini "
        "bersyarat pada17 ID yang lolos penyaringan. Perbandingan "
        "berpasangan MFCC−ResNet pralatih pada K10 adalah "
        "+9,17 pp, CI95% +6,02 hingga+12,71 pp. Semua perbandingan "
        "keluarga tersimpan di JSON; studi eksploratif tidak "
        "menetapkan peringkat universal atau SOTA.",
        "",
        "![Dukungan dan kontrol](dukungan-dan-kontrol.png)",
        "",
        "Akurasi dukungan K10 juga belum tinggi pada kepala linear "
        "ini. Hasil konsisten dengan kemungkinan keterbatasan "
        "representasi, regularisasi atau ketepatan crop; belum "
        "memisahkan penyebabnya. Audit dual ridge cocok, sehingga "
        "penyelesaian kepala sesuai objektif yang ditetapkan.",
        "",
        f"K0 head sumber ResNet, logits dibatasi ke12 kelas, "
        f"mencapai{percent(analysis['source_zero_shot']['mean'])}% "
        f"(CI95% {interval(analysis['source_zero_shot'])}). "
        "Kosakata subset sudah diketahui; tidak ada fit target. "
        "Tidak membandingkan angka ini dengan peluang36 kelas.",
        "",
        "![Variasi MFCC](mfcc-per-id.png)",
        "",
        "Rerata MFCC K10 per ID berkisar37,78–75,56%. Tidak ada "
        "kebutuhan label minimum80%/90% yang dapat ditetapkan "
        "dari grid ini. K10 berarti120 label dukungan; hasil belum "
        "membuktikan bahwa sekadar menambah label akan mencapai "
        "ambang tertentu.",
        "",
        "## Observasi kesalahan",
        "",
        "![Confusion MFCC](confusion-mfcc-k10.png)",
        "",
        "Confusion memuat3.060 prediksi MFCC K10, yaitu1.020 query "
        "yang sama diulang pada tiga draw dukungan. Ini bukan3.060 "
        "audio independen. Confusion berikut bersifat deskriptif; "
        "tidak menjelaskan penyebab fisik kesalahan.",
        "",
        "| Acuan → prediksi | Hitungan query berulang | Persentase baris acuan |",
        "|---|---:|---:|",
    ]
    for row in error_analysis["top_confusions"]:
        lines.append(
            f"| {row['true_key']} → {row['prediction']} | {row['count_repeated_queries']} | "
            f"{row['row_percent']:.2f}% |"
        )
    lines += [
        "",
        "Seluruh kesalahan ada di [CSV MFCC K10](errors-mfcc-k10.csv). "
        "[Prediksi seluruh kondisi](../skaid-pilot-audit/classifications.csv) "
        "menyimpan label acuan, prediksi utama dan prediksi replay CPU.",
        "",
        "## Verifikasi komputasi",
        "",
        "Ekstraksi neural memakai **AMD Radeon RX6600 LE**, backend "
        "ROCm. Bobot dan buffer tetap identik sebelum/sesudah. "
        "Kontrol sumber BEATs36 clip serta ResNet180 contoh cocok "
        "dengan artefak Harrison lama; prediksi sumber identik.",
        "",
        "Audit mendekode ulang seluruh34 file dan6.137 crop: hash "
        "native, resample, padding dan array identik. Seluruh fitur "
        "dihitung ulang di CPU, dalam toleransi atol/rtol1e-3. "
        "Semua2.448 kepala dan51 evaluasi K0 di-replay. Error replay "
        f"kepala maksimum{audit['max_head_replay_error']:.3g}; "
        "error terhadap dual centered ridge independen maksimum "
        f"{audit['max_independent_dual_error']:.3g}. "
        "Mengganti seluruh fitur dengan hasil CPU dan mem-fit ulang "
        "semua kepala tidak mengubah prediksi query, termasuk K0. "
        "[Audit terstruktur](../skaid-pilot-audit/audit.json).",
        "",
        "## Batas interpretasi dan langkah berikutnya",
        "",
        "Budget12×K hanya label kepala klasifikasi. Segmentasi, "
        "kepemilikan event, penyaringan tetangga, stratifikasi pool "
        "dan query menggunakan anotasi log lebih lengkap. Karena itu "
        "angka ini tidak mengukur biaya label end-to-end atau deteksi "
        "bebas timestamp. Onset fisik belum dianotasi manual dan "
        "ketelitian fitur clock masih10 ms. Crop tanpa event lain "
        "di log juga dapat memuat ekor suara/noise.",
        "",
        "Seleksi17 ID dan subset12 huruf terjadi sebelum akurasi, "
        "tetapi berdasarkan kelayakan waktu/label dataset yang sudah "
        "diperiksa. Fase merupakan file terpisah, belum bukti "
        "mikrofon, keyboard fisik atau sesi baru. Tidak menggunakan "
        "hasil ini sebagai reproduksi literal EchoStrike atau "
        "pengenalan seluruh alfabet.",
        "",
        "Kelanjutan yang layak adalah menguji regularisasi dan "
        "normalisasi kepala melalui validasi pada ID sumber yang "
        "berbeda, dengan seluruh pilihan dikunci sebelum evaluasi "
        "ID yang ditahan. Perlu menghitung biaya label sumber "
        "tambahan dan mempertahankan baseline ini. Jangan memilih "
        "alpha/scaler dari query ID yang sedang dievaluasi.",
        "",
        "## Reproduksi",
        "",
        "Jalankan dari checkout dengan artefak "
        "audit SKAID dan sumber Harrison/BEATs yang sudah dipatok. "
        "Setiap tahap memakai direktori baru.",
        "",
        "```bash",
        ".venv/bin/python -m ketik_audio.skaid_pilot_data",
        ".venv-rocm/bin/python -m ketik_audio.skaid_pilot_features",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \\",
        "  .venv/bin/python -m ketik_audio.skaid_pilot_probe",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \\",
        "  .venv/bin/python research/audit_skaid_pilot.py",
        ".venv/bin/python research/report_skaid_pilot.py",
        "```",
        "",
        "[CSV per kondisi](per-condition.csv), [CSV per ID](per-id.csv). "
        "State/source hashes, split/rank, model dan skor tersedia "
        "pada manifest serta arsip NPZ yang direferensikan audit.",
        "",
    ]
    (output / "laporan.md").write_text("\n".join(lines), encoding="utf-8")
    for name, digest in sig.items():
        if file_hash(ROOT / name) != digest:
            raise ResearchError("Masukan/kode laporan berubah.")
    summary = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "analysis": analysis,
        "errors_mfcc_k10": error_analysis,
        "subjects": 17,
        "classes": list(KEYS),
        "unique_query_examples": 1020,
        "heads": 2448,
        "backbone_updates": 0,
        "conditional_on_oracle_log_segmentation": True,
        "cpu_feature_prediction_disagreements": audit[
            "cpu_feature_query_prediction_disagreements_repeated_trials"
        ],
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
    parser.add_argument("--output", type=Path, default=ROOT / "reports/skaid-pilot")
    parser.add_argument(
        "--snapshot", type=Path, default=ROOT / "research/results/skaid-pilot-2026-11-09.json"
    )
    args = parser.parse_args()
    try:
        run(args.output, args.snapshot)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
