"""Summarize SKAID native and timing audits without classification claims."""

from __future__ import annotations

import argparse
import csv
import string
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "reports/skaid-audit"
TIMING = ROOT / "reports/skaid-alignment"
REPLAY = ROOT / "reports/skaid-alignment-audit"
PAPER = (
    "https://www.researchgate.net/publication/405307424_From_Keystrokes_to_Semantics_"
    "Acoustic_Side-Channel_Attacks_on_Typed_Text_Using_Deep_Neural_Networks_"
    "and_Large_Language_Models"
)


def verify(base, record):
    for name, digest in record["input_sha256"].items():
        if file_hash(ROOT / name) != digest:
            raise ResearchError(f"Masukan berubah: {name}")
    for name, digest in record["artifact_sha256"].items():
        if file_hash(base / name) != digest:
            raise ResearchError(f"Artefak berubah: {base / name}")


def feasibility(pairs, tokens):
    groups = {t: {p["phase"]: p for p in pairs if p["session_token"] == t} for t in tokens}
    fully_covered = {
        t: d
        for t, d in groups.items()
        if all(p["press_before_audio"] + p["press_after_audio"] == 0 for p in d.values())
    }
    rows = []
    for k in (1, 2, 3, 5, 10):
        row = {"K": k}
        for group_name, group in (
            ("both_stable", groups),
            ("both_stable_no_outside_press", fully_covered),
        ):
            for label, letters in (
                ("26", string.ascii_lowercase),
                ("12_exploratory", "etaoinshrdlu"),
            ):
                row[f"{group_name}_{label}"] = sum(
                    all(
                        d["Phase 1"]["provisional_in_audio_letter_counts"].get(c, 0) >= k
                        and d["Phase 2"]["provisional_in_audio_letter_counts"].get(c, 0) >= 1
                        for c in letters
                    )
                    for d in group.values()
                )
        rows.append(row)
    return rows, len(fully_covered)


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def plots(assessed, output):
    fig, ax = plt.subplots(figsize=(8, 5.8), layout="constrained")
    for stable, marker, color, label in (
        (True, "o", "#177b58", "Lolos kriteria sementara (44)"),
        (False, "x", "#bb413c", "Belum lolos (6)"),
    ):
        group = [p for p in assessed if p["provisional_stable_match"] == stable]
        ax.scatter(
            [p["fit_peak"]["correlation"] for p in group],
            [p["validation"]["correlation"] for p in group],
            c=color,
            marker=marker,
            label=label,
            s=45,
            alpha=0.9,
        )
    for row in assessed:
        if not row["provisional_stable_match"]:
            ax.annotate(
                str(row["pair_index"]),
                (row["fit_peak"]["correlation"], row["validation"]["correlation"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )
    ax.axhline(0.1, color="0.5", linestyle="--", linewidth=1)
    ax.axvline(0.1, color="0.5", linestyle="--", linewidth=1)
    ax.plot([0, 0.56], [0, 0.56], color="0.8", linewidth=1)
    ax.set(
        xlabel="Korelasi template fit: paruh waktu awal log",
        ylabel="Korelasi paruh akhir pada offset fit yang tetap",
        title="Pola waktu SKAID: 50 pasangan audio–log",
        xlim=(-0.015, 0.56),
        ylim=(-0.12, 0.56),
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, output, "korelasi-waktu")

    residuals = np.asarray(
        [
            [
                q["residual_lag_ms"] if q["residual_lag_ms"] is not None else np.nan
                for q in p["quarters"]
            ]
            for p in assessed
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(7.5, 13.5), layout="constrained")
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dddddd")
    picture = ax.imshow(residuals, aspect="auto", cmap=cmap, vmin=-50, vmax=50)
    ax.set_xticks(range(4), ["0–25%", "25–50%", "50–75%", "75–100%"])
    ax.set_yticks(
        range(len(assessed)),
        [
            f"{p['pair_index']:02d}: {p['session_token'][3:]} / {p['phase'][-1]}"
            f"{' *' if not p['provisional_stable_match'] else ''}"
            for p in assessed
        ],
        fontsize=7,
    )
    for i, values in enumerate(residuals):
        for j, value in enumerate(values):
            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(value) >= 35 else "black",
                )
    ax.set_title(
        "Pergeseran puncak lokal terhadap offset fit (ms)\n"
        "* belum lolos; abu-abu: overlap tidak cukup",
        fontsize=11,
    )
    ax.set_xlabel("Bagian waktu log; offset hasil fit tetap dipertahankan")
    fig.colorbar(picture, ax=ax, label="Δ lag lokal (ms); warna dibatasi ±50")
    save_figure(fig, output, "pergeseran-waktu")

    selected = (0, 50, 45, 46)
    lookup = {p["pair_index"]: p for p in assessed}
    fig, axes = plt.subplots(4, 2, figsize=(15, 10), layout="constrained")
    for i, index in enumerate(selected):
        row = lookup[index]
        with np.load(TIMING / "arrays" / f"pair{index:02d}.npz", allow_pickle=False) as z:
            feature = z["onset"]
            relative = z["relative_press_ms"] / 1000
            position = relative + row["offset_seconds"]
            for j, fraction in enumerate((0.25, 0.75)):
                target = row["span_ms"] / 1000 * fraction
                center = position[np.argmin(np.abs(relative - target))]
                lo = max(0.0, min(center - 4, len(feature) / 100 - 8))
                hi = lo + 8
                bins = np.arange(max(0, int(lo * 100)), min(len(feature), int(hi * 100)))
                ax = axes[i, j]
                ax.plot(
                    bins / 100,
                    feature[bins],
                    color="#35698c",
                    linewidth=0.8,
                    label="Onset energi audio",
                )
                marks = position[(position >= lo) & (position <= hi)]
                upper = max(0.1, float(np.max(feature[bins]))) if len(bins) else 0.1
                ax.vlines(
                    marks,
                    -upper * 0.10,
                    -upper * 0.015,
                    color="#bd6c26",
                    linewidth=1,
                    label="Press log + offset fit",
                )
                ax.set(xlim=(lo, hi), xlabel="Waktu audio (detik)", ylabel="Δ log RMS positif")
                corr = (
                    row["fit_peak"]["correlation"] if j == 0 else row["validation"]["correlation"]
                )
                ax.set_title(
                    f"Pair {index:02d} / fase {row['phase'][-1]} / "
                    f"{'awal' if j == 0 else 'akhir'}: r={corr:.3f}",
                    fontsize=10,
                )
                ax.spines[["top", "right"]].set_visible(False)
                if i == 0 and j == 0:
                    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Overlay diagnostik: offset tidak diperbaiki dari bagian akhir\n"
        "Energi dan waktu press saja; bukan anotasi onset manual",
        fontsize=13,
    )
    save_figure(fig, output, "overlay-waktu")


def number(value, digits=3):
    return f"{value:.{digits}f}".replace(".", ",")


def run(output):
    native, timing, replay = (
        read_json(p)
        for p in (NATIVE / "summary.json", TIMING / "summary.json", REPLAY / "audit.json")
    )
    for base, record in ((NATIVE, native), (TIMING, timing), (REPLAY, replay)):
        verify(base, record)
    fresh_directory(output)
    paths = [
        NATIVE / "summary.json",
        TIMING / "summary.json",
        TIMING / "pairs.json",
        TIMING / "transformed-copy.json",
        REPLAY / "audit.json",
        Path(__file__),
    ]
    signature = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    write_json(output / "lock.json", {"created_at": now(), "input_sha256": signature})
    pairs = read_json(TIMING / "pairs.json")["pairs"]
    assessed = [p for p in pairs if p["status"] == "assessed"]
    stable = [p for p in assessed if p["provisional_stable_match"]]
    copy = read_json(TIMING / "transformed-copy.json")
    feasibility_rows, full_tokens = feasibility(stable, timing["tokens_with_both_phases_stable"])
    with (output / "kelayakan-k.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(feasibility_rows[0]))
        writer.writeheader()
        writer.writerows(feasibility_rows)
    plots(assessed, output)
    partial = [p for p in stable if p["press_before_audio"] + p["press_after_audio"]]
    stats = {
        "stable_fit_correlation_median": float(
            np.median([p["fit_peak"]["correlation"] for p in stable])
        ),
        "stable_validation_correlation_median": float(
            np.median([p["validation"]["correlation"] for p in stable])
        ),
        "stable_consecutive_press_gap_median_ms_across_files": float(
            np.median([p["consecutive_press_gaps"]["median_ms"] for p in stable])
        ),
        "stable_fraction_consecutive_gaps_under_400ms_median_across_files": float(
            np.median([p["consecutive_press_gaps"]["fraction_under_400ms"] for p in stable])
        ),
        "stable_validation_nearest_onset_median_ms_across_files": float(
            np.median([p["validation_nearest_onset"]["median_ms"] for p in stable])
        ),
        "stable_validation_nearest_onset_p90_median_ms_across_files": float(
            np.median([p["validation_nearest_onset"]["p90_ms"] for p in stable])
        ),
    }
    summary = {
        "created_at": now(),
        "local_date": "2026-11-09",
        "status": "complete",
        "input_sha256": signature,
        "native": {k: v for k, v in native.items() if k not in ("artifact_sha256", "input_sha256")},
        "timing": {k: v for k, v in timing.items() if k not in ("artifact_sha256", "input_sha256")},
        "replay": {k: v for k, v in replay.items() if k not in ("artifact_sha256", "input_sha256")},
        "transformed_copy_check": copy,
        "descriptive_statistics": stats,
        "provisional_stable_partial_pairs": len(partial),
        "both_stable_no_outside_press_tokens": full_tokens,
        "label_feasibility": feasibility_rows,
        "exploratory_12_letters": list("etaoinshrdlu"),
        "models_trained_or_evaluated": 0,
        "segmentation_certified": False,
    }
    lines = [
        "# SKAID: audit audio dan sinkronisasi awal",
        "",
        "Tanggal lokal: 9 November 2026.",
        "",
        "**44 dari 50 pasangan yang bisa diperiksa memenuhi kriteria kecocokan waktu "
        "sementara.** Ada 20 ID dengan kedua fase lolos; 17 ID di antaranya tidak "
        "memiliki initial press di luar audio pada resolusi bin 10 ms. Ini dasar "
        "untuk memeriksa segmentasi lintas rekaman. Belum ada klasifikasi huruf, "
        "pengujian mikrofon baru, atau penetapan jumlah label minimum.",
        "",
        "## Audit berkas",
        "",
        "Sumber publik: [SKAID v1, Quattrone & Badr, Zenodo]"
        "(https://zenodo.org/records/17282184), CC BY 4.0. Arsip audio "
        "182.788.356 byte diverifikasi MD5 penyedia dan SHA256; semua anggota "
        "ZIP dibaca hingga CRC terverifikasi. Seluruh 53 audio berhasil didekode "
        "dan bernilai finite. Audio, log dan README tidak diubah.",
        "",
        "| Temuan aktual | Jumlah / konsekuensi |",
        "|---|---|",
        "| Log event dan demografi | 27 ID; 202.510 event |",
        "| Rekaman utama | 52 M4A, 26 ID × dua fase; AAC mono 48 kHz |",
        "| Berkas tambahan | Satu WAV mono 16 kHz pada Phase 2 salah satu ID |",
        "| Audio tidak tersedia | `qt_1750019330494`, kedua fase |",
        "| Timestamp runtuh ke satu nilai | `qt_1753467140611`, "
        "7.452 event; kedua fase tidak diperiksa |",
        "| Metadata perangkat | Desktop/Laptop saja; tidak ada ID mikrofon atau keyboard fisik |",
        "",
        "Durasi total 366,37 menit mencakup WAV tambahan. Tidak ada hash berkas "
        "atau PCM identik pada 53 file, tetapi itu tidak membuktikan semua "
        "berkas adalah pengambilan independen. Seluruh M4A memiliki tag encoder "
        "Apple Voice Memos; tag creation_time berasal dari jam perangkat dan "
        "tidak dijadikan onset ketukan.",
        "",
        f"WAV tambahan memiliki {copy['wav_frames']:,} frame pada 16 kHz; "
        "hasil resample M4A pasangannya memiliki panjang sama dan lag terbaik nol. "
        f"Pearson {number(copy['pearson'], 6)}, gain {number(copy['least_squares_gain'], 6)}, "
        f"residual relatif {number(copy['relative_residual_rms'], 6)}. "
        "Keduanya sangat mirip, tetapi belum memenuhi ambang ketat protokol "
        "(r≥0,999, residual≤0,05) untuk flag salinan hasil resample. WAV tetap "
        "tidak ditambahkan sebagai unit atau pengambilan independen; hanya "
        "M4A kanonis digunakan.",
        "",
        "## Diagnostik waktu yang sudah dijalankan",
        "",
        "[Protokol](../../research/SKAID_ALIGNMENT_PROTOCOL.md) dikunci sebelum "
        "menghitung korelasi. Initial physical press direkonstruksi dari "
        "press/release; autorepeat yang ditandai dikeluarkan dari template. "
        "Nama huruf tidak ikut pencarian. Pola press paruh awal log dicocokkan "
        "dengan positive difference log-RMS audio per 10 ms. Offset hasil fit "
        "dipertahankan pada paruh akhir. Seluruh audio tanpa label memang "
        "dipakai untuk transformasi dan pencarian; ini validasi pola waktu "
        "berikutnya, bukan holdout audio yang sama sekali tak dilihat.",
        "",
        "Setiap pasangan memiliki 19 kontrol gap-permuted dengan pencarian "
        "offset ulang dan 99 kontrol validasi pada offset tetap. Flag lolos "
        "memerlukan korelasi fit/validasi≥0,10, event coverage≥80%, margin "
        "puncak≥0,02, ≥25 press per bagian, seluruh kontrol di bawah observasi, "
        "dan pergeseran puncak validasi lokal≤30 ms. Resolusi kontrol terbatas; "
        "ini bukan pengujian konfirmatori dengan koreksi banyak perbandingan.",
        "",
        f"Median korelasi pada 44 pasangan yang lolos: fit "
        f"{number(stats['stable_fit_correlation_median'])}, validasi "
        f"{number(stats['stable_validation_correlation_median'])}. "
        "Pergeseran puncak validasi lokal pada kelompok ini −20 hingga +20 ms. "
        "Offset beberapa rekaman negatif: sebagian log mendahului audio. "
        "Lima pasangan yang lolos masih memiliki press di luar rekaman "
        "(1, 80, 1, 1, dan 3); press itu tidak boleh dibuat menjadi segmen "
        "audio berlabel. Coverage penuh pada bin 10 ms pun belum menjamin "
        "jendela crop lengkap.",
        "",
        "![Korelasi waktu](korelasi-waktu.png)",
        "",
        "Enam pasangan belum lolos. Semuanya gagal ambang korelasi; dua juga "
        "gagal keunikan puncak/kontrol fit, tiga gagal kontrol validasi atau "
        "stabilitas lokal, dan satu gagal coverage. Ini tidak menetapkan "
        "bahwa log salah: offset tunggal, drift, release, suara lain atau "
        "rekaman terpotong dapat memengaruhi fitur energi. Peta kuartal "
        "menunjukkan perubahan lag bertahap 10–40 ms pada beberapa rekaman; "
        "koreksi drift belum diterapkan.",
        "",
        "![Pergeseran lokal](pergeseran-waktu.png)",
        "",
        "[Overlay awal/akhir](overlay-waktu.png) menampilkan dua contoh lolos "
        "dan dua contoh gagal. Garis bawah menunjukkan waktu press dari log "
        "dengan offset fit; kurva menunjukkan fitur energi, bukan ground "
        "truth onset manual. Seluruh pasangan ada di "
        "[CSV waktu](../skaid-alignment/timing.csv) dan "
        "[hasil terstruktur](../skaid-alignment/pairs.json).",
        "",
        "## Kelayakan label dan jendela",
        "",
        "Hitungan berikut hanya memakai initial press yang diproyeksikan "
        "berada di audio pada 20 ID dengan kedua fase lolos. Syarat: ≥K "
        "contoh tiap kelas pada Phase 1 dan ≥1 pada Phase 2. Belum ada "
        "penyaringan berdasarkan kelengkapan crop, overlap atau anotasi manual.",
        "",
        "| K | 26 huruf, 20 ID | 12 huruf eksploratif, 20 ID | "
        "26 huruf, 17 ID tanpa press di luar | 12 huruf, 17 ID tanpa press di luar |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in feasibility_rows:
        lines.append(
            f"| {r['K']} | {r['both_stable_26']} | "
            f"{r['both_stable_12_exploratory']} | "
            f"{r['both_stable_no_outside_press_26']} | "
            f"{r['both_stable_no_outside_press_12_exploratory']} |"
        )
    lines.extend(
        [
            "",
            "Subset eksploratif: **e,t,a,o,i,n,s,h,r,d,l,u**. Subset ini "
            "diperiksa setelah melihat kelayakan 26 huruf; belum merupakan "
            "kosakata praregistrasi untuk eksperimen klasifikasi. Hasil pada "
            "12 huruf tidak boleh dinyatakan sebagai pengenalan seluruh alfabet. "
            "Anggaran K=10 pada 12 kelas berarti 120 label unik per ID.",
            "",
            f"Median jarak press berturut-turut per rekaman yang lolos adalah "
            f"{number(stats['stable_consecutive_press_gap_median_ms_across_files'], 1)} ms. "
            "Median fraksi gap<400 ms per rekaman adalah "
            f"{
                number(
                    100 * stats['stable_fraction_consecutive_gaps_under_400ms_median_across_files'],
                    2,
                )
            }%. "
            "Karena itu jendela 400 ms dari Harrison tidak dapat dipindahkan "
            "begitu saja sebagai suara satu tombol pada ketikan alami. "
            "Perlu aturan segmentasi/overlap yang dinyatakan sebelum melatih model.",
            "",
            "Median jarak ke puncak energi terdekat pada validasi adalah 10 ms "
            "dan median p90 antar-rekaman 20 ms pada kelompok lolos. Banyak "
            "press boleh mendekati puncak yang sama; angka itu bukan "
            "precision/recall detektor dan tidak menyertifikasi label per huruf.",
            "",
            "## Verifikasi dan hubungan dengan paper",
            "",
            "Seluruh 50 pasangan diputar ulang: semua metrik dan array "
            "identik. Rekonstruksi state press dilakukan kembali dari event "
            "mentah; template diperiksa dengan konvolusi independen. "
            f"Selisih maksimum template {replay['max_independent_template_error']:.3g}; "
            "korelasi FFT dibandingkan centered dot product langsung, selisih "
            f"maksimum {replay['max_direct_correlation_error']:.3g}. "
            "Hash kode, sumber dan artefak diverifikasi sebelum/sesudah replay. "
            "[Audit replay](../skaid-alignment-audit/audit.json) menegaskan "
            "reprodusibilitas numerik, bukan kebenaran setiap onset fisik.",
            "",
            f"Paper [EchoStrike, Bagian III.D]({PAPER}) "
            "menjelaskan penyesuaian satu offset manual sebelum segmentasi. "
            "Arsip audio, log, README dan metadata yang diperiksa belum "
            "memberikan tabel offset. Prosedur otomatis ini berbeda, sehingga belum "
            "merupakan reproduksi hasil klasifikasi EchoStrike. Tidak "
            "menyaring data berdasarkan akurasi classifier.",
            "",
            "## Keputusan riset berikutnya",
            "",
            "1. Uji model offset+drift dengan parameter dipelajari dari bagian "
            "awal saja dan pemeriksaan pola waktu bagian akhir yang tetap. "
            "Pertahankan hasil offset tunggal sebagai baseline dan laporkan "
            "seluruh pasangan yang gagal.",
            "2. Periksa ketepatan onset dan kelengkapan jendela pada sampel "
            "awal/akhir; tetapkan toleransi dan aturan overlap tanpa akurasi "
            "huruf. Jangan menggeser tiap contoh ke peak terbesar tanpa batas.",
            "3. Setelah segmentasi lulus, kunci kosakata dan eksperimen "
            "K=1/2/3/5/10 sebelum melihat akurasi: kalibrasi Phase 1, uji "
            "Phase 2, hitung label unik dan bandingkan baseline serta kontrol. "
            "Sebut hasil lintas rekaman/tugas; identitas mikrofon, keyboard "
            "atau sesi baru belum didukung metadata.",
            "",
            "## Reproduksi",
            "",
            "Jalankan dari checkout proyek dengan direktori "
            "keluaran baru; seluruh tahap ini memakai CPU dan memerlukan "
            "ffmpeg/ffprobe. Jangan memakai direktori hasil yang sudah ada.",
            "",
            "```bash",
            ".venv/bin/python research/fetch_skaid.py",
            ".venv/bin/python research/audit_skaid.py --output reports/skaid-audit",
            "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/align_skaid.py",
            "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "
            ".venv/bin/python research/audit_skaid_alignment.py",
            ".venv/bin/python research/report_skaid.py --output reports/skaid",
            "```",
            "",
            "Katalog log awal perlu tersedia di `data/external/skaid-catalog/`; "
            "lihat `research/catalog_skaid.py`. Untuk checkout dengan data/hasil "
            "yang sudah terverifikasi, tidak perlu mengunduh ulang. Manifest "
            "unduhan menyimpan timestamp; jangan menulis ulang provenance sumber "
            "di tengah replay hasil lama.",
            "",
        ]
    )
    (output / "laporan.md").write_text("\n".join(lines), encoding="utf-8")
    for name, digest in signature.items():
        if file_hash(ROOT / name) != digest:
            raise ResearchError("Masukan laporan berubah.")
    summary["artifact_sha256"] = {
        str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/skaid")
    args = parser.parse_args()
    try:
        run(args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
