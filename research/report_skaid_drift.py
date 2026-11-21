"""Report clock ablations and label availability after event-overlap filtering."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.skaid_drift import METHODS

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/skaid-drift"
BASE = ROOT / "reports/skaid-alignment"
AUDIT = ROOT / "reports/skaid-drift-audit"
LABELS = {
    "legacy": "Offset 10 ms",
    "refined_constant": "Offset diperhalus",
    "affine_policy": "Offset + drift",
}


def verify(base, record):
    for name, digest in record["input_sha256"].items():
        if file_hash(ROOT / name) != digest:
            raise ResearchError(f"Masukan berubah: {name}")
    for name, digest in record["artifact_sha256"].items():
        if file_hash(base / name) != digest:
            raise ResearchError(f"Artefak berubah: {base / name}")


def analyse(records, selection, eligibility):
    pairs = [r for r in records if r["status"] == "assessed"]
    methods = {}
    for method in METHODS:
        good = [r for r in pairs if r["methods"][method]["provisional_stable_match"]]
        tokens = sorted(
            {
                r["session_token"]
                for r in good
                if sum(x["session_token"] == r["session_token"] for x in good) == 2
            }
        )
        methods[method] = {
            "stable_pairs": len(good),
            "both_stable_ids": tokens,
            "median_fit_correlation_all_50": float(
                np.median([r["methods"][method]["fit"]["correlation"] for r in pairs])
            ),
            "median_validation_correlation_all_50": float(
                np.median([r["methods"][method]["validation"]["correlation"] for r in pairs])
            ),
        }
    comparisons = []
    for before, after in (
        ("legacy", "refined_constant"),
        ("refined_constant", "affine_policy"),
        ("legacy", "affine_policy"),
    ):
        delta = np.array(
            [
                r["methods"][after]["validation"]["correlation"]
                - r["methods"][before]["validation"]["correlation"]
                for r in pairs
            ]
        )
        comparisons.append(
            {
                "before": before,
                "after": after,
                "improved": int((delta > 1e-10).sum()),
                "unchanged": int((np.abs(delta) <= 1e-10).sum()),
                "worsened": int((delta < -1e-10).sum()),
                "median_delta": float(np.median(delta)),
                "minimum_delta": float(delta.min()),
                "maximum_delta": float(delta.max()),
            }
        )
    selected = [r for r in selection if r["status"] == "assessed"]
    affine = [
        r["selection"]["models"]["affine_policy"]["drift_ppm"]
        for r in selected
        if r["selection"]["affine_admitted"]
    ]
    return {
        "methods": methods,
        "comparisons": comparisons,
        "affine_admitted_pairs": len(affine),
        "fallback_pairs": len(selected) - len(affine),
        "admitted_drift_ppm": {
            "min": min(affine),
            "median": float(np.median(affine)),
            "max": max(affine),
        },
        "failed_affine_pairs": [
            {
                "pair_index": r["pair_index"],
                "session_token": r["session_token"],
                "phase": r["phase"],
            }
            for r in pairs
            if not r["methods"]["affine_policy"]["provisional_stable_match"]
        ],
        "label_feasibility": eligibility,
    }


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def plots(records, selected, eligibility, output):
    pairs = [r for r in records if r["status"] == "assessed"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), layout="constrained")
    for pair in pairs:
        y = [pair["methods"][m]["validation"]["correlation"] for m in METHODS]
        axes[0].plot(range(3), y, color="#7595a8", alpha=0.35, linewidth=0.8)
        axes[0].scatter(range(3), y, s=8, color="#255a78", alpha=0.6)
        before, after = y[0], y[-1]
        recovered = (
            not pair["methods"]["legacy"]["provisional_stable_match"]
            and pair["methods"]["affine_policy"]["provisional_stable_match"]
        )
        good = pair["methods"]["affine_policy"]["provisional_stable_match"]
        color = "#bc4b35" if not good else "#d59620" if recovered else "#177b58"
        axes[1].scatter(before, after, s=42, color=color, alpha=0.8)
        if recovered or not good:
            axes[1].annotate(
                str(pair["pair_index"]),
                (before, after),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=9,
            )
    axes[0].set_xticks(range(3), [LABELS[m] for m in METHODS], fontsize=9)
    axes[0].set(
        ylabel="Korelasi pola waktu paruh akhir",
        title="50 pasangan; semua parameter dari paruh awal",
    )
    axes[1].plot([-0.1, 0.6], [-0.1, 0.6], color="0.65", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="Offset tunggal: korelasi paruh akhir",
        ylabel="Offset + drift: korelasi paruh akhir",
        title="Kuning: empat pasangan pulih; merah: dua belum lolos",
    )
    for axis in axes:
        axis.grid(alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Koreksi drift membantu kecocokan waktu SKAID", fontsize=14)
    save(fig, output, "korelasi-sebelum-sesudah")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    for ax, tier, title in zip(
        axes,
        ("no_neighbor_press", "no_other_key_event"),
        ("Tanpa press tetangga", "Tanpa press/release tombol lain"),
        strict=True,
    ):
        values = np.array(
            [
                [
                    next(
                        r["eligible_ids"]
                        for r in eligibility
                        if r["method"] == "affine_policy"
                        and r["length_ms"] == length
                        and r["tier"] == tier
                        and r["vocabulary"] == "exploratory12"
                        and r["query_min_per_class"] == 5
                        and r["K"] == k
                    )
                    for k in (1, 2, 3, 5, 10)
                ]
                for length in (400, 200, 120, 80)
            ]
        )
        image = ax.imshow(values, cmap="viridis", vmin=0, vmax=23, aspect="auto")
        for i in range(4):
            for j in range(5):
                ax.text(
                    j,
                    i,
                    str(values[i, j]),
                    color="white" if values[i, j] < 12 else "black",
                    ha="center",
                    va="center",
                )
        ax.set_xticks(range(5), [1, 2, 3, 5, 10])
        ax.set_yticks(range(4), [400, 200, 120, 80])
        ax.set(xlabel="K kalibrasi per huruf pada Phase 1", ylabel="Panjang crop (ms)", title=title)
    fig.colorbar(image, ax=axes, label="ID memenuhi semua 12 kelas")
    fig.suptitle(
        "Kelayakan hitungan: 12 huruf, ≥5 query/kelas Phase 2\n"
        "23 ID memiliki dua fase lolos; belum klasifikasi",
        fontsize=12,
    )
    save(fig, output, "kelayakan-jendela")

    # Fixed illustrative long recording recovered by the affine policy.
    pair = next(p for p in pairs if p["pair_index"] == 45)
    chosen = next(p for p in selected if p["pair_index"] == 45)
    with np.load(BASE / "arrays/pair45.npz", allow_pickle=False) as z:
        feature, bins = z["onset"], z["relative_press_ms"] // 10
        models = chosen["selection"]["models"]
        affine = models["affine_policy"]
        target = bins[np.argmin(np.abs(bins - chosen["total_bins"] * 0.75))]
        center = (affine["a_bins"] + (1 + affine["b"]) * target) / 100
        left, right = center - 2, center + 2
        indices = np.arange(max(0, int(left * 100)), min(len(feature), int(right * 100)))
        fig, axes = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True, layout="constrained")
        for ax, method in zip(axes, ("legacy", "affine_policy"), strict=True):
            model = models[method]
            positions = (model["a_bins"] + (1 + model["b"]) * bins) / 100
            ax.plot(indices / 100, feature[indices], color="#285f80", linewidth=1)
            top = float(feature[indices].max())
            ax.vlines(
                positions[(positions >= left) & (positions <= right)],
                -top * 0.15,
                -top * 0.025,
                color="#b15a28",
                linewidth=1.4,
            )
            score = pair["methods"][method]["validation"]["correlation"]
            ax.set_title(f"{LABELS[method]}: r paruh akhir={score:.3f}", fontsize=11)
            ax.set(ylabel="Δ log RMS positif", xlim=(left, right))
            ax.spines[["top", "right"]].set_visible(False)
        axes[-1].set_xlabel("Waktu audio (detik); garis bawah: press log pada grid 10 ms")
        fig.suptitle("Pair45 / Phase 2: koreksi dipelajari dari bagian awal saja", fontsize=13)
        save(fig, output, "contoh-drift")


def run(output, snapshot):
    result, audit = read_json(RUN / "results.json"), read_json(AUDIT / "audit.json")
    verify(RUN, result)
    verify(AUDIT, audit)
    selection = read_json(RUN / "selection.json")
    pairs = read_json(RUN / "pairs.json")["records"]
    eligibility = read_json(RUN / "label-feasibility.json")["rows"]
    fresh_directory(output)
    paths = [
        Path(__file__),
        RUN / "results.json",
        RUN / "selection.json",
        RUN / "pairs.json",
        RUN / "label-feasibility.json",
        AUDIT / "audit.json",
    ]
    sig = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    write_json(output / "lock.json", {"created_at": now(), "input_sha256": sig})
    analysis = analyse(pairs, selection["records"], eligibility)
    plots(pairs, selection["records"], eligibility, output)
    lines = [
        "# SKAID: koreksi drift dan kelayakan segmentasi",
        "",
        "Tanggal lokal: 9 November 2026.",
        "",
        "**Koreksi offset+drift menaikkan pasangan yang lolos kriteria waktu "
        "sementara dari 44 menjadi 48 dari 50.** Offset diperhalus saja "
        "menghasilkan 45. Setelah penyaringan jendela 80 ms, ada 17 ID "
        "yang menyediakan ≥10 kalibrasi dan ≥5 query untuk masing-masing "
        "12 huruf eksploratif, tanpa press/release tombol lain yang "
        "tercatat di dalam crop. Belum mengukur akurasi huruf.",
        "",
        "## Perbandingan yang telah dijalankan",
        "",
        "[Protokol](../../research/SKAID_DRIFT_PROTOCOL.md) ditulis sebelum "
        "fit drift. Delapan blok pada paruh awal log menyediakan offset "
        "lokal; regresi berbobot memperkirakan offset dan perubahan laju "
        "jam. Gate fit menentukan penggunaan affine atau fallback. Semua "
        "parameter 50 pasangan dan 950 kontrol dikunci sebelum evaluasi "
        "paruh akhir; parameter tidak diperbarui dari hasil tersebut.",
        "",
        "Representasi evaluasi sama untuk tiga metode: interpolasi "
        "linear fitur onset audio pada koordinat log. Baseline legacy "
        "cocok dengan skor lama (selisih maksimum 2,46e-14), seluruh "
        "kontrol dan flag cocok. Perbandingan offset diperhalus vs affine "
        "memisahkan manfaat ketelitian offset dari koreksi laju waktu.",
        "",
        "| Metode | Pasangan lolos /50 | ID kedua fase lolos | Median r validasi, semua 50 |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        m = analysis["methods"][method]
        lines.append(
            f"| {LABELS[method]} | {m['stable_pairs']} | {len(m['both_stable_ids'])} | "
            f"{m['median_validation_correlation_all_50']:.4f} |"
        )
    lines += [
        "",
        "Median r pada semua 50 pasangan tidak sama dengan median "
        "pada subset yang lolos pada laporan sebelumnya.",
        "",
        "![Perbandingan waktu](korelasi-sebelum-sesudah.png)",
        "",
        "Dibanding legacy, kebijakan affine memperbaiki korelasi akhir "
        "pada 44 pasangan dan sama pada enam; tidak ada yang menurun "
        "pada dataset ini (toleransi 1e-10). Dibanding offset diperhalus, "
        "affine memperbaiki 41, sama enam, dan menurunkan tiga pasangan. "
        "Seluruh hasil tetap dilaporkan; tidak memilih metode per "
        "rekaman dari skor akhir.",
        "",
        "Gate affine lolos pada 44 pasangan; enam memakai fallback "
        "karena blok awal yang layak tidak cukup. Drift terestimasi "
        f"pada pasangan admitted berkisar {analysis['admitted_drift_ppm']['min']:.2f} "
        f"hingga {analysis['admitted_drift_ppm']['max']:.2f} ppm. "
        "Ini parameter kecocokan pola waktu, bukan pengukuran perangkat "
        "keras atau bukti penyebab fisik tertentu.",
        "",
        "Contoh pair45, Phase 2, rekaman sekitar 25 menit: korelasi "
        "akhir −0,0759 (legacy), −0,0758 (offset diperhalus), menjadi "
        "0,4279 (affine, drift≈−41,56 ppm). Perbedaan laju waktu kecil "
        "dapat menggeser onset puluhan milidetik pada rekaman panjang.",
        "",
        "![Contoh koreksi](contoh-drift.png)",
        "",
        "Dua pasangan masih gagal: pair15 (`qt_1749848552186`, Phase 2) "
        "dan pair46 (`qt_1753464926271`, Phase 1). Keduanya tidak memiliki "
        "blok fit yang memenuhi kriteria untuk memperkirakan drift; "
        "kebijakan memakai legacy. Tidak memaksa alignment atau "
        "membuang hasil ini berdasarkan akurasi classifier.",
        "",
        "## Apa yang tersisa setelah memeriksa jendela",
        "",
        "Event diproyeksikan ke frame native terdekat. Crop tidak "
        "digeser ke peak terbesar. Tiga tingkatan: lengkap di dalam "
        "PCM; lengkap tanpa initial press lain; dan lengkap tanpa "
        "press/release tombol lain, dengan release target berpasangan "
        "positif pada fase sama. Release milik target boleh ada. "
        "Tombol di luar kosakata tetap dihitung sebagai tetangga.",
        "",
        "Tabel berikut memakai kebijakan affine, 23 ID dengan kedua "
        "fase lolos, **K=10 Phase 1 dan ≥5 query/kelas Phase 2**, "
        "subset `e,t,a,o,i,n,s,h,r,d,l,u`.",
        "",
        "| Crop (pre) ms | Lengkap | Tanpa press tetangga | Tanpa event tombol lain |",
        "|---|---:|---:|---:|",
    ]
    for length, pre in ((400, 50), (200, 30), (120, 20), (80, 15)):
        values = [
            next(
                r["eligible_ids"]
                for r in eligibility
                if r["method"] == "affine_policy"
                and r["length_ms"] == length
                and r["tier"] == tier
                and r["vocabulary"] == "exploratory12"
                and r["query_min_per_class"] == 5
                and r["K"] == 10
            )
            for tier in ("complete", "no_neighbor_press", "no_other_key_event")
        ]
        lines.append(f"| {length} ({pre}) | {values[0]} | {values[1]} | {values[2]} |")
    lines += [
        "",
        "![Kelayakan jendela dan K](kelayakan-jendela.png)",
        "",
        "Angka tersebut menghitung ID yang semua kelasnya cukup, "
        "bukan jumlah akurasi benar. Jendela 80 ms paling mendukung "
        "kecukupan contoh pada grid ini; belum terbukti paling baik "
        "untuk klasifikasi karena juga memotong ekor suara tombol. "
        "Tidak ada event log lain di crop juga tidak menjamin bebas "
        "ekor ketukan sebelumnya atau noise.",
        "",
        "Untuk 26 huruf, bahkan syarat ≥5 query per kelas tidak "
        "terpenuhi oleh satu ID pun pada crop80 ms, sebelum maupun "
        "sesudah filter event tetangga. Karena itu pilot12 kelas harus "
        "dinyatakan eksplisit dan tidak diekstrapolasi ke alfabet penuh.",
        "",
        "## Verifikasi dan batas",
        "",
        f"Audit mengulang {audit['clock_pipelines_replayed']} pipeline "
        f"pemilihan clock dan {audit['method_evaluations_replayed']} "
        "evaluasi metode dari masukan yang dipatok. Semua hasil "
        "identik. Regresi diperiksa terhadap penyelesaian weighted "
        "least squares independen; selisih maksimum "
        f"{audit['max_independent_regression_error_bins']:.3g} bin. "
        f"Sebanyak {audit['direct_interval_checks']:,} interval crop "
        "diperiksa dengan scan event langsung; kepemilikan release "
        "diverifikasi lewat pencarian release berikutnya. "
        "[Audit lengkap](../skaid-drift-audit/audit.json).",
        "",
        "Studi ini eksploratif setelah melihat drift pada data yang "
        "sama. Semua audio tanpa identitas huruf dipakai untuk "
        "pencarian; bagian akhir bukan audio yang sepenuhnya tak "
        "dilihat. Kontrol waktu 19/99 memiliki resolusi terbatas dan "
        "belum dikoreksi untuk banyak perbandingan. Flag sementara "
        "tidak mensertifikasi onset fisik per event. Ketelitian fitur "
        "masih 10 ms, meskipun proyeksi crop memakai frame native.",
        "",
        "Hasil belum merupakan evaluasi detektor bebas timestamp, "
        "klasifikasi huruf, atau efisiensi adaptasi ke mikrofon/sesi "
        "baru. Metadata SKAID tidak memberikan ID mikrofon dan "
        "keyboard fisik. Fase adalah rekaman terpisah pada log yang sama.",
        "",
        "## Eksperimen berikutnya",
        "",
        "Kunci pilot12 kelas pada crop80 ms dengan 17 ID yang memenuhi "
        "filter dan jumlah contoh di atas. Tetapkan K=1/2/3/5/10, "
        "dukungan Phase 1, query tetap yang seimbang Phase 2, kontrol "
        "label, frontend dan model sebelum melihat akurasi. Ini pilot "
        "berbasis timestamp acuan yang masih memiliki ketidakpastian "
        "onset. Bandingkan representasi akustik pada label unik yang "
        "sama; evaluasi detektor dilakukan terpisah.",
        "",
        "## Reproduksi",
        "",
        "Memerlukan audit native dan offset lama "
        "pada path yang dipatok; gunakan direktori hasil baru. Semua CPU.",
        "",
        "```bash",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\",
        "  .venv/bin/python research/run_skaid_drift.py select",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\",
        "  .venv/bin/python research/run_skaid_drift.py evaluate",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\",
        "  .venv/bin/python research/audit_skaid_drift.py",
        ".venv/bin/python research/report_skaid_drift.py",
        "```",
        "",
        "[CSV waktu](../../runs/skaid-drift/timing.csv), "
        "[CSV kelayakan lengkap](../../runs/skaid-drift/label-feasibility.csv), "
        "[pilihan parameter terkunci](../../runs/skaid-drift/selection.json).",
        "",
    ]
    (output / "laporan.md").write_text("\n".join(lines), encoding="utf-8")
    for name, digest in sig.items():
        if file_hash(ROOT / name) != digest:
            raise ResearchError("Masukan laporan berubah.")
    summary = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "analysis": analysis,
        "run_results": {
            k: v for k, v in result.items() if k not in ("input_sha256", "artifact_sha256")
        },
        "classifiers_trained_or_evaluated": 0,
        "segmentation_certified": False,
    }
    summary["artifact_sha256"] = {
        str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()
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
    parser.add_argument("--output", type=Path, default=ROOT / "reports/skaid-drift")
    parser.add_argument(
        "--snapshot", type=Path, default=ROOT / "research/results/skaid-drift-2026-11-09.json"
    )
    args = parser.parse_args()
    try:
        run(args.output, args.snapshot)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
