"""Audit the verified MKA archive and decide which evaluation its provenance supports."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.mka import (
    PLATFORMS,
    compare_harrison,
    scan_archive,
    summarise_inventory,
    trace_segments,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA256 = "01fad57bf972782aa1cab74602e1965c0e489d7dceb37e36668a1d44f95ecb64"


def diagnostic_plot(archive, rows, traces, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(6, 1, figsize=(12, 10), constrained_layout=True)
    with zipfile.ZipFile(archive) as source:
        for platform, ax in zip(PLATFORMS, axes, strict=True):
            parent = next(
                r
                for r in rows
                if r["scope"] == "canonical"
                and r["platform"] == platform
                and r["role"] == "raw"
                and r["label"] == "a"
            )
            audio, rate = sf.read(io.BytesIO(source.read(parent["path"])), always_2d=True)
            mono = audio[:, np.var(audio, axis=0).argmax()]
            hop = max(1, round(rate * 0.01))
            valid = len(mono) // hop * hop
            rms = np.sqrt(np.mean(mono[:valid].reshape(-1, hop) ** 2, axis=1))
            ax.plot((np.arange(len(rms)) + 0.5) * hop / rate, rms, color="#267a9e", linewidth=0.8)
            for trace in traces["traces"]:
                if trace["platform"] == platform and trace["key"] == "a" and trace["matched"]:
                    ax.axvspan(
                        trace["start_seconds"], trace["stop_seconds"], color="#d57b28", alpha=0.18
                    )
            ax.set_ylabel(f"{platform}\nRMS")
            ax.set_xlim(0, len(mono) / rate)
            ax.set_ylim(bottom=0)
    axes[-1].set_xlabel("Waktu dalam WAV mentah (detik)")
    figure.suptitle(
        "MKA, contoh tombol a: RMS 10 ms dan interval potongan yang ditelusuri\n"
        "Bidang jingga = potongan tersedia dengan |r| ≥ 0,999; bukan anotasi event ketukan",
        fontsize=12,
    )
    figure.savefig(output / "asal-potongan.png", dpi=180)
    figure.savefig(output / "asal-potongan.pdf")
    plt.close(figure)


def render(summary, output):
    lines = [
        "# Audit kelayakan MKA v3",
        "",
        "Sumber: Karwan Mahdi Rawf dan Ayub Abdulrahman, "
        "[MKA v3](https://data.mendeley.com/datasets/bpt2hvf8n3/3), "
        "DOI 10.17632/bpt2hvf8n3.3, CC BY 4.0. Angka di bawah berasal "
        "dari analisis lokal arsip, bukan evaluasi model. Audio asli "
        "dipertahankan tanpa perubahan; float64 dipakai untuk fingerprint.",
        "",
        f"Arsip: {summary['archive_bytes']:,} byte, SHA256 "
        f"`{summary['archive_sha256']}`. {summary['zip_entries']:,} entri, "
        f"{summary['file_count']:,} berkas; semua CRC anggota sudah diperiksa.",
        "",
        "## Audio dan kelas",
        "",
        "| Folder | WAV mentah | Kelas mentah | Tombol bersama Harrison | "
        "Potongan audio tersedia | WAV mentah/tombol bersama |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for platform in PLATFORMS:
        row = summary["platforms"][platform]
        counts = row["raw_files_per_common_key"].values()
        lines.append(
            f"| {platform} | {row['raw_files']} | {row['raw_classes']} | "
            f"{len(row['common_36_keys_present'])} | {row['provided_segment_files']} | "
            f"{min(counts)}–{max(counts)} |"
        )
    lines += [
        "",
        "Nama platform tidak dianggap identitas partisipan, keyboard, atau sesi. "
        "`All Dataset` dianalisis sebagai agregat dan tidak dihitung sebagai "
        "platform atau rekaman baru. Kelas bersama ditetapkan dari nama a–z/0–9.",
        "",
        "| Folder | Format mentah | Rentang durasi mentah | Rentang durasi potongan |",
        "|---|---|---:|---:|",
    ]
    for platform in PLATFORMS:
        row = summary["platforms"][platform]
        low, high = row["raw_duration_seconds_range"]
        clip_low, clip_high = row["segment_duration_seconds_range"]
        lines.append(
            f"| {platform} | {', '.join(row['raw_rates_channels_subtypes'])} | "
            f"{low:.3f}–{high:.3f} s | {clip_low:.3f}–{clip_high:.3f} s |"
        )
    lines += [
        "",
        f"Total audio terdecode: {summary['audio_files']:,} berkas, "
        f"{summary['unique_audio_pcm']:,} fingerprint PCM unik. "
        f"Kegagalan decode: {len(summary['audio_decode_errors'])}; "
        f"audio hening/non-finite: {len(summary['silent_or_nonfinite_audio'])}.",
        "",
        "## Duplikasi dan provenance",
        "",
        f"Dari {summary['aggregate_audio_files']:,} audio agregat, "
        f"{summary['aggregate_audio_with_canonical_byte_match']:,} mempunyai "
        "salinan byte persis di folder platform dan "
        f"{summary['aggregate_audio_with_canonical_pcm_match']:,} mempunyai "
        "PCM persis. Salinan tersebut bukan tambahan label independen.",
        "",
        f"Kelompok duplikasi byte seluruh berkas: {summary['byte_duplicate_groups']:,}; "
        f"kelompok duplikasi PCM: {summary['pcm_duplicate_groups_count']:,}. "
        f"Konflik PCM dengan nama kelas berbeda dalam folder platform yang sama: "
        f"{len(summary['within_platform_pcm_label_conflicts'])}. "
        "Rincian path tersedia pada `duplicates.json`; konflik nama membutuhkan "
        "pemeriksaan label, bukan penggabungan otomatis.",
        "",
        f"Pembanding Harrison: {summary['harrison_files_checked']} WAV asli. "
        f"Kecocokan byte/PCM lengkap: {summary['harrison_exact_match_records']}. "
        "Pencarian salinan yang dipotong, diganti rate, atau ditransformasi "
        "terhadap Harrison belum dilakukan; tidak ada kecocokan persis "
        "bukan bukti independensi lengkap.",
        "",
        f"Penelusuran 36 kelas: {summary['excerpt_trace']['matched']} dari "
        f"{summary['excerpt_trace']['attempts']} potongan cocok dengan WAV "
        "mentah nominal pada |r|≥0,999; "
        f"{summary['excerpt_trace']['all_channels_exact']} cocok persis "
        "pada seluruh kanal. "
        f"Terdapat {summary['excerpt_trace']['overlapping_pairs']} pasangan "
        "interval potongan yang bertumpang tindih di induk yang sama. "
        "Korelasi adalah pemeriksaan asal waveform, bukan validasi event "
        "atau kebenaran nama tombol.",
        "",
        "![Asal potongan contoh a](asal-potongan.png)",
        "",
        "## Isi metadata",
        "",
        "| Folder | Berkas pada folder metadata | Tipe isi | Hit teks konteks rekaman |",
        "|---|---:|---|---:|",
    ]
    for platform in PLATFORMS:
        row = summary["platforms"][platform]
        lines.append(
            f"| {platform} | {row['metadata_files']} | {row['metadata_types']} | "
            f"{len(row['metadata_context_keyword_hits'])} |"
        )
    lines += [
        "",
        "Tipe isi dan pencarian kata konteks dicatat pada inventaris. "
        "Tidak menemukan kata sesi bukan bukti bahwa sesi fisik hanya satu; "
        "audit tidak mengarang ID sesi dari nomor potongan atau waktu arsip ZIP.",
        "",
        "## Kecukupan jumlah potongan yang disediakan",
        "",
        "K yang lolos syarat semua 36 kelas memiliki ≥K+T potongan PCM unik. "
        "T adalah jumlah potongan uji per kelas. Ini hanya pemeriksaan jumlah, "
        "belum menjamin ketukan berbeda atau pemisahan rekaman.",
        "",
        "| Folder | Minimum potongan unik/kelas | K jika T=1 | K jika T=5 | K jika T=10 |",
        "|---|---:|---|---|---|",
    ]
    for platform in PLATFORMS:
        row = summary["clip_budgets"][platform]
        feasible = [
            ", ".join(map(str, row["count_only_feasible_K"][str(t)])) or "tidak ada"
            for t in (1, 5, 10)
        ]
        lines.append(
            f"| {platform} | {row['minimum_unique_supplied_clips_per_key']} | "
            f"{' | '.join(feasible)} |"
        )
    lines += [
        "",
        "Jumlah ketukan valid dalam audio mentah belum ditetapkan melalui "
        "anotasi event. Jumlah file, puncak amplitudo, dan augmentasi tidak "
        "diperlakukan sebagai jumlah label ketukan independen.",
        "",
        "## Keputusan rancangan",
        "",
        summary["decision"]["explanation"],
        "",
        "Tidak ada pelatihan, pemilihan model, atau evaluasi akurasi MKA "
        "pada tahap ini. Audit dijalankan di CPU. Resep dan checkpoint "
        "Harrison sebelumnya tidak diubah.",
        "",
        "Protokol: `research/MKA_AUDIT_PROTOCOL.md`. Inventaris, CSV audio, "
        "duplikasi, penelusuran interval, dan hash keluaran tersedia bersama "
        "laporan ini. Snapshot riset menyimpan agregat dan keputusan.",
        "",
    ]
    (output / "laporan.md").write_text("\n".join(lines), encoding="utf-8")


def run(data, reference, output, snapshot, resume=False):
    if snapshot and snapshot.exists() and not (resume and output.exists()):
        raise ResearchError("Snapshot sudah ada; pilih snapshot baru atau resume audit yang sama.")
    archive = data / "MKA datasets.zip"
    source, fetched = read_json(data / "source.json"), read_json(data / "fetch.json")
    if (
        fetched["status"] != "complete"
        or source["snapshot"]["version"] != 3
        or source["snapshot"]["id"] != "bpt2hvf8n3"
        or fetched["source_metadata_sha256"] != file_hash(data / "source.json")
    ):
        raise ResearchError("Unduhan MKA v3 belum lengkap.")
    if file_hash(archive) != ARCHIVE_SHA256:
        raise ResearchError("Arsip tidak sesuai SHA256 pin penyedia.")
    signature = {
        "archive_sha256": ARCHIVE_SHA256,
        "metadata_sha256": file_hash(data / "source.json"),
        "fetch_manifest_sha256": file_hash(data / "fetch.json"),
        "module_sha256": file_hash(ROOT / "src/ketik_audio/mka.py"),
        "protocol_sha256": file_hash(ROOT / "research/MKA_AUDIT_PROTOCOL.md"),
    }
    if output.exists():
        if not resume or read_json(output / "config.json")["signature"] != signature:
            raise ResearchError("Direktori audit sudah ada atau signature berbeda.")
        if (output / "summary.json").exists():
            saved = read_json(output / "summary.json")
            if saved["status"] != "complete" or saved["signature"] != signature:
                raise ResearchError("Ringkasan audit selesai tidak sesuai signature.")
            if any(
                file_hash(output / name) != digest
                for name, digest in saved["artifact_sha256"].items()
            ):
                raise ResearchError("Artefak audit selesai berubah.")
            if snapshot:
                if snapshot.exists() and read_json(snapshot) != saved:
                    raise ResearchError("Snapshot tidak sama dengan audit selesai.")
                if not snapshot.exists():
                    write_json(snapshot, saved)
            print(f"ALREADY COMPLETE: {output / 'laporan.md'}", flush=True)
            return saved
    else:
        fresh_directory(output)
        write_json(output / "config.json", {"created_at": now(), "signature": signature})
    if resume and (output / "inventory-lock.json").exists():
        if read_json(output / "inventory-lock.json")["sha256"] != file_hash(
            output / "inventory.json"
        ):
            raise ResearchError("Inventaris tersimpan berubah.")
        inventory = read_json(output / "inventory.json")
    else:
        inventory = scan_archive(archive, output)
        write_json(output / "inventory-lock.json", {"sha256": file_hash(output / "inventory.json")})
    rows = inventory["files"]
    summary = summarise_inventory(rows)
    byte_groups = summary.pop("file_hash_duplicate_groups")
    pcm_groups = summary.pop("pcm_duplicate_groups")
    write_json(output / "duplicates.json", {"byte": byte_groups, "pcm": pcm_groups})
    traces = trace_segments(archive, rows, output)
    compared = compare_harrison(rows, reference, output)
    canonical = [r for r in rows if r["scope"] == "canonical" and "audio" in r]
    aggregate = [r for r in rows if r["scope"] == "aggregate" and "audio" in r]
    canonical_bytes = {r["sha256"] for r in canonical}
    canonical_pcm = {r["audio"]["pcm_sha256"] for r in canonical}
    labels = defaultdict(list)
    for row in canonical:
        labels[row["platform"], row["audio"]["pcm_sha256"]].append(row)
    conflicts = [
        {"platform": platform, "paths": [r["path"] for r in selected]}
        for (platform, _), selected in labels.items()
        if len({r["label"] for r in selected}) > 1
    ]
    one_per_key = all(
        all(n == 1 for n in summary["platforms"][p]["raw_files_per_common_key"].values())
        for p in PLATFORMS
    )
    decision = {
        "cross_recording_calibration_test_eligible": False,
        "status": "not_supported_by_supplied_raw_files"
        if one_per_key
        else "requires_metadata_review",
        "explanation": (
            "Untuk seluruh 36 kelas pada setiap folder platform, hanya ada satu WAV mentah "
            "per kelas. Potongan dan folder agregat tidak menyediakan rekaman target "
            "kedua untuk kalibrasi/uji lintas rekaman. Dengan struktur yang tersedia, "
            "MKA belum mendukung pertanyaan utama tersebut. Split dalam WAV atau "
            "evaluasi lintas dataset tanpa kalibrasi merupakan pertanyaan lain dan "
            "memerlukan protokol terpisah sebelum menilai model."
            if one_per_key
            else "Identitas kelompok rekaman dan sesi perlu ditinjau sebelum membuat split; "
            "jumlah berkas saja tidak menetapkan kelayakan evaluasi lintas rekaman."
        ),
    }
    summary.update(
        status="complete",
        created_at=now(),
        source="https://data.mendeley.com/datasets/bpt2hvf8n3/3",
        creators=["Karwan Mahdi Rawf", "Ayub Abdulrahman"],
        license=source["snapshot"]["licence"]["short_name"],
        signature=signature,
        archive_bytes=archive.stat().st_size,
        archive_sha256=ARCHIVE_SHA256,
        zip_entries=inventory["zip_entries"],
        file_count=inventory["file_count"],
        all_member_crcs_verified=inventory["all_member_crcs_verified"],
        audio_files=len(canonical) + len(aggregate),
        unique_audio_pcm=len({r["audio"]["pcm_sha256"] for r in rows if "audio" in r}),
        aggregate_audio_files=len(aggregate),
        aggregate_audio_with_canonical_byte_match=sum(
            r["sha256"] in canonical_bytes for r in aggregate
        ),
        aggregate_audio_with_canonical_pcm_match=sum(
            r["audio"]["pcm_sha256"] in canonical_pcm for r in aggregate
        ),
        byte_duplicate_groups=len(byte_groups),
        pcm_duplicate_groups_count=len(pcm_groups),
        within_platform_pcm_label_conflicts=conflicts,
        harrison_files_checked=len(compared["reference_files"]),
        harrison_exact_match_records=len(compared["exact_matches"]),
        excerpt_trace={
            "attempts": len(traces["traces"]),
            "matched": traces["matched"],
            "all_channels_exact": traces["all_channels_exact"],
            "overlapping_pairs": len(traces["overlapping_pairs"]),
        },
        session_metadata_verified=False,
        keystroke_event_annotations_verified=False,
        models_trained_or_evaluated=0,
        decision=decision,
    )
    with (output / "audio.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "path",
            "scope",
            "platform",
            "role",
            "label",
            "sha256",
            "sample_rate",
            "channels",
            "frames",
            "duration_seconds",
            "subtype",
            "pcm_sha256",
            "peak",
            "rms",
            "usable_waveform",
        )
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**r, **r["audio"]} for r in rows if "audio" in r)
    diagnostic_plot(archive, rows, traces, output)
    render(summary, output)
    summary["artifact_sha256"] = {
        p.name: file_hash(p)
        for p in sorted(output.iterdir())
        if p.is_file() and p.name != "summary.json"
    }
    write_json(output / "summary.json", summary)
    if snapshot:
        if snapshot.exists():
            raise ResearchError("Snapshot sudah ada.")
        write_json(snapshot, summary)
    print(f"COMPLETE: {output / 'laporan.md'}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/external/mka-v3")
    parser.add_argument("--harrison", type=Path, default=ROOT / "data/external/harrison-2023")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/mka-audit")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        run(args.data, args.harrison, args.output, args.snapshot, args.resume)
    except (ResearchError, OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
