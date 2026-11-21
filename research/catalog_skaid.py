"""Count SKAID labels before committing to an audio adaptation protocol."""

from __future__ import annotations

import argparse
import csv
import io
import string
import zipfile
from collections import Counter
from decimal import Decimal
from pathlib import Path

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, write_json

ROOT = Path(__file__).resolve().parents[1]
KS = (1, 2, 3, 5, 10)
VOCABULARIES = {
    "letters26": string.ascii_lowercase,
    "alphanumeric36": string.ascii_lowercase + string.digits,
}
FIELDS = {"Session ID", "Timestamp (ms)", "Event", "Key", "Phase"}


def catalogue(archive: Path) -> dict:
    logs, events, phases = [], Counter(), Counter()
    with zipfile.ZipFile(archive) as source:
        names = sorted(
            n for n in source.namelist() if n.endswith(".csv") and not n.endswith("_full_text.csv")
        )
        if not names:
            raise ResearchError("Tidak ada log event SKAID.")
        for name in names:
            reader = csv.DictReader(io.StringIO(source.read(name).decode("utf-8-sig")))
            if set(reader.fieldnames or ()) != FIELDS:
                raise ResearchError(f"Kolom log berubah: {name}")
            counts = {"Phase 1": Counter(), "Phase 2": Counter()}
            sessions, timestamps, encodings = set(), [], Counter()
            for row in reader:
                event, phase = row["Event"], row["Phase"]
                if event not in {"press", "release"} or phase not in counts:
                    raise ResearchError(f"Event/fase tidak dikenal: {name}")
                sessions.add(row["Session ID"])
                timestamp_text = row["Timestamp (ms)"]
                timestamp = Decimal(timestamp_text)
                if not timestamp.is_finite() or timestamp != timestamp.to_integral_value():
                    raise ResearchError(f"Timestamp bukan milidetik bulat: {name}")
                timestamps.append(int(timestamp))
                encodings["scientific" if "e" in timestamp_text.lower() else "integer"] += 1
                events[event] += 1
                phases[phase] += 1
                key = row["Key"].lower()
                # Uppercase and lowercase refer to the same physical letter here.
                if event == "press" and len(key) == 1 and key in VOCABULARIES["alphanumeric36"]:
                    counts[phase][key] += 1
            logs.append(
                {
                    "file": name,
                    "session_ids": len(sessions),
                    "events": len(timestamps),
                    "nondecreasing_timestamps": all(
                        a <= b for a, b in zip(timestamps, timestamps[1:], strict=False)
                    ),
                    "timestamp_encodings": dict(encodings),
                    "timestamp_unique_count": len(set(timestamps)),
                    "timestamp_span_ms": max(timestamps) - min(timestamps) if timestamps else 0,
                    "counts": {phase: dict(sorted(c.items())) for phase, c in counts.items()},
                }
            )
    eligibility = {}
    for vocabulary, alphabet in VOCABULARIES.items():
        eligibility[vocabulary] = {
            "classes": alphabet,
            "eligible_logs_phase1_K_and_phase2_one_per_class": {
                str(k): sum(
                    all(log["counts"]["Phase 1"].get(c, 0) >= k for c in alphabet)
                    and all(log["counts"]["Phase 2"].get(c, 0) >= 1 for c in alphabet)
                    for log in logs
                )
                for k in KS
            },
            "phase1_min_class_presses_per_log": [
                min(log["counts"]["Phase 1"].get(c, 0) for c in alphabet) for log in logs
            ],
        }
    return {
        "created_at": now(),
        "source": "https://zenodo.org/records/17282184",
        "archive_sha256": file_hash(archive),
        "catalogue_code_sha256": file_hash(Path(__file__)),
        "count_only_assessment": True,
        "audio_alignment_verified": False,
        "recording_pairing_verified": False,
        "participant_independence_verified": False,
        "log_files": len(logs),
        "events": dict(events),
        "phases": dict(phases),
        "logs_with_scientific_timestamps": sum(
            bool(log["timestamp_encodings"].get("scientific")) for log in logs
        ),
        "logs_with_only_one_unique_timestamp": sum(
            log["timestamp_unique_count"] == 1 for log in logs
        ),
        "eligibility": eligibility,
        "logs": logs,
    }


def run(archive: Path, output: Path) -> dict:
    result = catalogue(archive)
    fresh_directory(output)
    write_json(output / "catalogue.json", result)
    lines = [
        "# Kelayakan label SKAID: audit hitungan log",
        "",
        f"{result['log_files']} berkas log event, "
        f"{result['events']['press']:,} press dan {result['events']['release']:,} release. "
        "CSV full_text tidak dibaca. Huruf besar/kecil disatukan sebagai tombol fisik; "
        "Key.space dan tombol khusus tidak masuk kosakata.",
        "",
        "Syarat kelayakan per log: setiap kelas memiliki ≥K press pada Phase 1 "
        "untuk kalibrasi dan ≥1 press pada Phase 2 untuk uji. Ini hanya "
        "memeriksa jumlah label, belum memastikan potongan audio valid/terpisah.",
        "",
        "| K | Log memenuhi 26 huruf | Log memenuhi 26 huruf +10 digit |",
        "|---:|---:|---:|",
    ]
    for k in KS:
        eligible = [
            result["eligibility"][v]["eligible_logs_phase1_K_and_phase2_one_per_class"][str(k)]
            for v in VOCABULARIES
        ]
        lines.append(f"| {k} | {eligible[0]} | {eligible[1]} |")
    lines += [
        "",
        f"{result['logs_with_scientific_timestamps']} log mengandung timestamp "
        "notasi ilmiah; "
        f"{result['logs_with_only_one_unique_timestamp']} log hanya mempunyai satu "
        "nilai timestamp unik. Notasi ilmiah sendiri tidak membuktikan hilangnya "
        "presisi, tetapi timestamp yang sama sepanjang rekaman tidak cukup untuk "
        "alignment event dalam milidetik tanpa sumber waktu lain.",
        "",
        "Tidak mengartikan satu berkas/session ID sebagai satu partisipan independen. "
        "Belum memverifikasi pasangan rekaman, offset/drift audio-log, auto-repeat, "
        "duplikasi, overlap jendela, maupun metadata keyboard. Mengurangi kosakata "
        "atau menggabungkan sesi mengubah pertanyaan riset dan perlu protokol baru. "
        "Pengulangan sampel tidak menambah jumlah label unik.",
        "",
        "Sumber dataset: [SKAID, Zenodo](https://zenodo.org/records/17282184). "
        "Jumlah pada tabel berasal dari analisis arsip log lokal, bukan klaim "
        "evaluasi model. Tidak ada pelatihan SKAID pada tahap ini.",
        "",
        f"SHA256 log: `{result['archive_sha256']}`.",
        "",
    ]
    (output / "kelayakan.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Katalog hitungan tersimpan: {output / 'kelayakan.md'}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=ROOT / "data/external/skaid-catalog/logs.zip"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/skaid-feasibility")
    args = parser.parse_args()
    try:
        run(args.archive, args.output)
    except (ResearchError, OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
