"""Check whether MKA raw WAVs contain extra RIFF metadata chunks."""

from __future__ import annotations

import argparse
import struct
import zipfile
from pathlib import Path

from ketik_audio.common import ResearchError, file_hash, now, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def run(data, audit, output):
    inventory = audit / "inventory.json"
    if read_json(audit / "inventory-lock.json")["sha256"] != file_hash(inventory):
        raise ResearchError("Inventaris audit berubah.")
    if output.exists():
        raise ResearchError("Pemeriksaan header sudah ada; pilih berkas baru.")
    rows = read_json(inventory)["files"]
    checked = []
    with zipfile.ZipFile(data / "MKA datasets.zip") as source:
        for row in rows:
            if row["scope"] != "canonical" or row["role"] != "raw" or "audio" not in row:
                continue
            with source.open(row["path"]) as stream:
                header = stream.read(44)
            values = struct.unpack("<4sI4s4sIHHIIHH4sI", header)
            audio = row["audio"]
            plain = (
                values[0] == b"RIFF"
                and values[1] + 8 == row["bytes"]
                and values[2:4] == (b"WAVE", b"fmt ")
                and values[4:6] == (16, 1)
                and values[6] == audio["channels"]
                and values[7] == audio["sample_rate"]
                and values[8] == audio["sample_rate"] * audio["channels"] * 2
                and values[9] == audio["channels"] * 2
                and values[10:12] == (16, b"data")
                and values[12] == audio["frames"] * audio["channels"] * 2
                and values[12] + 44 == row["bytes"]
            )
            checked.append({"path": row["path"], "only_standard_fmt_and_data_chunks": plain})
    result = {
        "created_at": now(),
        "inventory_sha256": file_hash(inventory),
        "archive_sha256_from_verified_inventory": read_json(audit / "config.json")["signature"][
            "archive_sha256"
        ],
        "inspector_code_sha256": file_hash(Path(__file__)),
        "raw_files": len(checked),
        "only_standard_fmt_and_data_chunks": sum(
            r["only_standard_fmt_and_data_chunks"] for r in checked
        ),
        "session_metadata_in_extra_riff_chunks_found": False
        if all(r["only_standard_fmt_and_data_chunks"] for r in checked)
        else None,
        "files": checked,
    }
    write_json(output, result)
    print(
        f"Standard fmt/data only: {result['only_standard_fmt_and_data_chunks']}"
        f"/{len(checked)} raw WAVs"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/external/mka-v3")
    parser.add_argument("--audit", type=Path, default=ROOT / "reports/mka-audit")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/mka-headers/headers.json")
    args = parser.parse_args()
    try:
        run(args.data, args.audit, args.output)
    except (ResearchError, OSError, ValueError, struct.error) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
