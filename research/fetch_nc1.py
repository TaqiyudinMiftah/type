"""Fetch the public two-keyboard Novox NC1 corpus and pin its Zenodo files."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

from ketik_audio.common import ResearchError, file_hash, now, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/external/nc1-17536616"
RECORD = "https://zenodo.org/api/records/17536616"
PINS = {
    "readme.txt": (1079, "058c60443b1d7b508718f528cf26f0f6"),
    "dataset.zip": (174886324, "e77a82ab451348b93b2340b0efe5d030"),
}


def request(url):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "Ketik-Audio-Research/0.1"}),
        timeout=45,
    )


def check_file(path, size, checksum):
    with path.open("rb") as f:
        md5 = hashlib.file_digest(f, lambda: hashlib.md5(usedforsecurity=False)).hexdigest()
    if path.stat().st_size != size or md5 != checksum:
        raise ResearchError(f"Ukuran/checksum tidak cocok: {path.name}")


def run(output):
    meta = output / "record.json"
    if meta.exists():
        record = read_json(meta)
    else:
        if output.exists() and any(output.iterdir()):
            raise ResearchError("Tujuan berisi data tanpa provenance NC1.")
        with request(RECORD) as response:
            record = json.load(response)
    rows = {r["key"]: r for r in record["files"]}
    if str(record["id"]) != "17536616" or record["metadata"]["license"]["id"] != "cc-by-4.0":
        raise ResearchError("Versi atau lisensi NC1 berbeda dari pin.")
    for name, (size, checksum) in PINS.items():
        if rows[name]["size"] != size or rows[name]["checksum"] != f"md5:{checksum}":
            raise ResearchError("Metadata file NC1 berbeda dari pin.")
    if not meta.exists():
        write_json(meta, record)
    files = {}
    for name, (size, checksum) in PINS.items():
        target, part = output / name, output / f"{name}.part"
        url = f"https://zenodo.org/records/17536616/files/{urllib.parse.quote(name)}?download=1"
        if not target.exists():
            with request(url) as response, part.open("wb") as f:
                downloaded = 0
                while block := response.read(1024 * 1024):
                    downloaded += len(block)
                    if downloaded > size:
                        raise ResearchError("Unduhan NC1 melampaui ukuran yang dipatok.")
                    f.write(block)
            check_file(part, size, checksum)
            part.replace(target)
        else:
            check_file(target, size, checksum)
        files[name] = {"bytes": size, "md5": checksum, "sha256": file_hash(target), "url": url}
        print(f"NC1 {name}: {size:,} bytes terverifikasi", flush=True)
    result = {
        "created_at": now(),
        "status": "complete",
        "record_api": RECORD,
        "record_sha256": file_hash(meta),
        "doi": "10.5281/zenodo.17536616",
        "license": "CC BY 4.0",
        "files": files,
        "fetch_code_sha256": file_hash(Path(__file__)),
    }
    manifest = output / "fetch.json"
    if manifest.exists():
        saved = read_json(manifest)
        if any(saved.get(k) != v for k, v in result.items() if k != "created_at"):
            raise ResearchError("Provenance unduhan NC1 berubah.")
        return saved
    write_json(manifest, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        run(args.output)
    except (OSError, ResearchError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
