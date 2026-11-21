"""Fetch the pinned public HEAR on-device archive and verify the release digest."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from ketik_audio.common import ResearchError, file_hash, now, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/external/hear-v1"
RELEASE = "https://api.github.com/repos/vikrant-vikram/Hear-Dataset/releases/tags/v1"
URL = "https://github.com/vikrant-vikram/Hear-Dataset/releases/download/v1/HEAR.zip"
SIZE = 768636949
SHA256 = "4b0f1908d0d1498ed448577a500acd3a0ad24c0362ba37558034660516b1dcf6"
ASSET_ID = 484738918


def request(url, offset=0):
    headers = {"User-Agent": "Ketik-Audio-Research/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30)


def fetch(output):
    metadata = output / "release.json"
    if metadata.exists():
        record = read_json(metadata)
    else:
        with request(RELEASE) as response:
            record = json.load(response)
    assets = [a for a in record["assets"] if a["id"] == ASSET_ID]
    if len(assets) != 1 or record["tag_name"] != "v1" or record["id"] != 357383468:
        raise ResearchError("Rilis HEAR berbeda dari pin.")
    asset = assets[0]
    if (
        asset["name"] != "HEAR.zip"
        or asset["size"] != SIZE
        or asset["digest"] != f"sha256:{SHA256}"
        or asset["browser_download_url"] != URL
        or asset["state"] != "uploaded"
    ):
        raise ResearchError("Metadata arsip HEAR berbeda dari pin.")
    if not metadata.exists():
        if output.exists() and any(output.iterdir()):
            raise ResearchError("Tujuan berisi data tanpa provenance HEAR.")
        write_json(metadata, record)
    target, part = output / "HEAR.zip", output / "HEAR.zip.part"
    if not target.exists():
        for attempt in range(3):
            offset = part.stat().st_size if part.exists() else 0
            if offset > SIZE:
                raise ResearchError("Unduhan parsial lebih besar dari arsip yang dipatok.")
            if offset == SIZE:
                break
            try:
                with request(URL, offset) as response:
                    if offset and response.status == 206:
                        if (
                            response.headers.get("Content-Range")
                            != f"bytes {offset}-{SIZE - 1}/{SIZE}"
                        ):
                            raise ResearchError("Range unduhan HEAR tidak cocok.")
                    elif response.status == 200:
                        offset = 0
                    else:
                        raise ResearchError(f"Status unduhan HEAR: {response.status}")
                    tick = time.monotonic()
                    with part.open("ab" if offset else "wb") as stream:
                        while block := response.read(1024 * 1024):
                            stream.write(block)
                            offset += len(block)
                            if offset > SIZE:
                                raise ResearchError("Unduhan HEAR lebih besar dari pin.")
                            if time.monotonic() - tick >= 10:
                                print(f"HEAR.zip: {offset / SIZE:.1%}", flush=True)
                                tick = time.monotonic()
                if offset != SIZE:
                    raise OSError("Unduhan terhenti sebelum ukuran yang dipatok.")
                break
            except OSError:
                if attempt == 2:
                    raise
                print("Unduhan terputus; melanjutkan berkas parsial.", flush=True)
        verify_archive(part)
        part.replace(target)
    else:
        verify_archive(target)
    result = {
        "created_at": now(),
        "status": "complete",
        "release": record["html_url"],
        "release_api": RELEASE,
        "release_metadata_sha256": file_hash(metadata),
        "asset_id": ASSET_ID,
        "url": URL,
        "bytes": SIZE,
        "sha256": SHA256,
        "scope_from_release": "on-device recordings; mobile phone and VoIP not included",
        "fetch_code_sha256": file_hash(Path(__file__)),
    }
    if (output / "fetch.json").exists():
        previous = read_json(output / "fetch.json")
        if any(previous.get(k) != v for k, v in result.items() if k != "created_at"):
            raise ResearchError("Provenance unduhan HEAR berubah.")
        result = previous
    else:
        write_json(output / "fetch.json", result)
    print(f"HEAR.zip terverifikasi: {SIZE:,} bytes, SHA256 {SHA256}", flush=True)
    return result


def verify_archive(path):
    if path.stat().st_size != SIZE or file_hash(path) != SHA256:
        raise ResearchError("Ukuran/SHA256 arsip HEAR tidak cocok.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        fetch(args.output)
    except (OSError, ResearchError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
