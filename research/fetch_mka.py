"""Fetch the pinned public MKA v3 files and verify provider SHA256 checksums."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from ketik_audio.common import ResearchError, file_hash, now, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://data.mendeley.com/datasets/bpt2hvf8n3/3"
FILES = "https://data.mendeley.com/public-api/datasets/bpt2hvf8n3/files?folder_id=root&version=3"
PINNED = {
    "MKA datasets.zip": (
        1997707877,
        "01fad57bf972782aa1cab74602e1965c0e489d7dceb37e36668a1d44f95ecb64",
    ),
    "MKA flowchart.bmp": (
        1469574,
        "f12a0c1a095107ea4380b50fc2927cb803520abcb535f237d3fffff81e073e38",
    ),
}


def request(url, offset=0):
    headers = {"User-Agent": "Ketik-Audio-Research/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45)


def metadata(output):
    path = output / "source.json"
    if path.exists():
        result = read_json(path)
        if result["source"] != SOURCE:
            raise ResearchError("Direktori berisi sumber lain.")
        return result
    if output.exists() and any(output.iterdir()):
        raise ResearchError("Direktori bukan unduhan MKA yang dapat dipulihkan.")
    with request(SOURCE) as response:
        html = response.read().decode("utf-8")
    match = re.search(r"window\.INITIAL_STATE\s*=\s*", html)
    if not match:
        raise ResearchError("Metadata halaman Mendeley berubah.")
    state, _ = json.JSONDecoder().raw_decode(html[match.end() :])
    snapshot = state["dataset"]["snapshot"]
    with request(FILES) as response:
        files = json.loads(response.read())
    found = {f["filename"]: (f["size"], f["content_details"]["sha256_hash"]) for f in files}
    if found != PINNED or snapshot["version"] != 3 or snapshot["id"] != "bpt2hvf8n3":
        raise ResearchError("Versi/daftar/checksum publik berbeda dari pin MKA v3.")
    result = {
        "created_at": now(),
        "source": SOURCE,
        "files_api": FILES,
        "snapshot": snapshot,
        "files": files,
    }
    write_json(path, result)
    return result


def download(output, row):
    name, details = row["filename"], row["content_details"]
    if Path(name).name != name or name not in PINNED:
        raise ResearchError("Nama berkas unduhan tidak sesuai pin.")
    size, expected = PINNED[name]
    if row["size"] != size or details["sha256_hash"] != expected:
        raise ResearchError("Metadata unduhan tidak sesuai pin.")
    target, partial = output / name, output / f"{name}.part"
    if target.exists():
        if target.stat().st_size != size or file_hash(target) != expected:
            raise ResearchError(f"Berkas selesai berubah: {target}")
    else:
        for attempt in range(3):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > size:
                raise ResearchError("Berkas parsial melebihi ukuran yang dipublikasikan.")
            if offset == size:
                break
            try:
                with request(details["download_url"], offset) as response:
                    if offset and response.status == 206:
                        if not response.headers.get("Content-Range", "").startswith(
                            f"bytes {offset}-"
                        ):
                            raise ResearchError("Offset resume server berbeda.")
                    elif response.status == 200:
                        offset = 0
                    else:
                        raise ResearchError(f"Status unduhan tidak diharapkan: {response.status}")
                    last_log = time.monotonic()
                    with partial.open("ab" if offset else "wb") as stream:
                        while block := response.read(1024 * 1024):
                            stream.write(block)
                            offset += len(block)
                            if offset > size:
                                raise ResearchError("Unduhan melebihi ukuran pin.")
                            if time.monotonic() - last_log >= 15:
                                print(
                                    f"{name}: {offset / size:.1%}, {offset:,}/{size:,} bytes",
                                    flush=True,
                                )
                                last_log = time.monotonic()
                if offset != size:
                    raise OSError("Unduhan berakhir sebelum lengkap.")
                break
            except urllib.error.HTTPError:
                raise
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
                print("Koneksi terputus; melanjutkan berkas parsial.", flush=True)
                time.sleep(2)
        if file_hash(partial) != expected:
            raise ResearchError(f"SHA256 penyedia tidak cocok: {partial}")
        partial.replace(target)
    print(f"VERIFIED {name}: {size:,} bytes SHA256={expected}", flush=True)
    return {"filename": name, "bytes": size, "sha256": expected, "verified_at": now()}


def run(output):
    record = metadata(output)
    verified = []
    # Fetch the small author diagram first so it can be inspected during the archive transfer.
    for row in sorted(record["files"], key=lambda r: r["size"]):
        verified.append(download(output, row))
        write_json(
            output / "fetch.json",
            {"source": SOURCE, "status": "downloading", "verified_files": verified},
        )
    result = {
        "source": SOURCE,
        "status": "complete",
        "completed_at": now(),
        "source_metadata_sha256": file_hash(output / "source.json"),
        "fetch_code_sha256": file_hash(Path(__file__)),
        "verified_files": verified,
    }
    write_json(output / "fetch.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/external/mka-v3")
    args = parser.parse_args()
    try:
        run(args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
