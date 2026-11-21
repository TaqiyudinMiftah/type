"""Fetch pinned public SKAID audio/demographics; preserve earlier log catalogue."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from ketik_audio.common import ResearchError, file_hash, now, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
RECORD = "https://zenodo.org/api/records/17282184"
PINS = {
    "Participant Recordings_zip.zip": (
        "recordings.zip",
        182788356,
        "e4c9fd0edf4586ac99a886ecaa15c275",
    ),
    "all_participants_demographics.csv": (
        "demographics.csv",
        1585,
        "8a3fdb18d7ca72defa0ce6e39d85e4a1",
    ),
}


def request(url, offset=0):
    headers = {"User-Agent": "Ketik-Audio-Research/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45)


def md5(path):
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(output):
    path = output / "source.json"
    if path.exists():
        saved = read_json(path)
    else:
        if output.exists() and any(output.iterdir()):
            raise ResearchError("Direktori tidak berisi unduhan SKAID yang dikenal.")
        with request(RECORD) as response:
            saved = {"fetched_at": now(), "source": RECORD, "record": json.load(response)}
    record = saved["record"]
    if (
        str(record["id"]) != "17282184"
        or record["metadata"].get("version") != "V 1.0"
        or record["metadata"].get("license", {}).get("id") != "cc-by-4.0"
    ):
        raise ResearchError("Versi/lisensi metadata berbeda dari pin.")
    rows = {r["key"]: r for r in record["files"]}
    for name, (_, size, checksum) in PINS.items():
        if rows[name]["size"] != size or rows[name]["checksum"] != f"md5:{checksum}":
            raise ResearchError("Ukuran/checksum berkas publik berubah.")
    if not path.exists():
        write_json(path, saved)
    return rows


def download(output, row):
    local, size, checksum = PINS[row["key"]]
    target, partial = output / local, output / f"{local}.part"
    if not target.exists():
        for attempt in range(3):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > size:
                raise ResearchError("Ukuran berkas parsial melebihi pin.")
            if offset == size:
                break
            try:
                with request(row["links"]["self"], offset) as response:
                    if offset and response.status == 206:
                        if not response.headers.get("Content-Range", "").startswith(
                            f"bytes {offset}-"
                        ):
                            raise ResearchError("Offset resume server berbeda.")
                    elif response.status == 200:
                        offset = 0
                    else:
                        raise ResearchError(f"Status unduhan: {response.status}")
                    tick = time.monotonic()
                    with partial.open("ab" if offset else "wb") as stream:
                        while block := response.read(1024 * 1024):
                            stream.write(block)
                            offset += len(block)
                            if offset > size:
                                raise ResearchError("Unduhan lebih besar dari pin.")
                            if time.monotonic() - tick >= 10:
                                print(
                                    f"{local}: {offset / size:.1%}, {offset:,}/{size:,} bytes",
                                    flush=True,
                                )
                                tick = time.monotonic()
                if offset != size:
                    raise OSError("Unduhan terputus sebelum lengkap.")
                break
            except urllib.error.HTTPError:
                raise
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
                print("Koneksi terputus; melanjutkan berkas parsial", flush=True)
                time.sleep(2)
        if partial.stat().st_size != size or md5(partial) != checksum:
            raise ResearchError("Checksum penyedia tidak cocok; berkas parsial dipertahankan.")
        partial.replace(target)
    if target.stat().st_size != size or md5(target) != checksum:
        raise ResearchError("Berkas selesai berbeda dari pin.")
    digest = file_hash(target)
    print(f"VERIFIED {local}: {size:,} bytes SHA256={digest}", flush=True)
    return {
        "source_name": row["key"],
        "local_name": local,
        "bytes": size,
        "provider_checksum": f"md5:{checksum}",
        "sha256": digest,
        "url": row["links"]["self"],
        "verified_at": now(),
    }


def run(output):
    rows = metadata(output)
    verified = []
    for name in sorted(PINS, key=lambda n: PINS[n][1]):
        verified.append(download(output, rows[name]))
        write_json(output / "fetch.json", {"status": "downloading", "files": verified})
    result = {
        "created_at": now(),
        "status": "complete",
        "source": RECORD,
        "license": "CC BY 4.0",
        "source_metadata_sha256": file_hash(output / "source.json"),
        "fetch_code_sha256": file_hash(Path(__file__)),
        "files": verified,
    }
    write_json(output / "fetch.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/external/skaid-v1")
    args = parser.parse_args()
    try:
        run(args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
