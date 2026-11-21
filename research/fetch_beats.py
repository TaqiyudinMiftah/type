"""Fetch pinned BEATs sources and a checksum-verified public checkpoint mirror."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.request import Request, urlopen

from ketik_audio.common import ResearchError, file_hash, read_json, write_json
from ketik_audio.fewshot import BEATS_SHA256

ROOT = Path(__file__).resolve().parent
MIRROR_COMMIT = "5b53b0404df452a3a607d7e67687227730e5bad1"
NAME = "BEATs_iter3_plus_AS2M.pt"
ORIGINAL = "https://1drv.ms/u/s!AqeByhGUtINrgcpke6_lRSZEKD5j2Q?e=A3FpOf"
MIRROR = f"https://huggingface.co/lpepino/beats_ckpts/resolve/{MIRROR_COMMIT}/{NAME}"


def fetch(output: Path):
    vendor = ROOT / "vendor/beats"
    provenance = read_json(vendor / "provenance.json")
    for name, entry in provenance["files"].items():
        path = vendor / name
        if not path.exists():
            with urlopen(
                Request(entry["url"], headers={"User-Agent": "ketik-audio"}), timeout=30
            ) as r:
                contents = r.read()
            import hashlib

            if hashlib.sha256(contents).hexdigest() != entry["upstream_sha256"]:
                raise ResearchError(f"Checksum upstream berubah: {name}")
            if name == "BEATs.py":
                contents = contents.replace(
                    b"import torchaudio.compliance.kaldi as ta_kaldi", b"import kaldi as ta_kaldi"
                )
            elif name == "kaldi.py":
                contents = contents.replace(
                    b"import torchaudio\n",
                    b"# torchaudio native import omitted; only pure-PyTorch fbank is used.\n",
                )
                contents = contents.replace(
                    b"    dct_matrix = torchaudio.functional.create_dct",
                    b'    raise NotImplementedError("Vendored Kaldi adapter supports fbank only")\n'
                    b"    dct_matrix = torchaudio.functional.create_dct",
                )
            path.write_bytes(contents)
        if file_hash(path) != entry["sha256"]:
            raise ResearchError(f"Kode vendor berbeda dari revisi yang dipatok: {name}")
    output.mkdir(parents=True, exist_ok=True)
    target = output / NAME
    if not target.exists():
        temporary = target.with_suffix(".download")
        start, size = time.perf_counter(), 0
        with (
            urlopen(Request(MIRROR, headers={"User-Agent": "ketik-audio"}), timeout=60) as response,
            temporary.open("wb") as stream,
        ):
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                size += len(chunk)
                if size % (32 * 1024 * 1024) == 0:
                    print(f"{size / 1e6:.0f} MB; {time.perf_counter() - start:.1f} s", flush=True)
        if file_hash(temporary) != BEATS_SHA256:
            raise ResearchError("Checkpoint mirror gagal verifikasi SHA256.")
        temporary.rename(target)
    if file_hash(target) != BEATS_SHA256:
        raise ResearchError(f"Checkpoint berubah: {target}")
    write_json(
        output / "provenance.json",
        {
            "model": "BEATs_iter3_plus_AS2M",
            "original_url": ORIGINAL,
            "original_access": "403 during pilot on 2026-11-07; pinned mirror used",
            "mirror_url": MIRROR,
            "mirror_commit": MIRROR_COMMIT,
            "sha256": BEATS_SHA256,
            "digest_authority": "Hugging Face LFS object ID; no Microsoft digest available",
            "bytes": target.stat().st_size,
            "source_readme": "https://github.com/microsoft/unilm/blob/master/beats/README.md",
        },
    )
    print(f"BEATs checkpoint and sources verified: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/external/beats"))
    args = parser.parse_args()
    try:
        fetch(args.output)
    except (ResearchError, OSError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
