"""Fetch the small, MIT-licensed CoAtNet implementation cited in paper ref. 36."""

from pathlib import Path
from urllib.request import urlopen

from ketik_audio.common import ResearchError, file_hash, write_json

COMMIT = "c413dae1f053ac2bbff9e2d668e1d7a62f07c335"
ROOT = Path(__file__).resolve().parent / "vendor" / "external_attention"
FILES = {
    "LICENSE": "fab1a2eca11a5ac309b96e8e983a72e0ebdb1402e6725ebdbe1b4e4c9daf5192",
    "model/attention/CoAtNet.py": (
        "c89925a35ce36f9ad947083c45bee38c66cc1b07784c00e8267c3dcf9bc49f53"
    ),
    "model/attention/SelfAttention.py": (
        "5874b1e74d412d2fe7e785d8a590a2c31bac35880d65b5b85c2fd94aa241b9b3"
    ),
    "model/conv/MBConv.py": "b842ba165f6b6449975d3811ad79a0cab1578cc5053706f6c88ce2ad20eccec3",
}


def main():
    hashes = {}
    for name, expected in FILES.items():
        url = f"https://raw.githubusercontent.com/xmu-xiaoma666/External-Attention-pytorch/{COMMIT}/{name}"
        target = ROOT / name
        if not target.exists():
            with urlopen(url, timeout=30) as response:
                contents = response.read()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
        hashes[name] = file_hash(target)
        if hashes[name] != expected:
            raise ResearchError(f"Checksum referensi tidak cocok: {target}")
    write_json(
        ROOT / "provenance.json",
        {
            "repository": "https://github.com/xmu-xiaoma666/External-Attention-pytorch",
            "commit": COMMIT,
            "license": "MIT",
            "files": hashes,
        },
    )
    print(f"Referensi tersimpan: {ROOT}")


if __name__ == "__main__":
    main()
