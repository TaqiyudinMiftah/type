"""Check BEATs fbank compatibility against the publication-era official release."""

import argparse
import hashlib
import sys
import types
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch

from ketik_audio.common import ResearchError, file_hash, write_json

REFERENCE = "https://raw.githubusercontent.com/pytorch/audio/v0.13.0/torchaudio/compliance/kaldi.py"


def check(prepared: Path, output: Path):
    torch.set_num_threads(4)
    with urlopen(REFERENCE, timeout=30) as response:
        source = response.read()
    reference = types.ModuleType("kaldi_reference_013")
    # fbank is pure torch; only the unused MFCC function needs torchaudio.
    standalone = source.decode().replace("import torchaudio\n", "")
    exec(compile(standalone, REFERENCE, "exec"), reference.__dict__)
    vendor = Path(__file__).resolve().parent / "vendor/beats"
    sys.path.insert(0, str(vendor))
    import kaldi

    data = np.load(prepared / "segments.npz", allow_pickle=False)
    comparisons = []
    for domain in ["phone", "zoom"]:
        for index in [0, 224, 424, 899]:
            wave = torch.from_numpy(data[f"{domain}_wave"][index][None]) * 32768
            args = {
                "num_mel_bins": 128,
                "sample_frequency": 16000,
                "frame_length": 25,
                "frame_shift": 10,
            }
            expected = reference.fbank(wave, **args)
            actual = kaldi.fbank(wave, **args)
            difference = float((expected - actual).abs().max())
            if not torch.equal(expected, actual):
                raise ResearchError(f"Perbedaan fbank: {domain}/{index}: {difference}")
            comparisons.append({"domain": domain, "index": index, "max_abs_difference": difference})
    write_json(
        output,
        {
            "reference_url": REFERENCE,
            "purpose": (
                "Pure torch fbank compatibility with official torchaudio v0.13.0; no model tuning"
            ),
            "upstream_sha256": hashlib.sha256(source).hexdigest(),
            "standalone_sha256": hashlib.sha256(standalone.encode()).hexdigest(),
            "vendored_sha256": file_hash(vendor / "kaldi.py"),
            "comparisons": comparisons,
        },
    )
    print(f"All {len(comparisons)} real-clip fbanks match exactly: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.prepared, args.output)
