"""Decode the long-text ooTextFile Sound 2 objects found in the MKA release."""

from __future__ import annotations

import re

import numpy as np

from .common import ResearchError


def read_praat_sound(blob):
    text = blob.decode("utf-8-sig")
    if not re.search(r'Object class\s*=\s*"Sound 2"', text) or not text.startswith(
        'File type = "ooTextFile"'
    ):
        raise ResearchError("Hanya format teks panjang Praat Sound 2 yang didukung.")

    def field(name):
        found = re.findall(rf"^{name}\s*=\s*([^\r\n]+)", text, re.MULTILINE)
        if len(found) != 1:
            raise ResearchError(f"Field Praat {name} hilang/berulang.")
        return found[0]

    frames, channels, step = int(field("nx")), int(field("ny")), float(field("dx"))
    if frames <= 0 or channels <= 0 or not np.isfinite(step) or step <= 0:
        raise ResearchError("Dimensi/waktu Praat tidak valid.")
    rate = round(1 / step)
    if rate <= 0 or abs(rate * step - 1) > 1e-9:
        raise ResearchError("Rate Praat tidak mendekati bilangan bulat.")
    values, seen = (
        np.empty((frames, channels), dtype=np.float64),
        np.zeros((frames, channels), bool),
    )
    pattern = r"^[ \t]*z\s*\[(\d+)\]\s*\[(\d+)\]\s*=\s*([^\r\n]+)"
    for match in re.finditer(pattern, text, re.MULTILINE):
        channel, frame = int(match[1]) - 1, int(match[2]) - 1
        if not (0 <= channel < channels and 0 <= frame < frames) or seen[frame, channel]:
            raise ResearchError("Indeks sampel Praat berulang/di luar dimensi.")
        values[frame, channel] = float(match[3])
        seen[frame, channel] = True
    if not seen.all() or not np.isfinite(values).all():
        raise ResearchError("Sampel Praat tidak lengkap/non-finite.")
    return values, rate
