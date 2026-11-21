"""Score a held-out Lenovo LOQ Uji pertama package with the frozen source models."""

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from ketik_audio.common import (
    FeatureConfig,
    ResearchError,
    file_hash,
    fresh_directory,
    read_json,
    write_json,
)
from ketik_audio.skaid_pilot_features import mfcc_features
from ketik_audio.web_collection import import_session
from research.run_loq_transfer import (
    PROTOCOL,
    RUN,
    SCHEMA,
    SOURCE,
    metadata,
    microphone,
    recording,
    saved_models,
    targets,
    waves,
)


def check_query(source, query):
    if source["schema"] != SCHEMA or query["schema"] != SCHEMA:
        raise ValueError("Gunakan paket kuota 27 kelas untuk Uji pertama.")
    if source["role"] != "calibration" or query["role"] != "test1":
        raise ValueError("Kirim paket Uji pertama dari studi kalibrasi Lenovo LOQ.")
    for name in ("participant_id", "study_id"):
        if source[name] != query[name]:
            raise ValueError("Profil peserta dan studi harus sama dengan paket kalibrasi.")
    if source["session_id"] == query["session_id"] or any(
        source["files"][name] == query["files"][name] for name in ("audio.wav", "events.jsonl")
    ):
        raise ValueError("Paket Uji pertama harus memiliki audio dan log ketukan tersendiri.")
    if source["local_day"] >= query["local_day"]:
        raise ValueError("Ambil Uji pertama pada hari setelah kalibrasi.")
    setup_source, setup_query = (meta["study"]["setup"] for meta in (source, query))
    for name in ("laptop", "keyboard"):
        if setup_source[name].strip().casefold() != setup_query[name].strip().casefold():
            raise ValueError("Uji pertama harus memakai laptop dan keyboard yang sama.")
    if microphone(source) != microphone(query):
        raise ValueError("Uji pertama harus memakai mikrofon yang sama dengan kalibrasi.")


def frozen_scores(query):
    prepared = RUN / "prepared"
    source_dataset = read_json(prepared / "dataset.json")
    checkpoints = read_json(RUN / "models/results.json")
    if file_hash(prepared / "dataset.json") != checkpoints["dataset_sha256"]:
        raise ValueError("Manifest dataset sumber berubah.")
    wave = waves(query, query.rows)
    config = FeatureConfig.from_dict(source_dataset["cnn_features"])
    _, scores = saved_models(wave, mfcc_features(wave), targets(query.rows), config, checkpoints)
    return scores


def evaluate(package, out):
    with TemporaryDirectory(prefix="ketik-loq-test1-") as scratch:
        directory = Path(scratch) / "query"
        imported = import_session(package, directory)
        source, query_meta = metadata(SOURCE), metadata(directory)
        check_query(source, query_meta)
        fresh_directory(out)
        write_json(
            out / "plan.json",
            {
                "source": source["session_id"],
                "query": query_meta["session_id"],
                "query_zip_sha256": imported["source_sha256"],
                "query_wav_sha256": query_meta["files"]["audio.wav"],
                "held_out": True,
                "target_fit_labels": 0,
                "output_classes": 27,
                "source_results_sha256": file_hash(RUN / "models/results.json"),
                "protocol_sha256": file_hash(PROTOCOL),
                "transfer_script_sha256": file_hash(
                    Path(__file__).with_name("run_loq_transfer.py")
                ),
                "first_session_script_sha256": file_hash(Path(__file__)),
            },
        )
        query = recording(directory, query_meta, out, "query")
        if not query.rows:
            raise ValueError("Semua potongan tersaring. Lihat query-coverage.json.")
        scores = frozen_scores(query)
        write_json(
            out / "results.json",
            {
                "status": "measured",
                "query_day": query_meta["local_day"],
                "target_fit_labels": 0,
                "classes_with_query_data": len({row["key"] for row in query.rows}),
                "query_timing": query.timing["status"],
                "uniform_chance": 1 / 27,
                "pretrained_models": scores,
                "hardware": "CPU",
            },
        )
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        scores = evaluate(args.package, args.output)
    except (ResearchError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    ridge = scores["ridge"]
    print(f"Uji pertama K0: Ridge {ridge['correct']}/{ridge['n']} ketukan. {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
