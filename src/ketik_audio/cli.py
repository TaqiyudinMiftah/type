"""User-facing command line. Heavy dependencies are loaded per command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .common import ResearchError, write_json


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="ketik",
        description=(
            "Riset lokal pengenalan a–z dari suara keyboard: rekam → siapkan → latih → evaluasi."
        ),
    )
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Periksa mikrofon, CPU, dan tes GPU.")
    doctor.add_argument("--json", action="store_true", help="Cetak hasil lengkap sebagai JSON.")
    doctor.add_argument("--output", type=Path, help="Simpan pemeriksaan ke JSON baru.")
    record = commands.add_parser("record", help="Rekam sesi berlabel di terminal Linux.")
    record.add_argument("--data", type=Path, default=Path("data"))
    record.add_argument("--session", type=int, required=True, choices=range(1, 7))
    record.add_argument("--per-key", type=int, default=25)
    record.add_argument("--device", type=int, help="ID mikrofon dari doctor.")
    record.add_argument("--seed", type=int, default=42)
    record.add_argument(
        "--retry", action="store_true", help="Arsipkan sesi gagal lalu rekam ulang."
    )
    web_import = commands.add_parser(
        "web-import", help="Impor dan audit ZIP dari web kolektor; terpisah dari dataset terminal."
    )
    web_import.add_argument("package", type=Path)
    web_import.add_argument("--output", type=Path, required=True)
    web_align = commands.add_parser(
        "web-align", help="Audit fit timing robust v2; pertahankan hasil impor v1."
    )
    web_align.add_argument("package", type=Path)
    web_align.add_argument("--output", type=Path, required=True)
    web_align.add_argument(
        "--evaluation-role",
        choices=["development", "validation", "negative_control", "unclassified"],
        default="unclassified",
    )
    prepare = commands.add_parser("prepare", help="Siapkan fitur sesi 1–5; sesi 6 tidak dibaca.")
    prepare.add_argument("--data", type=Path, default=Path("data"))
    prepare.add_argument("--output", type=Path, help="Default: <data>/processed.")
    train = commands.add_parser("train", help="Latih SVM atau CNN memakai split sesi tetap.")
    train.add_argument("kind", choices=["svm", "cnn"])
    train.add_argument("--dataset", type=Path, default=Path("data/processed"))
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", choices=["cpu", "gpu", "auto"], default="auto")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    evaluate = commands.add_parser("evaluate", help="Uji model pada sesi 6 yang ditahan.")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, default=Path("data"))
    evaluate.add_argument("--output", type=Path, help="Default: <model>/evaluation.")
    evaluate.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    predict = commands.add_parser("predict", help="Prediksi urutan huruf dari WAV tanpa label.")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--audio", type=Path, required=True)
    predict.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    predict.add_argument("--output", type=Path, help="Simpan hasil JSON baru.")
    report = commands.add_parser("report", help="Buat laporan Markdown dan grafik PNG.")
    report.add_argument("--models", type=Path, nargs="+", required=True)
    report.add_argument(
        "--evaluations",
        type=Path,
        nargs="+",
        help="Direktori evaluasi sesuai urutan --models, jika bukan default.",
    )
    report.add_argument("--output", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            from .devices import doctor

            if args.output and args.output.exists():
                raise ResearchError(f"File sudah ada: {args.output}")
            result = doctor()
            if args.output:
                write_json(args.output, result)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(f"Perekaman: {'siap' if result['recording_ready'] else 'belum siap'}")
                print(f"Pelatihan CPU: {'siap' if result['training_ready'] else 'belum siap'}")
                for device in result["audio"]["inputs"]:
                    print(f"  Mikrofon ID {device['id']}: {device['name']}")
                if result["audio"]["reason"]:
                    print(result["audio"]["reason"])
                if result["gpu"]["available"]:
                    print(f"GPU: siap ({result['gpu']['name']})")
                else:
                    print(
                        f"GPU: belum siap ({result['gpu']['name'] or result['gpu']['backend']}); "
                        f"{result['gpu']['reason']}"
                    )
            return 0 if result["recording_ready"] and result["training_ready"] else 1
        if args.command == "web-import":
            from .web_collection import import_session

            result = import_session(args.package, args.output)
            print(
                f"Sesi diimpor: {args.output}; status={result['status']}; "
                f"durasi={result['duration_seconds']:.1f} detik."
            )
            if result["synthetic"]:
                print("DATA SINTETIS: hanya untuk verifikasi perangkat lunak.")
        elif args.command == "web-align":
            from .web_alignment import align_package

            result = align_package(args.package, args.output, args.evaluation_role)
            print(
                f"Audit alignment v2: {args.output}; status={result['status']}; "
                f"v1={result['original_v1_status']}."
            )
            if result["synthetic"]:
                print("DATA SINTETIS: hanya untuk verifikasi perangkat lunak.")
        elif args.command == "record":
            from .recording import record

            result = record(
                args.data, args.session, args.per_key, args.device, args.seed, args.retry
            )
            print(
                f"Sesi {args.session}: {result['status']}; "
                f"{result['accepted_events']} ketukan diterima."
            )
            return 0 if result["status"] == "complete" else 130
        if args.command == "prepare":
            from .dataset import prepare

            destination = args.output or args.data / "processed"
            result = prepare(args.data, destination)
            print(f"Dataset disimpan: {destination}; {result['counts']}")
        elif args.command == "train":
            from .models import train_cnn, train_svm

            if args.kind == "svm":
                if args.device == "gpu":
                    raise ResearchError("SVM memakai CPU. Pilih train cnn untuk GPU.")
                train_svm(args.dataset, args.output)
            else:
                train_cnn(
                    args.dataset,
                    args.output,
                    args.device,
                    args.epochs,
                    args.patience,
                    args.batch_size,
                    tuple(args.seeds),
                )
            print(f"Model disimpan: {args.output}")
        elif args.command == "evaluate":
            from .evaluation import evaluate

            result = evaluate(args.model, args.data, args.output, args.device)
            print(
                f"Akurasi={result['classification']['accuracy']:.4f}; "
                f"macro-F1={result['classification']['macro_f1']:.4f}; "
                f"F1 deteksi={result['detection']['f1']:.4f}; "
                f"CER={result['end_to_end']['cer']:.4f}"
            )
            if result["synthetic"]:
                print("DATA SINTETIS: hasil ini hanya untuk verifikasi perangkat lunak.")
        elif args.command == "predict":
            from .evaluation import predict

            if args.output and args.output.exists():
                raise ResearchError(f"File sudah ada: {args.output}")
            result = predict(args.model, args.audio, args.device)
            if args.output:
                write_json(args.output, result)
                print(f"Prediksi disimpan: {args.output}")
            else:
                print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "report":
            from .reporting import report

            print(f"Laporan disimpan: {report(args.models, args.output, args.evaluations)}")
        return 0
    except (ResearchError, OSError, ValueError, RuntimeError) as exc:
        print(f"Gagal: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nDihentikan.", file=sys.stderr)
        return 130
