from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser(description="Gera panoramas para todas as WSIs de um split")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--output-root", default="evaluation/validation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--thumbnail-size", type=int, default=2400)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--margin", type=int, default=1024)
    parser.add_argument("--max-tiles", type=int, default=20000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-slides", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def main():
    args = arguments()
    data = json.loads(Path(args.split_json).read_text())
    entries = data.get(args.split, [])
    if not entries:
        raise SystemExit(f"Split '{args.split}' ausente ou vazio")
    stop = len(entries) if args.max_slides is None else min(len(entries), args.start_index + args.max_slides)
    selected = list(enumerate(entries))[args.start_index:stop]
    output_root = Path(args.output_root); output_root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("evaluate_wsi.py")
    status_path = output_root / "status.csv"
    statuses = []

    for position, entry in selected:
        case_dir = output_root / f"{position:03d}_{safe_name(entry['key'])}"
        overview = case_dir / "overview.png"
        if overview.exists() and not args.overwrite:
            print(f"[{position + 1}/{len(entries)}] {entry['key']}: já existe, ignorando")
            statuses.append([position, entry["key"], "skipped", ""])
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(script), "--checkpoint", args.checkpoint,
            "--slide", entry["slide"], "--xml", entry["xml"], "--output", str(overview),
            "--batch-size", str(args.batch_size), "--thumbnail-size", str(args.thumbnail_size),
            "--threshold", str(args.threshold), "--margin", str(args.margin),
            "--max-tiles", str(args.max_tiles),
        ]
        if args.stride is not None:
            command.extend(["--stride", str(args.stride)])
        print(f"[{position + 1}/{len(entries)}] {entry['key']}")
        result = subprocess.run(command, text=True)
        status = "ok" if result.returncode == 0 else "error"
        statuses.append([position, entry["key"], status, result.returncode])
        with status_path.open("w", newline="") as file:
            writer = csv.writer(file); writer.writerow(["index", "slide", "status", "return_code"])
            writer.writerows(statuses)
    successes = sum(row[2] in ("ok", "skipped") for row in statuses)
    failures = sum(row[2] == "error" for row in statuses)
    print(f"Concluído: {successes} disponíveis | {failures} erros | saída: {output_root.resolve()}")


if __name__ == "__main__":
    main()
