from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from her2_seg.data import SlidePair, WSIPatchDataset
from her2_seg.model import create_model


def arguments():
    parser = argparse.ArgumentParser(description="Avalia e visualiza um split de lâminas HER2")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--output", default="evaluation/test")
    parser.add_argument("--patches-per-slide", type=int, default=4)
    parser.add_argument("--max-patches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def overlay(rgb: np.ndarray, truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """XML em azul, modelo em vermelho e interseção em magenta."""
    result = rgb.astype(np.float32).copy()
    blue = np.zeros_like(result); blue[..., 2] = 255
    red = np.zeros_like(result); red[..., 0] = 255
    result[truth] = .5 * result[truth] + .5 * blue[truth]
    result[prediction] = .5 * result[prediction] + .5 * red[prediction]
    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    args = arguments()
    if not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold deve estar entre 0 e 1")
    split_data = json.loads(Path(args.split_json).read_text())
    entries = split_data.get(args.split, [])
    if not entries:
        available = ", ".join(key for key in ("validation", "test") if split_data.get(key))
        raise SystemExit(f"Split '{args.split}' vazio ou ausente. Disponíveis: {available or 'nenhum'}")
    pairs = [SlidePair(Path(item["slide"]), Path(item["xml"]), item["key"]) for item in entries]

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("args", {})
    channels = int(cfg.get("channels", 3))
    model = create_model(cfg.get("architecture", "unet"), cfg.get("encoder", "resnet50"),
                         channels, encoder_weights=None)
    model.load_state_dict(checkpoint["model"])
    invalid = [name for name, value in model.state_dict().items()
               if torch.is_floating_point(value) and not torch.isfinite(value).all()]
    if invalid:
        raise SystemExit(f"Checkpoint inválido: {len(invalid)} tensores contêm NaN/Inf")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA solicitada, mas indisponível")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    model.to(device).eval()

    dataset = WSIPatchDataset(
        pairs, patch_size=int(cfg.get("patch_size", 256)), level=int(cfg.get("level", 0)),
        samples_per_slide=args.patches_per_slide, positive_fraction=float(cfg.get("positive_fraction", .7)),
        channels=channels, augment=False, seed=int(cfg.get("seed", 42)) + 10_000,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    rows, processed = [], 0
    with torch.inference_mode():
        for images, truths in loader:
            probabilities = model(images.to(device, non_blocking=True)).sigmoid().float().cpu()
            for image, truth, probability in zip(images, truths, probabilities):
                if processed >= args.max_patches:
                    break
                pair_idx, sample_idx = divmod(processed, args.patches_per_slide)
                key = pairs[pair_idx].key
                truth_np = truth[0].numpy() >= .5
                pred_np = probability[0].numpy() >= args.threshold
                intersection = np.logical_and(truth_np, pred_np).sum()
                union = np.logical_or(truth_np, pred_np).sum()
                dice = (2 * intersection + 1) / (truth_np.sum() + pred_np.sum() + 1)
                iou = (intersection + 1) / (union + 1)
                rgb = np.rint(image[:3].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                name = f"{processed:04d}_{key}_p{sample_idx}"
                Image.fromarray(overlay(rgb, truth_np, pred_np)).save(output / f"{name}_overlay.png")
                Image.fromarray((truth_np * 255).astype(np.uint8)).save(output / f"{name}_xml.png")
                Image.fromarray((pred_np * 255).astype(np.uint8)).save(output / f"{name}_model.png")
                rows.append([processed, key, sample_idx, float(dice), float(iou),
                             float(truth_np.mean()), float(pred_np.mean())])
                processed += 1
            if processed >= args.max_patches:
                break
    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["patch", "slide", "sample", "dice", "iou", "xml_fraction", "model_fraction"])
        writer.writerows(rows)
    summary = {
        "checkpoint_epoch": checkpoint.get("epoch"), "checkpoint_val_dice": checkpoint.get("val_dice"),
        "split": args.split, "slides": len(pairs), "patches": len(rows), "threshold": args.threshold,
        "mean_patch_dice": float(np.mean([row[3] for row in rows])),
        "mean_patch_iou": float(np.mean([row[4] for row in rows])),
        "legend": {"blue": "XML (referência)", "red": "modelo", "magenta": "sobreposição"},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Resultados: {output.resolve()}")


if __name__ == "__main__":
    main()
