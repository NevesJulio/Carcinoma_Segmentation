from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import openslide
import torch
from PIL import Image

from her2_seg.data import parse_aperio_xml, rgb_to_hd
from her2_seg.model import create_model


def arguments():
    parser = argparse.ArgumentParser(description="Compara XML e modelo em uma imagem panorâmica da WSI")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--slide", help="Lâmina; alternativa a --split-json")
    parser.add_argument("--xml", help="XML correspondente; usado com --slide")
    parser.add_argument("--split-json", help="Seleciona automaticamente uma lâmina do split")
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--index", type=int, default=0, help="Índice da lâmina dentro do split")
    parser.add_argument("--output", required=True, help="PNG panorâmico de saída")
    parser.add_argument("--thumbnail-size", type=int, default=2400)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--stride", type=int, default=None, help="Stride em pixels no nível do modelo; padrão=patch")
    parser.add_argument("--margin", type=int, default=1024, help="Margem nível 0 em torno das anotações")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tiles", type=int, default=20000, help="Proteção contra inferência acidental enorme")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def tensor_from_rgb(rgb: np.ndarray, channels: int) -> torch.Tensor:
    image = rgb.astype(np.float32) / 255.0
    if channels == 5:
        image = np.concatenate((image, rgb_to_hd(rgb)), axis=2)
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def rasterize_overview(regions, width, height, scale_x, scale_y):
    mask = np.zeros((height, width), np.uint8)
    for negative in (False, True):
        for region in (item for item in regions if item.negative == negative):
            points = region.points.copy()
            points[:, 0] *= scale_x; points[:, 1] *= scale_y
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 0 if negative else 1)
    return mask.astype(bool)


def main():
    args = arguments()
    if args.split_json:
        split_data = json.loads(Path(args.split_json).read_text())
        entries = split_data.get(args.split, [])
        if not entries:
            raise SystemExit(f"Split '{args.split}' ausente ou vazio em {args.split_json}")
        if not 0 <= args.index < len(entries):
            raise SystemExit(f"--index deve estar entre 0 e {len(entries) - 1}")
        args.slide, args.xml = entries[args.index]["slide"], entries[args.index]["xml"]
        print(f"Caso selecionado [{args.index}/{len(entries) - 1}]: {entries[args.index]['key']}")
    elif not args.slide or not args.xml:
        raise SystemExit("Informe --slide e --xml, ou use --split-json")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("args", {})
    channels = int(cfg.get("channels", 3))
    model = create_model(cfg.get("architecture", "unet"), cfg.get("encoder", "resnet50"),
                         channels, encoder_weights=None)
    model.load_state_dict(checkpoint["model"])
    invalid = [name for name, value in model.state_dict().items()
               if torch.is_floating_point(value) and not torch.isfinite(value).all()]
    if invalid:
        raise SystemExit(f"Checkpoint inválido: {len(invalid)} tensores com NaN/Inf")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA solicitada, mas indisponível")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    model.to(device).eval()

    regions = parse_aperio_xml(args.xml)
    positives = [region for region in regions if not region.negative]
    if not positives:
        raise SystemExit("O XML não possui regiões positivas")
    patch_size = int(cfg.get("patch_size", 256))
    level = int(cfg.get("level", 0))
    stride = args.stride or patch_size

    with openslide.OpenSlide(args.slide) as slide:
        slide_w, slide_h = slide.dimensions
        overview = np.asarray(slide.get_thumbnail((args.thumbnail_size, args.thumbnail_size)).convert("RGB"))
        out_h, out_w = overview.shape[:2]
        scale_x, scale_y = out_w / slide_w, out_h / slide_h
        truth = rasterize_overview(regions, out_w, out_h, scale_x, scale_y)
        probability = np.zeros((out_h, out_w), np.float32)
        coverage = np.zeros((out_h, out_w), bool)

        downsample = float(slide.level_downsamples[level])
        span, step = patch_size * downsample, stride * downsample
        x0 = max(0, min(region.bounds[0] for region in positives) - args.margin)
        y0 = max(0, min(region.bounds[1] for region in positives) - args.margin)
        x1 = min(slide_w, max(region.bounds[2] for region in positives) + args.margin)
        y1 = min(slide_h, max(region.bounds[3] for region in positives) + args.margin)
        xs = list(range(int(x0), max(int(x0) + 1, int(x1 - span) + 1), max(1, int(step))))
        ys = list(range(int(y0), max(int(y0) + 1, int(y1 - span) + 1), max(1, int(step))))
        if not xs or xs[-1] < x1 - span: xs.append(max(0, int(x1 - span)))
        if not ys or ys[-1] < y1 - span: ys.append(max(0, int(y1 - span)))
        coordinates = [(x, y) for y in ys for x in xs]
        if len(coordinates) > args.max_tiles:
            raise SystemExit(
                f"A região exige {len(coordinates)} tiles (limite {args.max_tiles}). "
                "Aumente --stride ou --max-tiles conscientemente."
            )
        print(f"WSI: {slide_w}x{slide_h} | panorama: {out_w}x{out_h} | tiles: {len(coordinates)}")
        with torch.inference_mode():
            for start in range(0, len(coordinates), args.batch_size):
                batch_coordinates = coordinates[start:start + args.batch_size]
                rgbs = [np.asarray(slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB"))
                        for x, y in batch_coordinates]
                tensors = torch.stack([tensor_from_rgb(rgb, channels) for rgb in rgbs]).to(device)
                predictions = model(tensors).sigmoid()[:, 0].float().cpu().numpy()
                for (x, y), pred in zip(batch_coordinates, predictions):
                    ox0, oy0 = int(round(x * scale_x)), int(round(y * scale_y))
                    ox1 = min(out_w, int(round((x + span) * scale_x)))
                    oy1 = min(out_h, int(round((y + span) * scale_y)))
                    if ox1 <= ox0 or oy1 <= oy0: continue
                    resized = cv2.resize(pred, (ox1 - ox0, oy1 - oy0), interpolation=cv2.INTER_LINEAR)
                    probability[oy0:oy1, ox0:ox1] = np.maximum(probability[oy0:oy1, ox0:ox1], resized)
                    coverage[oy0:oy1, ox0:ox1] = True
                print(f"\rTiles processados: {min(start + args.batch_size, len(coordinates))}/{len(coordinates)}", end="", flush=True)
        print()

    prediction = probability >= args.threshold
    canvas = overview.astype(np.float32)
    blue = np.zeros_like(canvas); blue[..., 2] = 255
    red = np.zeros_like(canvas); red[..., 0] = 255
    canvas[truth] = .5 * canvas[truth] + .5 * blue[truth]
    canvas[prediction] = .5 * canvas[prediction] + .5 * red[prediction]
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    evaluated = coverage
    intersection = np.logical_and(truth, prediction) & evaluated
    truth_eval, pred_eval = truth & evaluated, prediction & evaluated
    dice = (2 * intersection.sum() + 1) / (truth_eval.sum() + pred_eval.sum() + 1)
    union = np.logical_or(truth_eval, pred_eval)
    iou = (intersection.sum() + 1) / (union.sum() + 1)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(output)
    Image.fromarray((truth * 255).astype(np.uint8)).save(output.with_name(output.stem + "_xml.png"))
    Image.fromarray((prediction * 255).astype(np.uint8)).save(output.with_name(output.stem + "_model.png"))
    Image.fromarray(np.rint(probability * 255).astype(np.uint8)).save(output.with_name(output.stem + "_probability.png"))
    summary = {"slide": str(args.slide), "xml": str(args.xml), "checkpoint_epoch": checkpoint.get("epoch"),
               "tiles": len(coordinates), "dice_overview": float(dice), "iou_overview": float(iou),
               "threshold": args.threshold, "blue": "XML", "red": "modelo", "magenta": "concordância"}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Imagem panorâmica: {output.resolve()}")


if __name__ == "__main__":
    main()
