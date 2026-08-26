from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Inferência de segmentação HER2 em um patch")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint best.pt ou last.pt")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Patch PNG/JPEG/TIFF")
    source.add_argument("--slide", help="Lâmina SVS/NDPI/TIFF")
    parser.add_argument("--x", type=int, help="Coordenada X no nível 0 da lâmina")
    parser.add_argument("--y", type=int, help="Coordenada Y no nível 0 da lâmina")
    parser.add_argument("--xml", help="XML Aperio; escolhe uma região tumoral se x/y forem omitidos")
    parser.add_argument("--level", type=int, default=None, help="Nível OpenSlide; usa o nível do treino por padrão")
    parser.add_argument("--patch-size", type=int, default=None, help="Tamanho para leitura da lâmina")
    parser.add_argument("--output", default="inference", help="Diretório de saída")
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def load_rgb(args, train_args) -> tuple[np.ndarray, str]:
    if args.input:
        path = Path(args.input)
        return np.asarray(Image.open(path).convert("RGB")), path.stem
    level = args.level if args.level is not None else int(train_args.get("level", 0))
    size = args.patch_size or int(train_args.get("patch_size", 256))
    x, y = args.x, args.y
    if x is None or y is None:
        if not args.xml:
            raise SystemExit("Ao usar --slide, informe --x/--y ou forneça --xml")
        regions = [region for region in parse_aperio_xml(args.xml) if not region.negative]
        if not regions:
            raise SystemExit("O XML não contém regiões tumorais positivas")
        region = max(regions, key=lambda item: cv2.contourArea(item.points))
        center = region.points.mean(axis=0)
        if cv2.pointPolygonTest(region.points, tuple(center), False) < 0:
            center = region.points[0]
        x, y = max(0, int(center[0] - size / 2)), max(0, int(center[1] - size / 2))
    with openslide.OpenSlide(args.slide) as slide:
        if not 0 <= level < slide.level_count:
            raise SystemExit(f"Nível {level} inválido; a lâmina possui {slide.level_count} níveis")
        patch = slide.read_region((x, y), level, (size, size)).convert("RGB")
    return np.asarray(patch), f"{Path(args.slide).stem}_x{x}_y{y}_l{level}"


def prepare_tensor(rgb: np.ndarray, channels: int) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = rgb.shape[:2]
    image = rgb.astype(np.float32) / 255.0
    if channels == 5:
        image = np.concatenate((image, rgb_to_hd(rgb)), axis=2)
    # Encoders reduzem a resolução sucessivamente; padding evita erro em
    # patches cujas dimensões não são múltiplas de 32.
    padded_h, padded_w = math.ceil(height / 32) * 32, math.ceil(width / 32) * 32
    image = cv2.copyMakeBorder(image, 0, padded_h - height, 0, padded_w - width, cv2.BORDER_REFLECT_101)
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)
    return tensor, (height, width)


def main():
    args = arguments()
    if not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold deve estar entre 0 e 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA foi solicitada, mas não está disponível")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint.get("args", {})
    architecture = train_args.get("architecture", "unet")
    encoder = train_args.get("encoder", "resnet50")
    channels = int(train_args.get("channels", 3))
    model = create_model(architecture, encoder, channels, encoder_weights=None)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    invalid = [name for name, value in model.state_dict().items()
               if torch.is_floating_point(value) and not torch.isfinite(value).all()]
    if invalid:
        raise SystemExit(f"Checkpoint inválido: {len(invalid)} tensores contêm NaN/Inf")

    rgb, name = load_rgb(args, train_args)
    tensor, (height, width) = prepare_tensor(rgb, channels)
    with torch.inference_mode():
        probability = model(tensor.to(device)).sigmoid()[0, 0, :height, :width].float().cpu().numpy()
    if not np.isfinite(probability).all():
        raise SystemExit("A inferência produziu valores NaN/Inf")

    mask = probability >= args.threshold
    overlay = rgb.copy()
    red = np.zeros_like(rgb); red[..., 0] = 255
    overlay[mask] = np.rint(.55 * rgb[mask] + .45 * red[mask]).astype(np.uint8)

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output / f"{name}_patch.png")
    Image.fromarray(np.rint(probability * 255).astype(np.uint8)).save(output / f"{name}_probability.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(output / f"{name}_mask.png")
    Image.fromarray(overlay).save(output / f"{name}_overlay.png")
    np.save(output / f"{name}_probability.npy", probability)
    print(f"Checkpoint: época {checkpoint.get('epoch')} | val_dice={checkpoint.get('val_dice')}")
    print(f"Dispositivo: {device} | canais: {channels} | tamanho: {width}x{height}")
    print(f"Pixels positivos: {mask.mean() * 100:.2f}% | threshold: {args.threshold}")
    print(f"Resultados: {output.resolve()}")


if __name__ == "__main__":
    main()
