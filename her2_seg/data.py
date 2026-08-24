from __future__ import annotations

import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import openslide
import torch
from torch.utils.data import Dataset

SLIDE_EXTENSIONS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}


@dataclass(frozen=True)
class SlidePair:
    slide: Path
    xml: Path
    key: str


@dataclass(frozen=True)
class Region:
    points: np.ndarray
    negative: bool = False

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        x, y, w, h = cv2.boundingRect(self.points.astype(np.int32))
        return x, y, x + w, y + h


def discover_pairs(root: str | Path) -> tuple[list[SlidePair], list[Path]]:
    """Pareia recursivamente pelo nome do arquivo, mantendo a coorte no desempate."""
    root = Path(root)
    slides = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SLIDE_EXTENSIONS]
    by_stem: dict[str, list[Path]] = {}
    for slide in slides:
        by_stem.setdefault(slide.stem.casefold(), []).append(slide)

    pairs, missing = [], []
    for xml in sorted(root.rglob("*.xml")):
        candidates = by_stem.get(xml.stem.casefold(), [])
        if not candidates:
            missing.append(xml)
            continue
        # O ancestral comum mais profundo evita cruzar coortes em nomes repetidos.
        slide = max(candidates, key=lambda p: len(set(p.parents) & set(xml.parents)))
        pairs.append(SlidePair(slide, xml, xml.stem))
    return pairs, missing


def parse_aperio_xml(path: str | Path) -> list[Region]:
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        # Alguns XMLs da coorte Yale contêm uma segunda <Annotation> depois do
        # fechamento de <Annotations>. Ambas as partes são XML válido, mas o
        # documento possui mais de uma raiz. Uma raiz sintética preserva todas
        # as regiões sem modificar o conjunto de dados original.
        if "junk after document element" not in str(exc):
            raise
        text = path.read_text(encoding="utf-8-sig")
        # Nesses arquivos, a segunda Annotation ainda termina com
        # </Annotations>, embora não tenha a abertura correspondente.
        text = re.sub(r"<\?xml[^>]*\?>", "", text)
        text = re.sub(r"</?Annotations\b[^>]*>", "", text)
        root = ET.fromstring(f"<AperioDocument>{text}</AperioDocument>")
    regions: list[Region] = []
    for node in root.findall(".//Region"):
        points = [(float(v.attrib["X"]), float(v.attrib["Y"])) for v in node.findall(".//Vertex")]
        if len(points) >= 3:
            regions.append(Region(np.asarray(points, dtype=np.float32), node.attrib.get("NegativeROA", "0") == "1"))
    return regions


def rasterize_patch(regions: list[Region], x: int, y: int, size: int, downsample: float) -> np.ndarray:
    """Rasteriza apenas o patch pedido; coordenadas XML permanecem no nível 0."""
    mask = np.zeros((size, size), dtype=np.uint8)
    x1, y1 = x + size * downsample, y + size * downsample
    positives = [r for r in regions if not r.negative]
    negatives = [r for r in regions if r.negative]
    for region in positives + negatives:
        rx0, ry0, rx1, ry1 = region.bounds
        if rx1 < x or ry1 < y or rx0 >= x1 or ry0 >= y1:
            continue
        pts = np.rint((region.points - (x, y)) / downsample).astype(np.int32)
        cv2.fillPoly(mask, [pts], 0 if region.negative else 1)
    return mask


def rgb_to_hd(rgb: np.ndarray) -> np.ndarray:
    """Deconvolução H-DAB (Ruifrok) normalizada em [0, 1]."""
    # Vetores H e DAB usuais; o terceiro vetor completa a base de cor.
    matrix = np.asarray([[0.650, 0.704, 0.286], [0.268, 0.570, 0.776], [0.711, 0.423, 0.561]], np.float32)
    optical_density = -np.log((rgb.astype(np.float32) + 1.0) / 256.0)
    concentrations = optical_density.reshape(-1, 3) @ np.linalg.inv(matrix)
    hd = concentrations[:, :2].reshape(*rgb.shape[:2], 2)
    lo, hi = np.percentile(hd, 1, axis=(0, 1)), np.percentile(hd, 99, axis=(0, 1))
    return np.clip((hd - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


class WSIPatchDataset(Dataset):
    """Lê RGB e cria a máscara diretamente do XML, sem patches em disco."""

    def __init__(self, pairs: list[SlidePair], patch_size=512, level=0, samples_per_slide=32,
                 positive_fraction=0.7, channels=3, augment=False, seed=42):
        if channels not in (3, 5):
            raise ValueError("channels deve ser 3 (RGB) ou 5 (RGB+H+DAB)")
        self.pairs, self.patch_size, self.level = pairs, patch_size, level
        self.samples_per_slide, self.positive_fraction = samples_per_slide, positive_fraction
        self.channels, self.augment, self.seed = channels, augment, seed
        self.regions = [parse_aperio_xml(p.xml) for p in pairs]

    def __len__(self):
        return len(self.pairs) * self.samples_per_slide

    def _coordinates(self, pair_idx: int, sample_idx: int, dimensions, downsample):
        rng = random.Random(self.seed + pair_idx * 1_000_003 + sample_idx)
        width, height = dimensions
        span = self.patch_size * downsample
        positives = [r for r in self.regions[pair_idx] if not r.negative]
        if positives and rng.random() < self.positive_fraction:
            region = rng.choice(positives)
            x0, y0, x1, y1 = region.bounds
            cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
            x, y = int(cx - span / 2), int(cy - span / 2)
        else:
            x, y = rng.randint(0, max(0, int(width - span))), rng.randint(0, max(0, int(height - span)))
        return max(0, min(x, int(width - span))), max(0, min(y, int(height - span)))

    def __getitem__(self, index):
        pair_idx, sample_idx = divmod(index, self.samples_per_slide)
        pair = self.pairs[pair_idx]
        with openslide.OpenSlide(str(pair.slide)) as slide:
            downsample = float(slide.level_downsamples[self.level])
            x, y = self._coordinates(pair_idx, sample_idx, slide.dimensions, downsample)
            rgb = np.asarray(slide.read_region((x, y), self.level, (self.patch_size, self.patch_size)).convert("RGB"))
        mask = rasterize_patch(self.regions[pair_idx], x, y, self.patch_size, downsample)
        if self.augment:
            rng = random.Random(self.seed + index)
            if rng.random() < .5: rgb, mask = np.fliplr(rgb), np.fliplr(mask)
            if rng.random() < .5: rgb, mask = np.flipud(rgb), np.flipud(mask)
            k = rng.randrange(4)
            rgb, mask = np.rot90(rgb, k), np.rot90(mask, k)
        image = rgb.astype(np.float32) / 255.0
        if self.channels == 5:
            image = np.concatenate((image, rgb_to_hd(rgb)), axis=2)
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        mask = np.ascontiguousarray(mask[None].astype(np.float32))
        return torch.from_numpy(image), torch.from_numpy(mask)
