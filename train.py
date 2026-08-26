from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from her2_seg.data import WSIPatchDataset, discover_pairs
from her2_seg.model import DiceFocalLoss, create_model, dice_score

DEFAULT_ROOT = "/mnt/HD_JULIO/Laminas_ROI/HER2-ROI"


def arguments():
    p = argparse.ArgumentParser(description="Treino end-to-end de segmentação tumoral em WSI/XML")
    p.add_argument("--data-root", default=DEFAULT_ROOT)
    p.add_argument("--output", default="runs/her2_unet")
    p.add_argument("--architecture", choices=["unet", "fpn", "deeplabv3plus"], default="unet")
    p.add_argument("--encoder", default="resnet50")
    p.add_argument("--channels", type=int, choices=[3, 5], default=3)
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--samples-per-slide", type=int, default=32)
    p.add_argument("--positive-fraction", type=float, default=.7)
    p.add_argument("--val-fraction", type=float, default=.2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--no-amp", action="store_true", help="Desativa mixed precision (mais estável, usa mais VRAM)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--freeze-bn", action="store_true", help="Congela estatísticas BatchNorm; recomendado com batch 1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inspect", action="store_true", help="Somente mostra pareamento e divisão")
    return p.parse_args()


def main():
    args = arguments()
    pairs, missing = discover_pairs(args.data_root)
    if not pairs:
        raise SystemExit("Nenhum par lâmina/XML encontrado")
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_val = max(1, round(len(pairs) * args.val_fraction))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"Pares: {len(pairs)} | treino: {len(train_pairs)} | validação: {len(val_pairs)} | XML sem lâmina: {len(missing)}")
    if args.inspect:
        for pair in pairs[:10]: print(f"{pair.key}: {pair.slide}")
        return

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))
    split = {
        "seed": args.seed,
        "train": [{"key": p.key, "slide": str(p.slide), "xml": str(p.xml)} for p in train_pairs],
        "validation": [{"key": p.key, "slide": str(p.slide), "xml": str(p.xml)} for p in val_pairs],
    }
    (output / "split.json").write_text(json.dumps(split, indent=2, ensure_ascii=False))
    common = dict(patch_size=args.patch_size, level=args.level, samples_per_slide=args.samples_per_slide,
                  positive_fraction=args.positive_fraction, channels=args.channels, seed=args.seed)
    train_ds = WSIPatchDataset(train_pairs, augment=True, **common)
    val_ds = WSIPatchDataset(val_pairs, augment=False, **common)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_dl = DataLoader(val_ds, args.batch_size, num_workers=args.workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.architecture, args.encoder, args.channels).to(device)
    loss_fn, optimizer = DiceFocalLoss(), torch.optim.AdamW(model.parameters(), lr=args.lr)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = GradScaler("cuda", enabled=amp_enabled)
    metrics_path = output / "metrics.csv"
    with metrics_path.open("w", newline="") as file:
        csv.writer(file).writerow(["epoch", "train_loss", "val_dice"])
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_bn or args.batch_size == 1:
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        running = 0.0
        for image, mask in tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}"):
            image, mask = image.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device.type, enabled=amp_enabled):
                logits = model(image)
                loss = loss_fn(logits, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Loss não finita na época {epoch}. O treino foi interrompido antes de salvar checkpoint. "
                    "Tente --no-amp, patch menor e --freeze-bn."
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer); scaler.update()
            running += loss.item()
        model.eval(); scores = []
        with torch.inference_mode():
            for image, mask in val_dl:
                image, mask = image.to(device, non_blocking=True), mask.to(device, non_blocking=True)
                with autocast(device.type, enabled=amp_enabled):
                    logits = model(image)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(f"Predição não finita na validação da época {epoch}")
                scores.append(dice_score(logits.float(), mask))
        score = sum(scores) / len(scores)
        train_loss = running / len(train_dl)
        print(f"epoch={epoch} loss={train_loss:.5f} val_dice={score:.5f}")
        with metrics_path.open("a", newline="") as file:
            csv.writer(file).writerow([epoch, train_loss, score])
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "train_loss": train_loss, "val_dice": score, "args": vars(args)}
        torch.save(state, output / "last.pt")
        if score > best:
            best = score; torch.save(state, output / "best.pt")


if __name__ == "__main__":
    main()
