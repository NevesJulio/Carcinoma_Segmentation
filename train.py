from __future__ import annotations

import argparse
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
    common = dict(patch_size=args.patch_size, level=args.level, samples_per_slide=args.samples_per_slide,
                  positive_fraction=args.positive_fraction, channels=args.channels, seed=args.seed)
    train_ds = WSIPatchDataset(train_pairs, augment=True, **common)
    val_ds = WSIPatchDataset(val_pairs, augment=False, **common)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_dl = DataLoader(val_ds, args.batch_size, num_workers=args.workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.architecture, args.encoder, args.channels).to(device)
    loss_fn, optimizer = DiceFocalLoss(), torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for image, mask in tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}"):
            image, mask = image.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(image), mask)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running += loss.item()
        model.eval(); scores = []
        for image, mask in val_dl:
            image, mask = image.to(device), mask.to(device)
            scores.append(dice_score(model(image), mask))
        score = sum(scores) / len(scores)
        print(f"epoch={epoch} loss={running/len(train_dl):.5f} val_dice={score:.5f}")
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "val_dice": score, "args": vars(args)}
        torch.save(state, output / "last.pt")
        if score > best:
            best = score; torch.save(state, output / "best.pt")


if __name__ == "__main__":
    main()

