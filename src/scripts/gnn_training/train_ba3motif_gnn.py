"""Train the BA3Motif target GNN and export its validation-selected weights."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader

from src.data.ba3motif_dataset import CLASS_NAMES, DATA_SHA256, load_ba3motif
from src.models.ba3motif_gnn import BA3MotifLEConv


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    confusion = torch.zeros((3, 3), dtype=torch.long)
    for batch in loader:
        batch = batch.to(device)
        labels = batch.y.view(-1)
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        pairs = 3 * labels + logits.argmax(dim=1)
        confusion += torch.bincount(pairs.cpu(), minlength=9).reshape(3, 3)

    count = int(confusion.sum())
    if not math.isfinite(loss_sum):
        raise RuntimeError("Non-finite validation loss")
    return {
        "loss": loss_sum / count,
        "accuracy": int(confusion.diag().sum()) / count,
        "class_accuracy": (confusion.diag() / confusion.sum(dim=1)).tolist(),
        "confusion_matrix": confusion.tolist(),
        "count": count,
    }


def main():
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=repo / "src/data/ba3motif")
    parser.add_argument("--output-path", type=Path, default=repo / "src/params/gnns/ba3motif/ba3motif_leconv.pt")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-units", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.num_units, args.hidden_dim) < 1:
        parser.error("epochs, batch size, layer count and hidden dimension must be positive")
    if args.output_path.exists() or args.output_path.is_symlink():
        raise FileExistsError(f"Checkpoint already exists: {args.output_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable for requested device {device}")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    config = {
        "arguments": vars(args),
        "command": shlex.join([sys.executable, "-m", "src.scripts.gnn_training.train_ba3motif_gnn", *sys.argv[1:]]),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "numpy": np.__version__,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "data_sha256": DATA_SHA256,
        "split_sizes": {"training": 700, "evaluation": 150},
        "source_sha256": {
            path: sha256((repo / path).read_bytes()).hexdigest()
            for path in (
                "src/data/ba3motif_dataset.py", "src/models/ba3motif_gnn.py",
                "src/scripts/gnn_training/train_ba3motif_gnn.py",
            )
        },
    }
    (args.run_dir / "run.json").write_text(json.dumps(config, indent=2, default=str) + "\n")

    data = load_ba3motif(args.data_path, splits=("training", "evaluation"))
    train_loader = DataLoader(data["training"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(data["evaluation"], batch_size=args.batch_size)
    model_kwargs = dict(in_channels=10, num_unit=args.num_units,
                        hidden_dim=args.hidden_dim, dropout=args.dropout)
    model = BA3MotifLEConv(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_loss = float("inf")
    run_checkpoint = args.run_dir / "best.pt"

    with (args.run_dir / "training.jsonl").open("x") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(batch.x, batch.edge_index, batch.batch), batch.y.view(-1))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
                loss.backward()
                optimizer.step()

            validation = evaluate(model, val_loader, device)
            record = {"epoch": epoch, "validation": validation}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(f"Epoch {epoch:03d}  val loss {validation['loss']:.4f}  "
                  f"val accuracy {validation['accuracy']:.4f}", flush=True)
            if validation["loss"] < best_loss:
                best_loss = validation["loss"]
                checkpoint = {
                    "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "model_kwargs": model_kwargs,
                    "class_names": CLASS_NAMES,
                    "data_sha256": DATA_SHA256,
                    "seed": args.seed,
                    **record,
                }
                temporary_checkpoint = args.run_dir / "best.tmp"
                torch.save(checkpoint, temporary_checkpoint)
                temporary_checkpoint.replace(run_checkpoint)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output_path.parent) as temporary_dir:
        staged_checkpoint = Path(temporary_dir) / "checkpoint.pt"
        shutil.copyfile(run_checkpoint, staged_checkpoint)
        os.link(staged_checkpoint, args.output_path)
    print(f"Saved validation-selected checkpoint to {args.output_path}")


if __name__ == "__main__":
    main()
