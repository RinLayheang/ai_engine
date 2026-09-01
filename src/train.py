"""Fine-tune ModerationModel on a labelled CSV (Document/LABEL-SCHEME (2).md schema).

    python -m src.train --data data/comments.csv --out saved_models/v1

The dataset itself is never committed to git (see ARCHITECTURE.md section 7);
point --data at a local CSV that lives outside this repo's tracked files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.config import Config
from src.dataset import ModerationDataset, load_split
from src.model import ModerationModel


def train(config: Config, data_csv: str, out_dir: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_df = load_split(data_csv, split="train")
    if train_df.empty:
        raise ValueError(f"No rows with split='train' in {data_csv}")

    train_loader = DataLoader(
        ModerationDataset(train_df, tokenizer, config.max_length),
        batch_size=config.batch_size,
        shuffle=True,
    )

    model = ModerationModel(config.model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(config.epochs):
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}/{config.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            severity_id = batch["severity_id"].to(device)
            target_id = batch["target_id"].to(device)

            severity_logits, target_logits = model(input_ids, attention_mask)
            loss = loss_fn(severity_logits, severity_id) + loss_fn(target_logits, target_id)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"epoch {epoch + 1}: avg loss {total_loss / len(train_loader):.4f}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path / "model.pt")
    tokenizer.save_pretrained(out_path)
    (out_path / "config.json").write_text(json.dumps(config.__dict__, indent=2))
    print(f"saved checkpoint to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to labelled CSV")
    parser.add_argument("--out", required=True, help="checkpoint output directory")
    parser.add_argument("--model-name", default=Config.model_name)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.learning_rate)
    args = parser.parse_args()

    cfg = Config(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    train(cfg, args.data, args.out)
