"""Evaluate a checkpoint on the eval split.

Reports macro-F1 and per-class recall per axis — never accuracy alone
(Document/PROPOSAL.md item 5: a model predicting SAFE for everything scores
~70% accuracy and is useless).

    python -m src.evaluate --data data/comments.csv --checkpoint saved_models/v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import Config
from src.dataset import ModerationDataset, load_split
from src.labels import ID_SEVERITY, ID_TARGET, Severity, Target
from src.model import ModerationModel


@torch.no_grad()
def collect_predictions(
    model: ModerationModel, loader: DataLoader, device: torch.device
) -> tuple[list[int], list[int], list[int], list[int], list[float], list[float]]:
    model.eval()
    severity_true, severity_pred = [], []
    target_true, target_pred = [], []
    severity_conf, target_conf = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        severity_logits, target_logits = model(input_ids, attention_mask)
        severity_probs = torch.softmax(severity_logits, dim=-1)
        target_probs = torch.softmax(target_logits, dim=-1)

        severity_true += batch["severity_id"].tolist()
        target_true += batch["target_id"].tolist()
        severity_pred += severity_probs.argmax(dim=-1).tolist()
        target_pred += target_probs.argmax(dim=-1).tolist()
        severity_conf += severity_probs.max(dim=-1).values.tolist()
        target_conf += target_probs.max(dim=-1).values.tolist()

    return severity_true, severity_pred, target_true, target_pred, severity_conf, target_conf


def calibrate_thresholds(
    confidences: list[float], correct: list[bool], target_abstain_rate: float = 0.1
) -> float:
    """Pick the lowest confidence threshold such that abstaining below it
    covers roughly `target_abstain_rate` of the eval set. A real calibration
    pass should instead pick the threshold that meets a target Missed Harm
    Rate / False Suppression Rate — this is a placeholder until there's real
    eval data to calibrate against.
    """
    if not confidences:
        return 0.6
    sorted_conf = sorted(confidences)
    cutoff_index = int(len(sorted_conf) * target_abstain_rate)
    return sorted_conf[min(cutoff_index, len(sorted_conf) - 1)]


def evaluate(config: Config, data_csv: str, checkpoint_dir: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint_dir)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    eval_df = load_split(data_csv, split="eval")
    if eval_df.empty:
        raise ValueError(f"No rows with split='eval' in {data_csv}")

    loader = DataLoader(
        ModerationDataset(eval_df, tokenizer, config.max_length), batch_size=config.batch_size
    )

    model = ModerationModel(config.model_name).to(device)
    model.load_state_dict(torch.load(checkpoint_path / "model.pt", map_location=device))

    sev_true, sev_pred, tgt_true, tgt_pred, sev_conf, tgt_conf = collect_predictions(
        model, loader, device
    )

    severity_report = classification_report(
        sev_true,
        sev_pred,
        labels=list(ID_SEVERITY.keys()),
        target_names=[ID_SEVERITY[i].value for i in ID_SEVERITY],
        output_dict=True,
        zero_division=0,
    )
    target_report = classification_report(
        tgt_true,
        tgt_pred,
        labels=list(ID_TARGET.keys()),
        target_names=[ID_TARGET[i].value for i in ID_TARGET],
        output_dict=True,
        zero_division=0,
    )

    severity_threshold = calibrate_thresholds(
        sev_conf, [p == t for p, t in zip(sev_pred, sev_true)]
    )
    target_threshold = calibrate_thresholds(
        tgt_conf, [p == t for p, t in zip(tgt_pred, tgt_true)]
    )

    return {
        "severity": {
            "macro_f1": severity_report["macro avg"]["f1-score"],
            "per_class_recall": {
                name: severity_report[name]["recall"] for name in (s.value for s in Severity)
            },
            "calibrated_threshold": severity_threshold,
        },
        "target": {
            "macro_f1": target_report["macro avg"]["f1-score"],
            "per_class_recall": {
                name: target_report[name]["recall"] for name in (t.value for t in Target)
            },
            "calibrated_threshold": target_threshold,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to labelled CSV")
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory from train.py")
    args = parser.parse_args()

    cfg_path = Path(args.checkpoint) / "config.json"
    cfg = Config(**json.loads(cfg_path.read_text())) if cfg_path.exists() else Config()

    results = evaluate(cfg, args.data, args.checkpoint)
    print(json.dumps(results, indent=2))
