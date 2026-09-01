"""Run the trained model on raw text, producing output shaped like
kcms-backend's Verdict (contracts.py) minus `surfaced_reason`, which is set
by the backend's routing layer, not the model.

    python -m src.infer --checkpoint saved_models/v1 --text "..."

To wire this in as the real classifier (Document/ARCHITECTURE.md section 8,
"Swapping in the real model is one line"), kcms-backend's
`RealClassifier.classify` should load a checkpoint with `ModerationClassifier`
below and map its dict output onto `Verdict`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.config import Config
from src.labels import ID_SEVERITY, ID_TARGET
from src.model import ModerationModel


class ModerationClassifier:
    def __init__(self, checkpoint_dir: str) -> None:
        from transformers import AutoTokenizer  # local import keeps CLI startup fast

        checkpoint_path = Path(checkpoint_dir)
        config_path = checkpoint_path / "config.json"
        self.config = Config(**json.loads(config_path.read_text())) if config_path.exists() else Config()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = ModerationModel(self.config.model_name).to(self.device)
        self.model.load_state_dict(
            torch.load(checkpoint_path / "model.pt", map_location=self.device)
        )
        self.model.eval()

    @torch.no_grad()
    def classify(self, text: str) -> dict:
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        severity_logits, target_logits = self.model(
            encoding["input_ids"], encoding["attention_mask"]
        )
        severity_probs = torch.softmax(severity_logits, dim=-1).squeeze(0)
        target_probs = torch.softmax(target_logits, dim=-1).squeeze(0)

        severity_id = int(severity_probs.argmax())
        target_id = int(target_probs.argmax())
        severity_confidence = float(severity_probs[severity_id])
        target_confidence = float(target_probs[target_id])

        abstain = (
            severity_confidence < self.config.severity_threshold
            or target_confidence < self.config.target_threshold
        )

        return {
            "severity": ID_SEVERITY[severity_id].value,
            "severity_confidence": severity_confidence,
            "target": ID_TARGET[target_id].value,
            "target_confidence": target_confidence,
            "abstain": abstain,
            "rationale": None,
            "model_version": self.config.model_version,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory from train.py")
    parser.add_argument("--text", required=True, help="comment text to classify")
    args = parser.parse_args()

    classifier = ModerationClassifier(args.checkpoint)
    print(json.dumps(classifier.classify(args.text), indent=2, ensure_ascii=False))
