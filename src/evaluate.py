"""Evaluate a checkpoint on the eval split.

Reports macro-F1 and per-class recall per axis — never accuracy alone
(Document/PROPOSAL.md item 5: a model predicting SAFE for everything scores
~70% accuracy and is useless) — and alongside them the two rates the product
is actually judged on, False Suppression Rate and Missed Harm Rate, defined
in `src/metrics.py`.

Macro-F1 and FSR answer different questions. A model can post a respectable
macro-F1 while failing the one cell this project exists to protect, because
that cell is small enough to disappear into an average. FSR does not average.

    python -m src.evaluate --data data/comments.csv --checkpoint saved_models/v1
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import Config
from src.dataset import ModerationDataset, load_split
from src.labels import ID_SEVERITY, ID_TARGET, Severity, Target
from src.metrics import Prediction, false_suppression_rate, missed_harm_rate
from src.model import ModerationModel

# Thresholds the calibration sweep considers. Coarse on purpose: a threshold
# read to two decimals off a few hundred eval rows is precision the data does
# not support.
DEFAULT_THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.30 + 0.05 * i, 2) for i in range(14))


@torch.no_grad()
def collect_predictions(
    model: ModerationModel, loader: DataLoader, device: torch.device
) -> tuple[list[int], list[int], list[Prediction]]:
    """Run the model once and keep raw argmax + confidence per row, so the
    threshold sweep can re-decide abstention without another forward pass."""
    model.eval()
    gold_severity: list[int] = []
    gold_target: list[int] = []
    predictions: list[Prediction] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        severity_logits, target_logits = model(input_ids, attention_mask)
        severity_probs = torch.softmax(severity_logits, dim=-1)
        target_probs = torch.softmax(target_logits, dim=-1)

        severity_best = severity_probs.max(dim=-1)
        target_best = target_probs.max(dim=-1)

        gold_severity += batch["severity_id"].tolist()
        gold_target += batch["target_id"].tolist()
        predictions += [
            Prediction(
                severity_id=int(severity_id),
                target_id=int(target_id),
                severity_confidence=float(severity_confidence),
                target_confidence=float(target_confidence),
            )
            for severity_id, target_id, severity_confidence, target_confidence in zip(
                severity_best.indices.tolist(),
                target_best.indices.tolist(),
                severity_best.values.tolist(),
                target_best.values.tolist(),
                strict=True,
            )
        ]

    return gold_severity, gold_target, predictions


def calibrate_thresholds(
    gold_severity: list[int],
    gold_target: list[int],
    predictions: list[Prediction],
    max_false_suppression_rate: float = 0.05,
    grid: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    auto_hide_offensive: bool = False,
) -> dict:
    """Pick the threshold pair with the lowest Missed Harm Rate among those
    holding False Suppression Rate at or under `max_false_suppression_rate`.

    This is the calibration ARCHITECTURE.md section 11 asks for, and it
    replaces an earlier placeholder that picked a confidence quantile so a
    fixed share of the eval set would abstain. That older rule optimised
    nothing: it hit an abstention budget while staying blind to which
    comments were being abstained on, which is the only part that matters.

    Raising a threshold trades the two rates against each other rather than
    improving both — more abstention means fewer hide-marked verdicts, which
    lowers FSR and raises MHR by exactly the rows that moved. Reporting how
    many pairs were eligible alongside the choice keeps that trade visible
    instead of letting a single selected pair imply there was no cost.

    A threshold is never chosen off an empty population, since a rate with no
    denominator cannot constrain anything (ARCHITECTURE.md section 11).
    """
    probe = false_suppression_rate(
        gold_severity, gold_target, predictions, 0.0, 0.0, auto_hide_offensive
    )
    if probe.denominator == 0:
        return {
            "calibrated": False,
            "reason": (
                f"cannot calibrate against False Suppression Rate: {probe.unavailable_reason}"
            ),
            "severity_threshold": None,
            "target_threshold": None,
        }

    sweep: list[dict] = []
    for severity_threshold in grid:
        for target_threshold in grid:
            fsr = false_suppression_rate(
                gold_severity,
                gold_target,
                predictions,
                severity_threshold,
                target_threshold,
                auto_hide_offensive,
            )
            mhr = missed_harm_rate(
                gold_severity,
                gold_target,
                predictions,
                severity_threshold,
                target_threshold,
                auto_hide_offensive,
            )
            sweep.append(
                {
                    "severity_threshold": severity_threshold,
                    "target_threshold": target_threshold,
                    "false_suppression_rate": fsr.value,
                    "missed_harm_rate": mhr.value,
                }
            )

    eligible = [
        point for point in sweep if point["false_suppression_rate"] <= max_false_suppression_rate
    ]
    if not eligible:
        best_possible = min(point["false_suppression_rate"] for point in sweep)
        return {
            "calibrated": False,
            "reason": (
                f"no threshold pair holds False Suppression Rate at or under "
                f"{max_false_suppression_rate}; the best any pair achieves is {best_possible}"
            ),
            "severity_threshold": None,
            "target_threshold": None,
        }

    # Lowest MHR wins. Ties break toward lower thresholds, because every
    # abstention costs a moderator's attention and two pairs that miss the
    # same harm are not equally cheap to run.
    selected = min(
        eligible,
        key=lambda point: (
            point["missed_harm_rate"] if point["missed_harm_rate"] is not None else 0.0,
            point["severity_threshold"],
            point["target_threshold"],
        ),
    )
    return {
        "calibrated": True,
        "reason": (
            f"lowest Missed Harm Rate among pairs holding False Suppression Rate "
            f"at or under {max_false_suppression_rate}"
        ),
        "severity_threshold": selected["severity_threshold"],
        "target_threshold": selected["target_threshold"],
        "false_suppression_rate": selected["false_suppression_rate"],
        "missed_harm_rate": selected["missed_harm_rate"],
        "pairs_considered": len(sweep),
        "pairs_eligible": len(eligible),
    }


def evaluate(
    config: Config,
    data_csv: str,
    checkpoint_dir: str,
    max_false_suppression_rate: float = 0.05,
) -> dict:
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

    gold_severity, gold_target, predictions = collect_predictions(model, loader, device)
    severity_pred = [prediction.severity_id for prediction in predictions]
    target_pred = [prediction.target_id for prediction in predictions]

    severity_report = classification_report(
        gold_severity,
        severity_pred,
        labels=list(ID_SEVERITY.keys()),
        target_names=[ID_SEVERITY[i].value for i in ID_SEVERITY],
        output_dict=True,
        zero_division=0,
    )
    target_report = classification_report(
        gold_target,
        target_pred,
        labels=list(ID_TARGET.keys()),
        target_names=[ID_TARGET[i].value for i in ID_TARGET],
        output_dict=True,
        zero_division=0,
    )

    calibration = calibrate_thresholds(
        gold_severity,
        gold_target,
        predictions,
        max_false_suppression_rate=max_false_suppression_rate,
    )

    # The headline rates are reported at the thresholds this checkpoint would
    # actually ship with — the calibrated pair when calibration succeeded, the
    # configured pair otherwise — so the published numbers describe the
    # deployed behaviour rather than a threshold nobody will run.
    severity_threshold = (
        calibration["severity_threshold"]
        if calibration["calibrated"]
        else config.severity_threshold
    )
    target_threshold = (
        calibration["target_threshold"] if calibration["calibrated"] else config.target_threshold
    )

    fsr = false_suppression_rate(
        gold_severity, gold_target, predictions, severity_threshold, target_threshold
    )
    mhr = missed_harm_rate(
        gold_severity, gold_target, predictions, severity_threshold, target_threshold
    )

    return {
        "eval_rows": len(predictions),
        "thresholds": {
            "severity": severity_threshold,
            "target": target_threshold,
            "source": "calibrated" if calibration["calibrated"] else "config default",
        },
        "false_suppression_rate": fsr.as_dict(),
        "missed_harm_rate": mhr.as_dict(),
        "calibration": calibration,
        "severity": {
            "macro_f1": severity_report["macro avg"]["f1-score"],
            "per_class_recall": {
                name: severity_report[name]["recall"] for name in (s.value for s in Severity)
            },
        },
        "target": {
            "macro_f1": target_report["macro avg"]["f1-score"],
            "per_class_recall": {
                name: target_report[name]["recall"] for name in (t.value for t in Target)
            },
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to labelled CSV")
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory from train.py")
    parser.add_argument(
        "--max-fsr",
        type=float,
        default=0.05,
        help="False Suppression Rate ceiling the calibrated thresholds must hold",
    )
    args = parser.parse_args()

    cfg_path = Path(args.checkpoint) / "config.json"
    cfg = Config(**json.loads(cfg_path.read_text())) if cfg_path.exists() else Config()

    results = evaluate(cfg, args.data, args.checkpoint, max_false_suppression_rate=args.max_fsr)
    print(json.dumps(results, indent=2))
