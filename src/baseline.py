"""TF-IDF character n-gram baseline, per Document/PROPOSAL.md item 5:
"Train xlm-roberta-base against a TF-IDF character n-gram baseline."

Character n-grams (not word n-grams) because Khmer has no spaces between
words — a word-level tokenizer would produce near-meaningless features.

    python -m src.baseline --data data/comments.csv
"""

from __future__ import annotations

import argparse
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from src.dataset import load_split
from src.labels import ID_SEVERITY, ID_TARGET, Severity, Target


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def run_axis(train_df, eval_df, label_column: str, id_map: dict, names: list[str]) -> dict:
    pipeline = build_pipeline()
    pipeline.fit(train_df["text"], train_df[label_column])
    predictions = pipeline.predict(eval_df["text"])

    report = classification_report(
        eval_df[label_column],
        predictions,
        labels=list(id_map.keys()),
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class_recall": {name: report[name]["recall"] for name in names},
    }


def run(data_csv: str) -> dict:
    train_df = load_split(data_csv, split="train")
    eval_df = load_split(data_csv, split="eval")
    if train_df.empty or eval_df.empty:
        raise ValueError(f"{data_csv} needs both split='train' and split='eval' rows")

    return {
        "severity": run_axis(
            train_df, eval_df, "severity_id", ID_SEVERITY, [s.value for s in Severity]
        ),
        "target": run_axis(train_df, eval_df, "target_id", ID_TARGET, [t.value for t in Target]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to labelled CSV")
    args = parser.parse_args()
    print(json.dumps(run(args.data), indent=2))
