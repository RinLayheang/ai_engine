from pathlib import Path

import pytest

from src.dataset import load_split

FIXTURE = Path(__file__).parent / "fixtures" / "sample_comments.csv"


def test_load_split_filters_by_split_column():
    train = load_split(FIXTURE, split="train")
    eval_ = load_split(FIXTURE, split="eval")

    assert len(train) == 4
    assert len(eval_) == 3
    assert set(train["split"]) == {"train"}
    assert set(eval_["split"]) == {"eval"}


def test_load_split_rejects_missing_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("comment_id,text\n1,hello\n")

    with pytest.raises(ValueError, match="missing required columns"):
        load_split(bad_csv, split="train")
