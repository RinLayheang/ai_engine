"""End-to-end smoke test on the synthetic fixture. Needs xlm-roberta-base
downloaded (or cached) from Hugging Face Hub — skips if unavailable rather
than failing, since that's an environment issue, not a code bug.
"""

from pathlib import Path

import pytest

from src.config import Config

FIXTURE = Path(__file__).parent / "fixtures" / "sample_comments.csv"


@pytest.fixture(scope="module")
def tiny_config() -> Config:
    return Config(model_name="xlm-roberta-base", batch_size=2, epochs=1, max_length=32)


def test_train_and_infer_round_trip(tmp_path, tiny_config):
    transformers = pytest.importorskip("transformers")
    try:
        transformers.AutoTokenizer.from_pretrained(tiny_config.model_name)
    except OSError:
        pytest.skip("xlm-roberta-base not available offline in this environment")

    from src.infer import ModerationClassifier
    from src.train import train

    checkpoint_dir = tmp_path / "checkpoint"
    train(tiny_config, str(FIXTURE), str(checkpoint_dir))

    classifier = ModerationClassifier(str(checkpoint_dir))
    result = classifier.classify("Why is the queue always this slow?")

    assert result["severity"] in {"SAFE", "OFFENSIVE", "HARMFUL"}
    assert result["target"] in {"NEITHER", "PERSON", "INSTITUTION"}
    assert 0.0 <= result["severity_confidence"] <= 1.0
    assert 0.0 <= result["target_confidence"] <= 1.0
    assert isinstance(result["abstain"], bool)
