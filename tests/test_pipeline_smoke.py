"""End-to-end smoke test on the synthetic fixture. Needs xlm-roberta-base
downloaded (or cached) from Hugging Face Hub — skips if unavailable rather
than failing, since that's an environment issue, not a code bug.

Nothing here asserts the model is any good; four synthetic rows cannot support
that claim. It asserts the pipeline holds together and that every number
`evaluate` publishes is shaped the way the metrics contract requires.
"""

from pathlib import Path

import pytest

from src.config import Config

FIXTURE = Path(__file__).parent / "fixtures" / "sample_comments.csv"


@pytest.fixture(scope="module")
def tiny_config() -> Config:
    return Config(model_name="xlm-roberta-base", batch_size=2, epochs=1, max_length=32)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory, tiny_config) -> Path:
    """Train once and share the checkpoint — a second fine-tune would buy no
    coverage and cost another minute."""
    transformers = pytest.importorskip("transformers")
    try:
        transformers.AutoTokenizer.from_pretrained(tiny_config.model_name)
    except OSError:
        pytest.skip("xlm-roberta-base not available offline in this environment")

    from src.train import train

    checkpoint_dir = tmp_path_factory.mktemp("checkpoint")
    train(tiny_config, str(FIXTURE), str(checkpoint_dir))
    return checkpoint_dir


def test_train_and_infer_round_trip(checkpoint):
    from src.infer import ModerationClassifier

    classifier = ModerationClassifier(str(checkpoint))
    result = classifier.classify("Why is the queue always this slow?")

    assert result["severity"] in {"SAFE", "OFFENSIVE", "HARMFUL"}
    assert result["target"] in {"NEITHER", "PERSON", "INSTITUTION"}
    assert 0.0 <= result["severity_confidence"] <= 1.0
    assert 0.0 <= result["target_confidence"] <= 1.0
    assert isinstance(result["abstain"], bool)


def test_evaluate_reports_both_headline_rates_with_their_denominators(checkpoint, tiny_config):
    """The fixture's eval split carries one row in each headline population, so
    both rates must come back with a denominator rather than an absence."""
    from src.evaluate import evaluate

    results = evaluate(tiny_config, str(FIXTURE), str(checkpoint))

    suppression = results["false_suppression_rate"]
    missed = results["missed_harm_rate"]

    assert suppression["denominator"] == 1  # synthetic-6, SAFE + INSTITUTION
    assert missed["denominator"] == 1  # synthetic-7, HARMFUL + PERSON
    assert 0.0 <= suppression["value"] <= 1.0
    assert 0.0 <= missed["value"] <= 1.0
    assert suppression["unavailable_reason"] is None
    assert missed["unavailable_reason"] is None

    # Reported thresholds are the ones the rates were computed at, whether they
    # came from calibration or from config.
    assert results["thresholds"]["source"] in {"calibrated", "config default"}
    assert results["eval_rows"] == 3


def test_evaluate_still_reports_per_class_recall_not_only_the_rates(checkpoint, tiny_config):
    """PROPOSAL.md item 5 asks for macro-F1 and per-class recall. The headline
    rates are added alongside those, not in place of them."""
    from src.evaluate import evaluate

    results = evaluate(tiny_config, str(FIXTURE), str(checkpoint))

    assert set(results["severity"]["per_class_recall"]) == {"SAFE", "OFFENSIVE", "HARMFUL"}
    assert set(results["target"]["per_class_recall"]) == {"NEITHER", "PERSON", "INSTITUTION"}
    assert 0.0 <= results["severity"]["macro_f1"] <= 1.0
    assert 0.0 <= results["target"]["macro_f1"] <= 1.0
