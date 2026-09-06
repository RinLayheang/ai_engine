"""Tests for the two rates the product is judged on.

These assert the invariants Document/PROPOSAL.md and Document/ARCHITECTURE.md
state as promises — institutional criticism is never marked for hiding however
hostile, an uncertain verdict is never acted on, an abstention is not a
suppression but is still a miss, and a rate with no denominator reports its
absence rather than zero. They are written against hand-built predictions so
they run offline, with no model and no download.
"""

from __future__ import annotations

import pytest

from src.labels import SEVERITY_ID, TARGET_ID, Severity, Target
from src.metrics import (
    Prediction,
    Rate,
    false_suppression_rate,
    is_hide_marked,
    is_legitimate_institutional_criticism,
    is_targeted_abuse,
    missed_harm_rate,
)

CONFIDENT = 0.99
UNSURE = 0.10
SEVERITY_THRESHOLD = 0.6
TARGET_THRESHOLD = 0.6

SAFE = SEVERITY_ID[Severity.SAFE]
OFFENSIVE = SEVERITY_ID[Severity.OFFENSIVE]
HARMFUL = SEVERITY_ID[Severity.HARMFUL]
NEITHER = TARGET_ID[Target.NEITHER]
PERSON = TARGET_ID[Target.PERSON]
INSTITUTION = TARGET_ID[Target.INSTITUTION]


def predict(
    severity_id: int,
    target_id: int,
    severity_confidence: float = CONFIDENT,
    target_confidence: float = CONFIDENT,
) -> Prediction:
    return Prediction(severity_id, target_id, severity_confidence, target_confidence)


def hide_marked(prediction: Prediction, **kwargs) -> bool:
    return is_hide_marked(prediction, SEVERITY_THRESHOLD, TARGET_THRESHOLD, **kwargs)


# --- the hide predicate, mirroring kcms-backend `auto_removable` -------------


def test_confident_harmful_at_a_person_is_marked_for_hiding():
    assert hide_marked(predict(HARMFUL, PERSON)) is True


def test_institution_is_never_marked_for_hiding_however_hostile():
    """PROPOSAL.md section 2: "Never auto-hides institutional criticism, at any
    confidence, however hostile." Maximum severity at maximum confidence is the
    strongest case that rule has to survive."""
    assert hide_marked(predict(HARMFUL, INSTITUTION)) is False


def test_uncertain_target_is_never_marked_for_hiding():
    """The institution carve-out is only as good as the target it reads, so a
    verdict unsure of the target may not be acted on — otherwise the carve-out
    is bypassed by ignorance rather than overruled."""
    assert hide_marked(predict(HARMFUL, PERSON, target_confidence=UNSURE)) is False


def test_uncertain_severity_is_never_marked_for_hiding():
    assert hide_marked(predict(HARMFUL, PERSON, severity_confidence=UNSURE)) is False


def test_offensive_is_not_marked_for_hiding_by_default():
    """Automatic hiding of OFFENSIVE is a page policy, off by default."""
    assert hide_marked(predict(OFFENSIVE, PERSON)) is False


def test_offensive_is_marked_for_hiding_only_under_page_policy():
    assert hide_marked(predict(OFFENSIVE, PERSON), auto_hide_offensive=True) is True


def test_institution_carve_out_outranks_the_offensive_policy():
    """A page enabling auto-hide for OFFENSIVE does not thereby buy the right
    to hide institutional criticism — the carve-out is a product invariant, not
    a default. Note this is stricter than kcms-backend's ingest path currently
    is; see the note in `src.metrics.is_hide_marked`."""
    assert hide_marked(predict(OFFENSIVE, INSTITUTION), auto_hide_offensive=True) is False


def test_the_forcing_case_from_the_label_scheme():
    """LABEL-SCHEME.md: identical hostility, opposite correct outcomes.

        "Sophea is a thief, she stole my money"  OFFENSIVE PERSON       review
        "ABA are thieves, they stole my money"   OFFENSIVE INSTITUTION  leave up

    Under a page that auto-hides OFFENSIVE, only the first may be touched.
    """
    at_a_person = predict(OFFENSIVE, PERSON)
    at_an_institution = predict(OFFENSIVE, INSTITUTION)

    assert hide_marked(at_a_person, auto_hide_offensive=True) is True
    assert hide_marked(at_an_institution, auto_hide_offensive=True) is False


# --- populations are read off the panel labels, never off model output ------


def test_fsr_population_is_institution_minus_harmful():
    assert is_legitimate_institutional_criticism(SAFE, INSTITUTION) is True
    assert is_legitimate_institutional_criticism(OFFENSIVE, INSTITUTION) is True
    assert is_legitimate_institutional_criticism(HARMFUL, INSTITUTION) is False
    assert is_legitimate_institutional_criticism(OFFENSIVE, PERSON) is False


def test_mhr_population_is_person_and_harmful():
    assert is_targeted_abuse(HARMFUL, PERSON) is True
    assert is_targeted_abuse(HARMFUL, INSTITUTION) is False
    assert is_targeted_abuse(OFFENSIVE, PERSON) is False


def test_fsr_denominator_ignores_what_the_model_predicted():
    """A row belongs to the population because the panel put it there. If the
    model's own verdict decided membership, a model that never predicted
    INSTITUTION would report a perfect FSR over an empty set."""
    gold_severity = [OFFENSIVE]
    gold_target = [INSTITUTION]
    predictions = [predict(HARMFUL, PERSON)]  # model disagrees on both axes

    rate = false_suppression_rate(
        gold_severity, gold_target, predictions, SEVERITY_THRESHOLD, TARGET_THRESHOLD
    )

    assert rate.denominator == 1
    assert rate.numerator == 1
    assert rate.value == 1.0


# --- the coupling between the two rates -------------------------------------


def test_abstention_is_not_a_suppression_but_is_still_a_miss():
    """ARCHITECTURE.md section 11. The same abstaining verdict must leave the
    FSR numerator and enter the MHR numerator, which is what stops either rate
    being improved by quietly sacrificing the other."""
    abstaining = predict(HARMFUL, PERSON, severity_confidence=UNSURE)

    suppression = false_suppression_rate(
        [OFFENSIVE], [INSTITUTION], [abstaining], SEVERITY_THRESHOLD, TARGET_THRESHOLD
    )
    missed = missed_harm_rate(
        [HARMFUL], [PERSON], [abstaining], SEVERITY_THRESHOLD, TARGET_THRESHOLD
    )

    assert suppression.numerator == 0
    assert missed.numerator == 1


def test_raising_thresholds_never_raises_fsr_and_never_lowers_mhr():
    """Raising a threshold can only turn a hide-marked verdict into an
    abstention, never the reverse — so the two rates move in opposite
    directions by construction, and no threshold change improves both."""
    gold_severity = [OFFENSIVE, SAFE, HARMFUL, HARMFUL, OFFENSIVE, HARMFUL]
    gold_target = [INSTITUTION, INSTITUTION, PERSON, PERSON, INSTITUTION, PERSON]
    predictions = [
        predict(HARMFUL, PERSON, 0.55, 0.95),
        predict(HARMFUL, PERSON, 0.75, 0.65),
        predict(HARMFUL, PERSON, 0.95, 0.85),
        predict(SAFE, NEITHER, 0.45, 0.45),
        predict(HARMFUL, NEITHER, 0.85, 0.35),
        predict(HARMFUL, PERSON, 0.65, 0.75),
    ]

    grid = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    previous_fsr, previous_mhr = 1.0, 0.0
    for threshold in grid:
        fsr = false_suppression_rate(
            gold_severity, gold_target, predictions, threshold, threshold
        )
        mhr = missed_harm_rate(gold_severity, gold_target, predictions, threshold, threshold)

        assert fsr.value <= previous_fsr
        assert mhr.value >= previous_mhr
        previous_fsr, previous_mhr = fsr.value, mhr.value

    # At a threshold nothing clears, everything abstains: nothing is suppressed
    # and every harm is missed.
    assert previous_fsr == 0.0
    assert previous_mhr == 1.0


# --- absence of data is reported, not rounded down --------------------------


def test_a_rate_with_no_denominator_is_none_rather_than_zero():
    empty = Rate(numerator=0, denominator=0, population="panel-judged targeted abuse")

    assert empty.value is None
    assert empty.unavailable_reason == "no panel-judged targeted abuse in this evaluation set"
    assert empty.as_dict()["value"] is None


def test_a_rate_with_a_denominator_has_no_unavailable_reason():
    assert Rate(1, 4, "x").value == 0.25
    assert Rate(1, 4, "x").unavailable_reason is None


def test_missed_harm_rate_reports_absence_when_no_targeted_abuse_was_labelled():
    rate = missed_harm_rate(
        [SAFE], [INSTITUTION], [predict(SAFE, INSTITUTION)], SEVERITY_THRESHOLD, TARGET_THRESHOLD
    )

    assert rate.denominator == 0
    assert rate.value is None
    assert rate.unavailable_reason is not None


# --- calibration ------------------------------------------------------------


def test_calibration_refuses_to_pick_thresholds_off_an_empty_population():
    """A rate with no denominator cannot constrain a threshold, so calibration
    says so instead of returning a number it cannot justify."""
    from src.evaluate import calibrate_thresholds

    result = calibrate_thresholds(
        [HARMFUL], [PERSON], [predict(HARMFUL, PERSON)], max_false_suppression_rate=0.05
    )

    assert result["calibrated"] is False
    assert result["severity_threshold"] is None
    assert "False Suppression Rate" in result["reason"]


def test_calibration_picks_a_pair_that_holds_the_suppression_ceiling():
    from src.evaluate import calibrate_thresholds

    # One institutional row the model would suppress unless it abstains, and
    # one targeted abuse row it catches confidently.
    gold_severity = [OFFENSIVE, HARMFUL]
    gold_target = [INSTITUTION, PERSON]
    predictions = [predict(HARMFUL, PERSON, 0.55, 0.55), predict(HARMFUL, PERSON, 0.99, 0.99)]

    result = calibrate_thresholds(
        gold_severity, gold_target, predictions, max_false_suppression_rate=0.0
    )

    assert result["calibrated"] is True
    assert result["false_suppression_rate"] == 0.0
    assert result["missed_harm_rate"] == 0.0

    # Either axis may be the one that rises — what matters is that the chosen
    # pair abstains on the row it would otherwise suppress while still catching
    # the confident harm.
    chosen = (result["severity_threshold"], result["target_threshold"])
    assert predictions[0].abstains(*chosen) is True
    assert predictions[1].abstains(*chosen) is False


def test_calibration_reports_when_no_pair_can_hold_the_ceiling():
    from src.evaluate import calibrate_thresholds

    # A confidently wrong suppression: no threshold in the grid abstains on it
    # without also abstaining on everything.
    result = calibrate_thresholds(
        [OFFENSIVE], [INSTITUTION], [predict(HARMFUL, PERSON, 0.999, 0.999)],
        max_false_suppression_rate=0.05,
    )

    assert result["calibrated"] is False
    assert "no threshold pair" in result["reason"]


@pytest.mark.parametrize("auto_hide_offensive", [False, True])
def test_hide_predicate_is_stable_under_the_page_policy_for_institutions(auto_hide_offensive):
    for severity_id in (SAFE, OFFENSIVE, HARMFUL):
        assert (
            hide_marked(predict(severity_id, INSTITUTION), auto_hide_offensive=auto_hide_offensive)
            is False
        )
