"""False Suppression Rate and Missed Harm Rate — the two numbers this product
is judged on (Document/PROPOSAL.md section 2, Document/ARCHITECTURE.md section 11).

Macro-F1 says how well the model separates classes. Neither of these does.
They ask a narrower question: on the cells where getting it wrong is the whole
point of the project, how often does the model get it wrong, and in which
direction. FSR is the headline; MHR is reported beside it so neither can be
improved by quietly sacrificing the other.

Both rates key off one predicate — `is_hide_marked` — which mirrors
`auto_removable` in kcms-backend's `moderation/pattern_matcher.py`. That is
deliberate: "marked for hiding" has to mean here exactly what it means there,
or the number this repo publishes is about a different system than the one
that runs. If that predicate changes in the backend, it changes here.

Because both rates read the same predicate from opposite sides, abstaining
more moves them in opposite directions by construction: an abstaining verdict
is never hide-marked, so it leaves the FSR numerator and enters the MHR
numerator. ARCHITECTURE.md section 11 requires exactly that coupling — "an
abstention is not a suppression and is still a miss".
"""

from __future__ import annotations

from dataclasses import dataclass

from src.labels import SEVERITY_ID, TARGET_ID, Severity, Target

# Gold severities that still count as legitimate criticism when aimed at an
# institution. PROPOSAL.md section 2 scopes FSR's denominator to "panel-judged
# legitimate institutional criticism"; ARCHITECTURE.md section 11 writes the
# same denominator more loosely as "panel-judged INSTITUTION comments". We
# take the narrower reading: a HARMFUL comment aimed at an institution (a scam
# impersonating it, incitement against its staff) is not criticism, and
# counting it as suppressed would flatter the headline number by padding the
# denominator with rows nobody claims should stay up.
LEGITIMATE_SEVERITIES: frozenset[int] = frozenset(
    {SEVERITY_ID[Severity.SAFE], SEVERITY_ID[Severity.OFFENSIVE]}
)


@dataclass(frozen=True)
class Rate:
    """A rate that refuses to round absence down to zero.

    ARCHITECTURE.md section 11: "A rate with no denominator says what is
    missing rather than showing zero. Zero would claim perfection where there
    is only absence of data, about the one thing this product says it is best
    at." So `value` is None on an empty population and `unavailable_reason`
    carries the explanation to whatever renders it.
    """

    numerator: int
    denominator: int
    population: str

    @property
    def value(self) -> float | None:
        if self.denominator == 0:
            return None
        return self.numerator / self.denominator

    @property
    def unavailable_reason(self) -> str | None:
        if self.denominator > 0:
            return None
        return f"no {self.population} in this evaluation set"

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "population": self.population,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True)
class Prediction:
    """One model verdict, kept as raw argmax + confidence so thresholds can be
    re-applied during calibration without re-running the model."""

    severity_id: int
    target_id: int
    severity_confidence: float
    target_confidence: float

    def abstains(self, severity_threshold: float, target_threshold: float) -> bool:
        return (
            self.severity_confidence < severity_threshold
            or self.target_confidence < target_threshold
        )


def is_hide_marked(
    prediction: Prediction,
    severity_threshold: float,
    target_threshold: float,
    auto_hide_offensive: bool = False,
) -> bool:
    """Whether this verdict would cost the comment its reach without a human
    first agreeing. Mirrors kcms-backend `auto_removable`, whose three clauses
    are load-bearing in the same order:

    * HARMFUL is the precondition, not the rule.
    * Institution-directed criticism is never hidden, however hostile.
    * An abstaining verdict is never hidden — the institution carve-out is
      only as good as the target it reads, so an unread target cannot be
      allowed to bypass it.

    `auto_hide_offensive` is the one per-page policy that widens this, and it
    is off by default (PROPOSAL.md: "Automatic hiding is off by default").
    FSR is reported at the default because it is meant to be
    policy-independent — a page turning this on changes what gets hidden
    without changing what the model said.

    Note one deliberate divergence: the institution and abstention clauses are
    checked before the policy, so they constrain the OFFENSIVE path too. The
    backend's ingest path does not currently do this — its `auto_hide_offensive`
    branch in `moderation/repository.py` tests severity alone — which would hide
    exactly the OFFENSIVE + INSTITUTION comment the product exists to protect.
    We measure against the stated invariant rather than the current code, so
    that if the two disagree the metric reports the gap instead of ratifying it.
    """
    if prediction.abstains(severity_threshold, target_threshold):
        return False
    if prediction.target_id == TARGET_ID[Target.INSTITUTION]:
        return False
    if prediction.severity_id == SEVERITY_ID[Severity.HARMFUL]:
        return True
    if auto_hide_offensive and prediction.severity_id == SEVERITY_ID[Severity.OFFENSIVE]:
        return True
    return False


def is_legitimate_institutional_criticism(severity_id: int, target_id: int) -> bool:
    """FSR's population, read off the panel labels — never off model output."""
    return target_id == TARGET_ID[Target.INSTITUTION] and severity_id in LEGITIMATE_SEVERITIES


def is_targeted_abuse(severity_id: int, target_id: int) -> bool:
    """MHR's population: ARCHITECTURE.md section 11, "panel-judged PERSON +
    HARMFUL comments"."""
    return (
        target_id == TARGET_ID[Target.PERSON]
        and severity_id == SEVERITY_ID[Severity.HARMFUL]
    )


def false_suppression_rate(
    gold_severity: list[int],
    gold_target: list[int],
    predictions: list[Prediction],
    severity_threshold: float,
    target_threshold: float,
    auto_hide_offensive: bool = False,
) -> Rate:
    """Share of panel-judged legitimate institutional criticism the model marks
    for hiding. The headline number."""
    numerator = denominator = 0
    for severity_id, target_id, prediction in zip(
        gold_severity, gold_target, predictions, strict=True
    ):
        if not is_legitimate_institutional_criticism(severity_id, target_id):
            continue
        denominator += 1
        if is_hide_marked(
            prediction, severity_threshold, target_threshold, auto_hide_offensive
        ):
            numerator += 1
    return Rate(numerator, denominator, "panel-judged legitimate institutional criticism")


def missed_harm_rate(
    gold_severity: list[int],
    gold_target: list[int],
    predictions: list[Prediction],
    severity_threshold: float,
    target_threshold: float,
    auto_hide_offensive: bool = False,
) -> Rate:
    """Share of panel-judged targeted abuse the model cleared — where "cleared"
    means "not marked for hiding", so an abstention counts as a miss."""
    numerator = denominator = 0
    for severity_id, target_id, prediction in zip(
        gold_severity, gold_target, predictions, strict=True
    ):
        if not is_targeted_abuse(severity_id, target_id):
            continue
        denominator += 1
        if not is_hide_marked(
            prediction, severity_threshold, target_threshold, auto_hide_offensive
        ):
            numerator += 1
    return Rate(numerator, denominator, "panel-judged targeted abuse (PERSON + HARMFUL)")
