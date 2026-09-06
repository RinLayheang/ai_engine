# KCMS AI Engine

Two-axis moderation classifier for KCMS (Khmer Comment Moderation System):
`severity` (SAFE/OFFENSIVE/HARMFUL) × `target` (NEITHER/PERSON/INSTITUTION),
each with its own confidence, plus abstention. See `../Document/ARCHITECTURE.md`
section 8 (the classifier seam) and `../Document/LABEL-SCHEME (2).md`.

**Status:** scaffolding only. No labelled dataset yet — that's the critical
path (`../Document/PROPOSAL.md` section 3/4). Nothing here has been trained.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate   # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

## Layout

- `src/labels.py` — Severity/Target enums + id maps, mirrors
  `kcms-backend/src/kcms/moderation/contracts.py`
- `src/dataset.py` — loads the labelled CSV (`comment_id,text,severity_id,
  target_id,has_pii,link_flagged,annotator,split,notes`)
- `src/model.py` — XLM-RoBERTa backbone + two independent classification heads
- `src/train.py` — fine-tunes on `split=train` rows
- `src/metrics.py` — False Suppression Rate and Missed Harm Rate, the two
  numbers the product is judged on (ARCHITECTURE.md section 11). The hide
  predicate mirrors kcms-backend's `auto_removable`, so "marked for hiding"
  means here what it means there
- `src/evaluate.py` — macro-F1 + per-class recall on `split=eval` rows, the
  two headline rates above, and abstention threshold calibration against a
  target FSR ceiling (never accuracy alone — see PROPOSAL.md item 5)
- `src/baseline.py` — TF-IDF char n-gram baseline for comparison
- `src/infer.py` — `ModerationClassifier`, output shaped like backend's
  `Verdict`; this is what gets wired into `kcms-backend`'s `Classifier`
  Protocol once trained
- `data/` — real datasets go here, **gitignored** — labelled data never
  enters git (ARCHITECTURE.md section 7)
- `saved_models/` — checkpoints, **gitignored**
- `tests/fixtures/sample_comments.csv` — synthetic rows (not real data) so
  the pipeline can be tested without a real dataset

## Usage

```bash
python -m src.train --data data/comments.csv --out saved_models/v1
python -m src.evaluate --data data/comments.csv --checkpoint saved_models/v1
python -m src.baseline --data data/comments.csv
python -m src.infer --checkpoint saved_models/v1 --text "..."
```

`evaluate` takes `--max-fsr` (default `0.05`), the False Suppression Rate
ceiling the calibrated thresholds must hold. It picks the threshold pair with
the lowest Missed Harm Rate among those that hold it, and reports
`calibrated: false` with a reason rather than a number it cannot justify when
no pair holds the ceiling or when the eval split contains no rows in a
population. A rate with no denominator reports `value: null` and an
`unavailable_reason` — never `0.0`, which would claim perfection on the
strength of no data.

## Tests

```bash
pytest
```

`test_labels.py`, `test_dataset.py` and `test_metrics.py` run offline —
`test_metrics.py` builds verdicts by hand, so the product invariants (an
institution is never marked for hiding however hostile, an uncertain verdict
is never acted on, an abstention is not a suppression but is still a miss) are
tested without a model. `test_pipeline_smoke.py` downloads `xlm-roberta-base`
and skips if unavailable.
