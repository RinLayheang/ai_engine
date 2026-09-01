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
- `src/evaluate.py` — macro-F1 + per-class recall on `split=eval` rows, plus
  abstention threshold calibration (never accuracy alone — see PROPOSAL.md
  item 5)
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

## Tests

```bash
pytest
```

`test_labels.py` and `test_dataset.py` run offline. `test_pipeline_smoke.py`
downloads `xlm-roberta-base` and skips if unavailable.
