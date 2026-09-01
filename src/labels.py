"""Label scheme for the KCMS moderation classifier.

Mirrors `kcms-backend/src/kcms/moderation/contracts.py` (Severity, Target) and
`Document/LABEL-SCHEME (2).md` (label ids). Keep the enum members and id maps
in sync with both if either changes.
"""

from __future__ import annotations

from enum import StrEnum


class Severity(StrEnum):
    SAFE = "SAFE"
    OFFENSIVE = "OFFENSIVE"
    HARMFUL = "HARMFUL"


class Target(StrEnum):
    NEITHER = "NEITHER"
    PERSON = "PERSON"
    INSTITUTION = "INSTITUTION"


# id <-> label, per Document/LABEL-SCHEME (2).md `data/severity_map.csv` and
# `data/target_map.csv`. Training data stores ids; inference output uses names.
SEVERITY_ID: dict[Severity, int] = {
    Severity.SAFE: 0,
    Severity.OFFENSIVE: 1,
    Severity.HARMFUL: 2,
}
TARGET_ID: dict[Target, int] = {
    Target.NEITHER: 0,
    Target.PERSON: 1,
    Target.INSTITUTION: 2,
}

ID_SEVERITY: dict[int, Severity] = {v: k for k, v in SEVERITY_ID.items()}
ID_TARGET: dict[int, Target] = {v: k for k, v in TARGET_ID.items()}
