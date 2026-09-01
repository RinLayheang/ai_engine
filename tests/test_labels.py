from src.labels import ID_SEVERITY, ID_TARGET, SEVERITY_ID, TARGET_ID, Severity, Target


def test_severity_ids_match_label_scheme_doc():
    assert SEVERITY_ID == {Severity.SAFE: 0, Severity.OFFENSIVE: 1, Severity.HARMFUL: 2}


def test_target_ids_match_label_scheme_doc():
    assert TARGET_ID == {Target.NEITHER: 0, Target.PERSON: 1, Target.INSTITUTION: 2}


def test_id_maps_are_inverses():
    assert all(ID_SEVERITY[SEVERITY_ID[label]] == label for label in Severity)
    assert all(ID_TARGET[TARGET_ID[label]] == label for label in Target)
