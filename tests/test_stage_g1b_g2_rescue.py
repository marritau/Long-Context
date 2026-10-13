from bikv.protocol import comparison_allowed


def test_comparison_allowed_preserves_gate_and_requires_explicit_rescue() -> None:
    assert comparison_allowed(True, False) is True
    assert comparison_allowed(True, True) is True
    assert comparison_allowed(False, False) is False
    assert comparison_allowed(False, True) is True
