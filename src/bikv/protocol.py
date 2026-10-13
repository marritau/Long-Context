from __future__ import annotations


def comparison_allowed(gate_passed: bool, rescue_enabled: bool) -> bool:
    """Allow a preregistered comparison or an explicitly labelled rescue analysis."""
    return bool(gate_passed or rescue_enabled)
