from pathlib import Path

from bikv.ruler_adapter import RULER_QUERY_MARKER
from scripts.prepare_ruler_vt import ANSWER_PREFIX, TEMPLATE, command_for


def test_ruler_template_preserves_query_marker_before_upstream_answer_prefix() -> None:
    assert TEMPLATE.count(RULER_QUERY_MARKER) == 1
    assert TEMPLATE.index(RULER_QUERY_MARKER) < TEMPLATE.index(ANSWER_PREFIX)
    assert "{context}" in TEMPLATE and "{query}" in TEMPLATE and "{num_v}" in TEMPLATE


def test_ruler_command_is_pinned_and_explicit() -> None:
    command = command_for(Path("repo"), Path("out"), "model", 4096, 50, 17)
    assert command[1].endswith("variable_tracking.py")
    assert command[command.index("--max_seq_length") + 1] == "4096"
    assert command[command.index("--num_samples") + 1] == "50"
    assert command[command.index("--template") + 1] == TEMPLATE
