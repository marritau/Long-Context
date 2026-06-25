"""Simple structure parser for markdown-like documents."""

from __future__ import annotations

import re

from pearl.data.schema import Section

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", flags=re.MULTILINE)


def parse_markdown_sections(text: str) -> list[Section]:
    """Split a markdown-like document into sections.

    If no headings are found, a single untitled section is returned.
    """

    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [Section(title=None, text=text, section_path=[], char_start=0, char_end=len(text))]

    sections: list[Section] = []
    path: list[str] = []
    for idx, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        path = path[: level - 1] + [title]
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append(
            Section(
                title=title,
                text=body,
                section_path=path.copy(),
                char_start=start,
                char_end=end,
            )
        )
    return sections
