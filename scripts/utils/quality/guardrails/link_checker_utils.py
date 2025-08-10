from __future__ import annotations

import re


def extract_links_from_markdown_with_lines(
    content: str,
) -> list[tuple[str, str, int]]:
    """Extract markdown links with line numbers.

    Returns list of (text, url, line_number).
    """
    links: list[tuple[str, str, int]] = []
    lines = content.split("\n")
    pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    for line_num, line in enumerate(lines, 1):
        for match in pattern.finditer(line):
            links.append((match.group(1), match.group(2), line_num))
    return links


def extract_links_from_html_with_lines(
    content: str,
) -> list[tuple[str, str, int]]:
    """Extract HTML links with line numbers."""
    links: list[tuple[str, str, int]] = []
    lines = content.split("\n")
    pattern = re.compile(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>')
    for line_num, line in enumerate(lines, 1):
        for match in pattern.finditer(line):
            links.append((match.group(2), match.group(1), line_num))
    return links
