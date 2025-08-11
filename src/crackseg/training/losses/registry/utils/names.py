"""Name similarity utilities for the loss registry."""

from __future__ import annotations


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def get_similar_names(
    available: list[str], name: str, max_suggestions: int = 3
) -> list[str]:
    """Return similar names based on containment and Levenshtein distance."""
    similar: list[str] = []
    name_lower = name.lower()
    for available_name in available:
        available_lower = available_name.lower()
        if (
            name_lower in available_lower
            or available_lower in name_lower
            or levenshtein_distance(name_lower, available_lower) <= 2
        ):
            similar.append(available_name)
    return similar[:max_suggestions]
