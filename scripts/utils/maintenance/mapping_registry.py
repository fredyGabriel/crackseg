"""
Central mapping registry for refactor moves: old import paths -> new import paths.

Used to update docs, configs, and code references and to provide temporary re-exports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.utils.common.io_utils import read_text, write_text  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAP_PATH = (
    PROJECT_ROOT
    / "docs"
    / "reports"
    / "analysis-reports"
    / "architecture"
    / "import_mapping.json"
)


@dataclass
class Mapping:
    old: str
    new: str


def load_mapping() -> list[Mapping]:
    if MAP_PATH.exists():
        import json

        data = json.loads(read_text(MAP_PATH))
        return [Mapping(**item) for item in data]
    return []


def save_mapping(mappings: list[Mapping]) -> None:
    import json

    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_text(
        MAP_PATH,
        json.dumps([m.__dict__ for m in mappings], indent=2),
    )


def add_mapping(old: str, new: str) -> None:
    mappings = load_mapping()
    # Deduplicate
    if not any(m.old == old and m.new == new for m in mappings):
        mappings.append(Mapping(old=old, new=new))
        save_mapping(mappings)


if __name__ == "__main__":
    # Example usage
    add_mapping(
        "crackseg.model.decoder.cnn_decoder.Decoder",
        "crackseg.model.decoder.decoder_head.Decoder",
    )
    print(MAP_PATH)
