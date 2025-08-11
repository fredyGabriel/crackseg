"""
Static validation of Hydra configs vs. model registries.

Generates: docs/reports/analysis-reports/architecture/hydra_registry_alignment.md

Checks (best-effort, static):
- Hydra component names (by YAML filenames) for encoder/decoder/bottleneck/architectures
- Registered component names discovered via decorators in code
- Reports mismatches: in-config-not-registered and registered-without-config
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from scripts.utils.common.io_utils import write_text  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_ROOT = PROJECT_ROOT / "configs" / "model"
SRC_ROOT = PROJECT_ROOT / "src" / "crackseg" / "model"
REPORT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "reports"
    / "analysis-reports"
    / "architecture"
    / "hydra_registry_alignment.md"
)


@dataclass(frozen=True)
class AlignmentSet:
    hydra: set[str]
    registered: set[str]

    def only_in_hydra(self) -> list[str]:
        return sorted(self.hydra - self.registered)

    def only_in_registry(self) -> list[str]:
        return sorted(self.registered - self.hydra)


def list_yaml_stems(dir_path: Path) -> set[str]:
    if not dir_path.exists():
        return set()
    stems: set[str] = set()
    for p in dir_path.rglob("*.y*ml"):
        stems.add(p.stem)
    return stems


def read_text_files(paths: Iterable[Path]) -> list[tuple[Path, str]]:
    contents: list[tuple[Path, str]] = []
    for p in paths:
        try:
            contents.append(
                (p, p.read_text(encoding="utf-8", errors="ignore"))
            )
        except OSError:
            continue
    return contents


def find_registered_names(src_root: Path, registry_var: str) -> set[str]:
    """Find names registered via @<registry_var>.register(...) decorators.

    Supports two patterns:
    - @encoder_registry.register("name")\nclass ClassName(...)
    - @encoder_registry.register()\nclass ClassName(...)
    In the second case, we use the class name.
    """
    registered: set[str] = set()
    py_files = list(src_root.rglob("*.py"))
    for _path, text in read_text_files(py_files):
        # Iterate through occurrences of the decorator
        for m in re.finditer(
            rf"@{re.escape(registry_var)}\.register\((.*?)\)\s*\n\s*class\s+(\w+)\(",
            text,
            re.DOTALL,
        ):
            args = (m.group(1) or "").strip()
            class_name = m.group(2)
            # Extract explicit name if provided as a string literal
            name_match = re.search(r'^(["\'])([^"\']+)\1', args)
            if name_match:
                registered.add(name_match.group(2))
            else:
                registered.add(class_name)
    return registered


def build_alignment() -> dict[str, AlignmentSet]:
    groups: dict[str, AlignmentSet] = {}
    # Hydra groups
    hydra_arch = list_yaml_stems(CONFIGS_ROOT / "architectures")
    hydra_enc = list_yaml_stems(CONFIGS_ROOT / "encoder")
    hydra_dec = list_yaml_stems(CONFIGS_ROOT / "decoder")
    hydra_btn = list_yaml_stems(CONFIGS_ROOT / "bottleneck")

    # Registry names from code
    registered_arch = find_registered_names(SRC_ROOT, "architecture_registry")
    registered_enc = find_registered_names(SRC_ROOT, "encoder_registry")
    registered_dec = find_registered_names(SRC_ROOT, "decoder_registry")
    registered_btn = find_registered_names(SRC_ROOT, "bottleneck_registry")

    groups["architectures"] = AlignmentSet(
        hydra=hydra_arch, registered=registered_arch
    )
    groups["encoder"] = AlignmentSet(
        hydra=hydra_enc, registered=registered_enc
    )
    groups["decoder"] = AlignmentSet(
        hydra=hydra_dec, registered=registered_dec
    )
    groups["bottleneck"] = AlignmentSet(
        hydra=hydra_btn, registered=registered_btn
    )
    return groups


def render_report(groups: dict[str, AlignmentSet]) -> str:
    lines: list[str] = []
    lines.append("<!-- markdownlint-disable-file -->")
    lines.append("# Hydra & Registry Alignment Report")
    lines.append("")
    for group_name, aset in groups.items():
        lines.append(f"## {group_name.title()}")
        lines.append("")
        lines.append("Summary | Count")
        lines.append(":-- | --:")
        lines.append(f"Hydra entries | {len(aset.hydra)}")
        lines.append(f"Registered | {len(aset.registered)}")
        only_h = aset.only_in_hydra()
        only_r = aset.only_in_registry()
        lines.append(f"Only in Hydra | {len(only_h)}")
        lines.append(f"Only in Registry | {len(only_r)}")
        lines.append("")
        if only_h:
            lines.append("### In Hydra but not Registered")
            for name in only_h:
                lines.append(f"- `{name}`")
            lines.append("")
        if only_r:
            lines.append("### Registered but no Hydra config")
            for name in only_r:
                lines.append(f"- `{name}`")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    groups = build_alignment()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_text(REPORT_PATH, render_report(groups))


if __name__ == "__main__":
    main()
