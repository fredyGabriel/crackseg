# Re-export dataclasses functionality for compatibility
from dataclasses import (
    MISSING,
    Field,
    asdict,
    astuple,
    dataclass,
    field,
    fields,
    make_dataclass,
    replace,
)

__all__ = [
    "asdict",
    "astuple",
    "dataclass",
    "field",
    "fields",
    "replace",
    "make_dataclass",
    "MISSING",
    "Field",
]
