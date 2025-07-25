from typing import Any

import pytest

from crackseg.evaluation.data import get_evaluation_dataloader


def test_get_evaluation_dataloader_invalid_config():
    # Config sin la clave esperada
    cfg: dict[str, Any] = {}
    with pytest.raises(KeyError):
        get_evaluation_dataloader(cfg)
