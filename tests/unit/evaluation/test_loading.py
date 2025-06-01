from pathlib import Path

import pytest
import torch

from src.evaluation.loading import load_model_from_checkpoint


def test_load_model_from_checkpoint_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_model_from_checkpoint(
            "nonexistent_checkpoint.pth", torch.device("cpu")
        )


def test_load_model_from_checkpoint_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Crear archivo dummy para simular checkpoint corrupto
    corrupt_ckpt = tmp_path / "corrupt.pth"
    corrupt_ckpt.write_bytes(b"not a real checkpoint")
    # Simular torch.load lanzando excepción
    monkeypatch.setattr(
        "torch.load",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("corrupt")),
    )
    with pytest.raises(RuntimeError):
        load_model_from_checkpoint(str(corrupt_ckpt), torch.device("cpu"))
