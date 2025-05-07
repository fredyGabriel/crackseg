import pytest
import yaml
from src.evaluation.results import save_evaluation_results


def test_save_evaluation_results_unwritable_dir(tmp_path, monkeypatch):
    # Simular error de E/S al abrir el archivo
    def raise_ioerror(*args, **kwargs):
        raise IOError("Permission denied")
    monkeypatch.setattr("builtins.open", raise_ioerror)
    results = {'test_metric': 1.0}
    config = {'foo': 'bar'}
    checkpoint = 'ckpt.pth.tar'
    with pytest.raises(IOError):
        save_evaluation_results(results, config, checkpoint, str(tmp_path))


def test_save_evaluation_results_non_serializable(tmp_path, monkeypatch):
    # Simular error de serialización en yaml.dump
    def raise_yaml_error(*args, **kwargs):
        raise yaml.YAMLError("Serialization failed")
    monkeypatch.setattr("yaml.dump", raise_yaml_error)
    results = {'test_metric': object()}
    config = {'foo': 'bar'}
    checkpoint = 'ckpt.pth.tar'
    with pytest.raises(yaml.YAMLError):
        save_evaluation_results(results, config, checkpoint, str(tmp_path))
