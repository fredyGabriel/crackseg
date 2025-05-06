import pytest
from src.data.validation import validate_data_config, validate_transform_config


def test_validate_data_config_valid():
    valid_cfg = {
        'data_root': 'data/',
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'image_size': [256, 256]
    }
    validate_data_config(valid_cfg)  # No debe lanzar excepción


def test_validate_data_config_missing_key():
    invalid_cfg = {
        'data_root': 'data/',
        'train_split': 0.7,
        'val_split': 0.2,
        # Falta 'test_split' y 'image_size'
    }
    with pytest.raises(ValueError):
        validate_data_config(invalid_cfg)


def test_validate_transform_config_valid():
    valid_transforms = [
        {'name': 'Resize', 'params': {'size': [256, 256]}},
        {'name': 'Normalize', 'params': {
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5]
        }}
    ]
    validate_transform_config(valid_transforms)  # No debe lanzar excepción


def test_validate_transform_config_missing_resize():
    invalid_transforms = [
        {'name': 'Normalize', 'params': {
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5]
        }}
    ]
    with pytest.raises(ValueError):
        validate_transform_config(invalid_transforms)
