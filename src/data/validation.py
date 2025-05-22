import warnings
from typing import Any, cast

from omegaconf import DictConfig, ListConfig


def validate_data_config(data_cfg: dict[str, Any] | DictConfig) -> None:
    """
    Validates the dataset configuration dictionary.
    Raises ValueError if required parameters are missing or invalid.
    """
    required_keys = [
        "data_root",
        "train_split",
        "val_split",
        "test_split",
        "image_size",
    ]
    for key in required_keys:
        if key not in data_cfg:
            raise ValueError(f"Missing required data config key: '{key}'")
    # Check split ratios sum to 1.0 (allowing small float error)
    train_split_val = data_cfg["train_split"]
    val_split_val = data_cfg["val_split"]
    test_split_val = data_cfg["test_split"]

    total = (
        float(train_split_val) + float(val_split_val) + float(test_split_val)
    )
    if not abs(total - 1.0) < 1e-4:  # noqa: PLR2004
        raise ValueError(f"train/val/test splits must sum to 1.0, got {total}")
    # Check image_size is a list/tuple of length 2
    img_size = data_cfg["image_size"]
    img_size_for_len: list[Any] | tuple[Any, ...] | ListConfig = img_size
    if isinstance(img_size, ListConfig):
        img_size_for_len = list(img_size)

    if not (
        isinstance(img_size, list | tuple | ListConfig)
        and len(img_size_for_len) == 2  # noqa: PLR2004
    ):
        raise ValueError(
            "image_size must be a list, tuple, or ListConfig of length 2"
        )


def _normalize_transform_config_input(
    transform_config: (
        list[dict[str, Any] | DictConfig]
        | ListConfig
        | dict[str, Any]
        | DictConfig
    ),
) -> list[dict[str, Any] | DictConfig]:
    """Normalizes the transform_config input into a list of transform dicts."""
    actual_transform_list: list[dict[str, Any] | DictConfig]
    if isinstance(transform_config, list | ListConfig):
        actual_transform_list = []
        for item_any in transform_config:
            if not isinstance(item_any, dict | DictConfig):
                raise ValueError(
                    "If transform_config is a list, its items must be dicts or"
                    "DictConfigs."
                )
            # Ensure correct type for the list items
            if isinstance(item_any, DictConfig):
                actual_transform_list.append(item_any)
            else:
                # We cast here after ensuring it's a dict
                actual_transform_list.append(cast(dict[str, Any], item_any))
    elif isinstance(transform_config, dict | DictConfig):
        actual_transform_list = []
        for name, params_data in transform_config.items():
            # Correctly handle DictConfig items if transform_config is a
            # DictConfig
            current_params = params_data if params_data is not None else {}
            actual_transform_list.append(
                {"name": str(name), "params": current_params}
            )
    else:
        raise ValueError(
            "Transform config must be a list, ListConfig, dict, or DictConfig."
        )
    return actual_transform_list


def _validate_resize_params(params: dict[str, Any] | DictConfig) -> None:
    """Validates parameters for a Resize transform."""
    if "size" not in params and (
        "height" not in params or "width" not in params
    ):
        raise ValueError(
            "Resize transform must have either 'size' or both "
            "'height' and 'width' parameters."
        )
    if "size" in params:
        size_val = params["size"]
        size_val_for_len: list[Any] | tuple[Any, ...] | ListConfig = size_val
        if isinstance(size_val, ListConfig):
            size_val_for_len = list(size_val)
        if not (
            isinstance(size_val, list | ListConfig | tuple)
            and len(size_val_for_len) == 2  # noqa: PLR2004
        ):
            raise ValueError(
                "Resize 'size' must be a list, tuple, or ListConfig of length"
                "2."
            )
    if "height" in params and "width" in params:
        height_val = params["height"]
        width_val = params["width"]
        if not (
            isinstance(height_val, int | float)
            and isinstance(width_val, int | float)
        ):
            raise ValueError(
                "Resize 'height' and 'width' must be numeric values."
            )


def _validate_normalize_params(params: dict[str, Any] | DictConfig) -> None:
    """Validates parameters for a Normalize transform."""
    if "mean" not in params or "std" not in params:
        raise ValueError("Missing 'mean' or 'std' in Normalize params.")
    mean_val = params["mean"]
    std_val = params["std"]
    mean_val_for_len: list[Any] | tuple[Any, ...] | ListConfig = mean_val
    if isinstance(mean_val, ListConfig):
        mean_val_for_len = list(mean_val)
    std_val_for_len: list[Any] | tuple[Any, ...] | ListConfig = std_val
    if isinstance(std_val, ListConfig):
        std_val_for_len = list(std_val)
    if not (
        isinstance(mean_val, list | ListConfig | tuple)
        and len(mean_val_for_len) == 3  # noqa: PLR2004
    ):
        raise ValueError(
            "Normalize 'mean' must be list, tuple or ListConfig of 3 values."
        )
    if not (
        isinstance(std_val, list | ListConfig | tuple)
        and len(std_val_for_len) == 3  # noqa: PLR2004
    ):
        raise ValueError(
            "Normalize 'std' must be list, tuple or ListConfig of 3 values."
        )


def validate_transform_config(
    transform_config: (
        list[dict[str, Any] | DictConfig]
        | ListConfig
        | dict[str, Any]
        | DictConfig
    ),
    mode_unused: (
        str | None
    ) = None,  # Mode parameter is not used in the current logic
) -> None:
    """Validates a list or dict of transform configurations.

    Checks if 'Resize' and 'Normalize' transforms are present with correct
    params.

    Args:
        transform_config: List/ListConfig of transform configs, or
        Dict/DictConfig where keys are transform names and values are params.
        mode_unused: 'train', 'val', or 'test'. Not currently used in
        validation logic.
    """
    actual_transform_list = _normalize_transform_config_input(transform_config)

    if not actual_transform_list:
        warnings.warn(
            "The transform configuration list is empty. Resize and Normalize "
            "checks might fail if they are mandatory.",
            stacklevel=2,
        )
        # Depending on requirements, you might return here if an empty list is
        # acceptable and doesn't require Resize/Normalize to be present.

    resize_found = False
    normalize_found = False

    for transform_item in actual_transform_list:
        name_any = transform_item.get("name")
        name = str(name_any) if name_any is not None else None
        # Ensure params is a dict, handling cases where it might be None from
        # DictConfig
        params_from_item = transform_item.get("params", {})
        params: dict[str, Any] | DictConfig = (
            params_from_item if params_from_item is not None else {}
        )

        if name == "Resize":
            resize_found = True
            _validate_resize_params(params)
        elif name == "Normalize":
            normalize_found = True
            _validate_normalize_params(params)

    # These checks are now conditional on actual_transform_list not being
    # empty, or they can be made stricter if Resize/Normalize are always
    # mandatory.
    if not resize_found and actual_transform_list:
        raise ValueError("Missing 'Resize' transform in the list.")
    if not normalize_found and actual_transform_list:
        raise ValueError("Missing 'Normalize' transform in the list.")
