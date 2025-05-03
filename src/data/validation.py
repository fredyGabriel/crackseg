from omegaconf import ListConfig, DictConfig


def validate_data_config(data_cfg):
    """
    Validates the dataset configuration dictionary.
    Raises ValueError if required parameters are missing or invalid.
    """
    required_keys = [
        "data_root", "train_split", "val_split", "test_split", "image_size"
    ]
    for key in required_keys:
        if key not in data_cfg:
            raise ValueError(f"Missing required data config key: '{key}'")
    # Check split ratios sum to 1.0 (allowing small float error)
    total = (
        float(data_cfg["train_split"]) +
        float(data_cfg["val_split"]) +
        float(data_cfg["test_split"])
    )
    if not abs(total - 1.0) < 1e-4:
        raise ValueError(f"train/val/test splits must sum to 1.0, got {total}")
    # Check image_size is a list/tuple of length 2
    img_size = data_cfg["image_size"]
    if not (isinstance(img_size, (list, tuple)) and len(img_size) == 2):
        raise ValueError("image_size must be a list or tuple of length 2")


def validate_transform_config(transform_list_cfg):
    """Validates a list of transform configurations for a specific split.

    Checks if 'Resize' and 'Normalize' transforms are present with correct
    params.

    Args:
        transform_list_cfg: List of transform configs for one split
                            (e.g., cfg.data.transforms.train)
    """
    if not isinstance(transform_list_cfg, (list, ListConfig)):
        raise ValueError("Transform config must be a list for a split.")

    resize_found = False
    normalize_found = False

    for transform_item in transform_list_cfg:
        if not isinstance(transform_item, (dict, DictConfig)):
            raise ValueError("Each transform item must be a dictionary.")

        name = transform_item.get("name")
        params = transform_item.get("params", {})

        if name == "Resize":
            resize_found = True
            # Permitir tanto 'size' como 'height'/'width' para compatibilidad
            if "size" not in params and (
                    "height" not in params or "width" not in params):
                raise ValueError(
                    "Resize transform must have either 'size' or both "
                    "'height' and 'width' parameters."
                )

            if "size" in params:
                size = params["size"]
                if not (isinstance(size, (list, ListConfig)) and len(size) == 2
                        ):
                    raise ValueError(
                        "Resize 'size' must be a list/tuple of length 2."
                    )
            # Verificar height/width si están presentes
            if "height" in params and "width" in params:
                if not (isinstance(params["height"], (int, float)) and
                        isinstance(params["width"], (int, float))):
                    raise ValueError(
                        "Resize 'height' and 'width' must be numeric values."
                    )

        elif name == "Normalize":
            normalize_found = True
            if "mean" not in params or "std" not in params:
                raise ValueError("Missing 'mean' or 'std' in Normalize params."
                                 )
            mean, std = params["mean"], params["std"]
            if not (isinstance(mean, (list, ListConfig)) and len(mean) == 3):
                raise ValueError(
                    "Normalize 'mean' must be list/tuple of 3 values."
                )
            if not (isinstance(std, (list, ListConfig)) and len(std) == 3):
                raise ValueError(
                    "Normalize 'std' must be list/tuple of 3 values."
                )

    if not resize_found:
        raise ValueError("Missing 'Resize' transform in the list.")
    if not normalize_found:
        raise ValueError("Missing 'Normalize' transform in the list.")
