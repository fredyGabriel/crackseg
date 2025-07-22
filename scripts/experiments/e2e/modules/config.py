"""Configuration module for end-to-end pipeline testing."""

from omegaconf import OmegaConf


def create_mini_config():
    """Create a minimal config for testing."""
    config = {
        "data": {
            "data_root": "data",
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
            "image_size": [256, 256],
            "batch_size": 4,
            "num_workers": 2,
            "seed": 42,
            "in_memory_cache": False,
            "transforms": {
                "train": [
                    {
                        "name": "Resize",
                        "params": {"height": 256, "width": 256},
                    },
                    {
                        "name": "Normalize",
                        "params": {
                            "mean": [0.5, 0.5, 0.5],
                            "std": [0.5, 0.5, 0.5],
                        },
                    },
                ],
                "val": [
                    {
                        "name": "Resize",
                        "params": {"height": 256, "width": 256},
                    },
                    {
                        "name": "Normalize",
                        "params": {
                            "mean": [0.5, 0.5, 0.5],
                            "std": [0.5, 0.5, 0.5],
                        },
                    },
                ],
                "test": [
                    {
                        "name": "Resize",
                        "params": {"height": 256, "width": 256},
                    },
                    {
                        "name": "Normalize",
                        "params": {
                            "mean": [0.5, 0.5, 0.5],
                            "std": [0.5, 0.5, 0.5],
                        },
                    },
                ],
            },
        },
        "model": {
            "_target_": "src.model.unet.BaseUNet",
            "encoder": {
                "type": "CNNEncoder",
                "in_channels": 3,
                "init_features": 16,
                "depth": 3,
            },
            "bottleneck": {
                "type": "CNNBottleneckBlock",
                "in_channels": 64,
                "out_channels": 128,
            },
            "decoder": {
                "type": "CNNDecoder",
                "in_channels": 128,
                "skip_channels_list": [16, 32, 64],
                "out_channels": 1,
                "depth": 3,
            },
            "final_activation": {"_target_": "torch.nn.Sigmoid"},
        },
        "training": {
            "epochs": 2,
            "optimizer": {"type": "torch.optim.Adam", "lr": 0.001},
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 1,
                "gamma": 0.5,
            },
            "loss": {
                "_target_": "src.training.losses.BCEDiceLoss",
                "bce_weight": 0.5,
                "dice_weight": 0.5,
            },
            "checkpoints": {
                "save_interval_epochs": 1,
                "save_best": {
                    "enabled": True,
                    "monitor_metric": "val_iou",
                    "monitor_mode": "max",
                },
                "save_last": True,
            },
            "amp_enabled": False,
        },
        "evaluation": {
            "metrics": {
                "dice": {"_target_": "src.training.metrics.F1Score"},
                "iou": {"_target_": "src.training.metrics.IoUScore"},
            }
        },
        "random_seed": 42,
        "require_cuda": False,
        "log_level": "INFO",
        "log_to_file": True,
    }
    return OmegaConf.create(config)
