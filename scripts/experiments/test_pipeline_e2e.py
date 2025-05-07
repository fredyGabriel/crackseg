#!/usr/bin/env python
"""
End-to-end test script to verify the full training pipeline.

This script performs a complete pipeline test:
1. Loads a small synthetic dataset
2. Configures a reduced UNet model
3. Trains for a few epochs
4. Saves checkpoints at intervals
5. Loads the model from checkpoint
6. Evaluates on the test set
7. Generates a results report
"""

import os
import sys
import time
import logging
from pathlib import Path
import torch
import yaml
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset, DataLoader, random_split

# Add project root to path for module imports
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, project_root)

from src.utils import set_random_seeds, save_checkpoint, load_checkpoint  # noqa E402
from src.utils.logging import get_logger, ExperimentLogger  # noqa E402
from src.data.factory import create_dataloaders_from_config  # noqa E402
from src.model.factory import create_unet  # noqa E402
from src.utils.factory import get_optimizer, get_loss_fn  # noqa E402
from src.training.factory import create_lr_scheduler  # noqa E402
from src.training.metrics import IoUScore, F1Score  # noqa E402

# Logging configuration
log = get_logger("e2e_test")
log_level = logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def create_mini_config():
    """Create a minimal config for testing."""
    config = {
        'data': {
            'data_root': 'data',
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'image_size': [256, 256],
            'batch_size': 4,
            'num_workers': 2,
            'seed': 42,
            'in_memory_cache': False,
            'transforms': {
                'train': [
                    {'name': 'Resize',
                     'params': {'height': 256, 'width': 256}},
                    {'name': 'Normalize',
                     'params': {'mean': [0.5, 0.5, 0.5],
                                'std': [0.5, 0.5, 0.5]}}
                ],
                'val': [
                    {'name': 'Resize',
                     'params': {'height': 256, 'width': 256}},
                    {'name': 'Normalize',
                     'params': {'mean': [0.5, 0.5, 0.5],
                                'std': [0.5, 0.5, 0.5]}}
                ],
                'test': [
                    {'name': 'Resize',
                     'params': {'height': 256, 'width': 256}},
                    {'name': 'Normalize',
                     'params': {'mean': [0.5, 0.5, 0.5],
                                'std': [0.5, 0.5, 0.5]}}
                ]
            }
        },
        'model': {
            '_target_': 'src.model.unet.BaseUNet',
            'encoder': {
                'type': 'CNNEncoder',
                'in_channels': 3,
                'init_features': 16,  # Fixed: base_channels -> init_features
                'depth': 3            # Reduced for test
            },
            'bottleneck': {
                'type': 'CNNBottleneckBlock',  # Fixed: CNNBottleneckBlock
                'in_channels': 64,    # Adapted to encoder
                'out_channels': 128   # Reduced for test
            },
            'decoder': {
                'type': 'CNNDecoder',
                'in_channels': 128,   # Adapted to bottleneck
                'skip_channels_list': [16, 32, 64],
                'out_channels': 1,
                'depth': 3            # Must match encoder
            },
            'final_activation': {
                '_target_': 'torch.nn.Sigmoid'
            }
        },
        'training': {
            'epochs': 2,              # Few epochs for quick test
            'optimizer': {
                'type': 'torch.optim.Adam',
                'lr': 0.001
            },
            'scheduler': {
                '_target_': 'torch.optim.lr_scheduler.StepLR',
                'step_size': 1,
                'gamma': 0.5
            },
            'loss': {
                '_target_': 'src.training.losses.BCEDiceLoss',
                'bce_weight': 0.5,
                'dice_weight': 0.5
            },
            'checkpoints': {
                'save_interval_epochs': 1,
                'save_best': {
                    'enabled': True,
                    'monitor_metric': 'val_iou',
                    'monitor_mode': 'max'
                },
                'save_last': True
            },
            'amp_enabled': False      # Disabled for simplicity
        },
        'evaluation': {
            'metrics': {
                'dice': {'_target_': 'src.training.metrics.F1Score'},
                'iou': {'_target_': 'src.training.metrics.IoUScore'}
            }
        },
        'random_seed': 42,
        'require_cuda': False,
        'log_level': 'INFO',
        'log_to_file': True
    }
    return OmegaConf.create(config)


def save_config(config, path):
    """Save config to YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(OmegaConf.to_container(config, resolve=True), f,
                  default_flow_style=False)
    log.info(f"Config saved to {path}")


def create_experiment_dir():
    """Create experiment directory."""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(project_root, "outputs", "e2e_test", timestamp)
    os.makedirs(exp_dir, exist_ok=True)

    # Create subdirectories
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "visualizations"), exist_ok=True)

    log.info(f"Experiment directory created: {exp_dir}")
    return exp_dir


def visualize_results(images, masks, predictions, output_path):
    """Save visualizations of predictions vs ground truth."""
    num_samples = min(4, len(images))
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))

    for i in range(num_samples):
        # Convert tensors to numpy and prepare for visualization
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)  # Denormalize

        mask = masks[i].cpu().squeeze().numpy()
        pred = predictions[i].cpu().squeeze().numpy()

        # Plot image, mask, and prediction
        axs[i, 0].imshow(img)
        axs[i, 0].set_title("Original image")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(mask, cmap='gray')
        axs[i, 1].set_title("Ground truth mask")
        axs[i, 1].axis('off')

        axs[i, 2].imshow(pred, cmap='gray')
        axs[i, 2].set_title("Prediction")
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    log.info(f"Visualization saved to {output_path}")


def create_synthetic_dataset():
    """Create a synthetic dataset for end-to-end tests."""
    # Create synthetic data (random 16x16 images)
    num_samples = 20
    image_size = 16

    # Create random tensors for images and masks
    images = torch.rand(num_samples, 3, image_size, image_size)
    masks = torch.randint(
        0, 2, (num_samples, 1, image_size, image_size)
    ).float()

    # Create dataset and split into train/val/test
    dataset = TensorDataset(images, masks)
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader


def get_metrics_from_cfg(metrics_cfg):
    """Get metrics from config."""
    metrics = {}
    for name, metric_config in metrics_cfg.items():
        if metric_config.get("_target_", "").endswith("F1Score"):
            metrics[name] = F1Score()
        elif metric_config.get("_target_", "").endswith("IoUScore"):
            metrics[name] = IoUScore()
    # If no metrics defined, use defaults
    if not metrics:
        metrics = {
            'dice': F1Score(),
            'iou': IoUScore()
        }
    return metrics


def run_e2e_test():
    """Run end-to-end test of the full pipeline."""
    # 1. Create config
    cfg = OmegaConf.create(create_mini_config())

    # 2. Create experiment directory
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(
        os.path.abspath(os.getcwd()),
        "outputs",
        "e2e_test",
        timestamp
    )
    os.makedirs(exp_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(exp_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    log.info(f"Config saved to {config_path}")

    # Directories for results
    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    metrics_dir = os.path.join(exp_dir, "metrics")
    vis_dir = os.path.join(exp_dir, "visualizations")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Configure logger
    experiment_logger = ExperimentLogger(
        log_dir=exp_dir,
        experiment_name="e2e_test",
        config=cfg,
        log_level='INFO',
        log_to_file=True
    )

    # 3. Set reproducibility
    set_random_seeds(cfg.random_seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and
        cfg.get('require_cuda', False) else "cpu"
    )
    log.info(f"Using device: {device}")

    # 4. Load data (using real dataset instead of synthetic)
    log.info("Loading real data from data/ directory...")
    dataloaders = create_dataloaders_from_config(
        data_config=cfg.data,
        transform_config=cfg.data.get("transforms", {}),
        dataloader_config=cfg.data
    )
    train_loader = dataloaders['train']['dataloader']
    val_loader = dataloaders['val']['dataloader']
    test_loader = dataloaders['test']['dataloader']
    log.info(
        f"Data loaded: {len(train_loader.dataset)} train, "
        f"{len(val_loader.dataset)} val, "
        f"{len(test_loader.dataset)} test"
    )

    # 5. Create model
    log.info("Creating model...")
    model = create_unet(cfg.model)
    model = model.to(device)

    # 6. Training setup
    log.info("Setting up training...")
    loss_fn = get_loss_fn(cfg.training.loss)
    optimizer = get_optimizer(model.parameters(), cfg.training.optimizer)
    lr_scheduler = create_lr_scheduler(optimizer, cfg.training.scheduler)

    # 7. Get metrics
    metrics = get_metrics_from_cfg(cfg.evaluation.metrics)

    # 8. Setup AMP
    use_amp = cfg.training.get("amp_enabled", False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    log.info(f"AMP Enabled: {scaler is not None}")
    log.info("Training setup complete.")

    try:
        # 9. Training
        log.info("Starting training...")
        best_metric = 0

        for epoch in range(cfg.training.epochs):
            log.info(f"Epoch {epoch+1}/{cfg.training.epochs}")

            # 9.1 Training
            model.train()
            train_loss = 0.0
            train_metrics = {k: 0.0 for k in metrics.keys()}

            for batch_idx, batch in enumerate(train_loader):
                if isinstance(batch, dict):
                    inputs, targets = batch['image'].to(device), batch[
                        'mask'].to(device)
                else:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)

                # Ensure inputs tensor has the correct shape (B, C, H, W)
                if inputs.shape[-1] == 3:  # If channels are last (B, H, W, C)
                    # Change to (B, C, H, W)
                    inputs = inputs.permute(0, 3, 1, 2)

                # Ensure targets tensor has a channel dimension (B, C, H, W)
                # If (B, H, W) without channel dim
                if len(targets.shape) == 3:
                    # Add channel dim -> (B, 1, H, W)
                    targets = targets.unsqueeze(1)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Update metrics
                train_loss += loss.item()
                with torch.no_grad():
                    for k, metric_fn in metrics.items():
                        train_metrics[k] += metric_fn(outputs, targets).item()

                # Show progress
                if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(
                        train_loader):
                    log.info(
                        f"Train Batch: {batch_idx+1}/{len(train_loader)}, "
                        f"Loss: {loss.item():.4f}"
                    )

            # Average metrics
            train_loss /= len(train_loader)
            for k in train_metrics:
                train_metrics[k] /= len(train_loader)

            log.info(f"Train Loss: {train_loss:.4f}, Metrics: {train_metrics}")
            experiment_logger.log_scalar("train/loss", train_loss, epoch)
            for k, v in train_metrics.items():
                experiment_logger.log_scalar(f"train/{k}", v, epoch)

            # 9.2 Validation
            model.eval()
            val_loss = 0.0
            val_metrics = {k: 0.0 for k in metrics.keys()}

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if isinstance(batch, dict):
                        inputs, targets = batch['image'].to(
                            device), batch['mask'].to(device)
                    else:
                        inputs, targets = batch
                        inputs, targets = inputs.to(device), targets.to(device)

                    # Ensure inputs tensor has the correct shape (B, C, H, W)
                    if inputs.shape[-1] == 3:  # If channels are last dimension
                        inputs = inputs.permute(0, 3, 1, 2)  # To (B, C, H, W)

                    # Ensure targets tensor has a channel dimension
                    if len(targets.shape) == 3:
                        targets = targets.unsqueeze(1)  # Add channel dim

                    # Forward pass
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)

                    # Update metrics
                    val_loss += loss.item()
                    for k, metric_fn in metrics.items():
                        val_metrics[k] += metric_fn(outputs, targets).item()

            # Average metrics
            val_loss /= len(val_loader)
            for k in val_metrics:
                val_metrics[k] /= len(val_loader)

            log.info(
                f"Validation Loss: {val_loss:.4f}, "
                f"Metrics: {val_metrics}"
            )
            experiment_logger.log_scalar("val/loss", val_loss, epoch)
            for k, v in val_metrics.items():
                experiment_logger.log_scalar(f"val/{k}", v, epoch)

            # 9.3 Update scheduler
            if lr_scheduler:
                if isinstance(lr_scheduler,
                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(val_loss)
                else:
                    lr_scheduler.step()

            # 9.4 Save checkpoint
            is_best = val_metrics['iou'] > best_metric
            if is_best:
                best_metric = val_metrics['iou']
                log.info(f"New best IoU: {best_metric:.4f}")

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                checkpoint_dir=checkpoints_dir,
                filename=f"checkpoint_epoch_{epoch+1}.pth.tar",
                additional_data={
                    'scheduler_state_dict': (
                        lr_scheduler.state_dict() if lr_scheduler else None
                    ),
                    'best_metric_value': best_metric,
                    'config': OmegaConf.to_container(cfg, resolve=True)
                }
            )
            log.info(f"Checkpoint saved for epoch {epoch+1}")

        log.info("Training finished")

        # 10. Load best model for evaluation
        log.info("Loading best model for evaluation...")
        best_model_path = os.path.join(checkpoints_dir, "model_best.pth.tar")

        if os.path.exists(best_model_path):
            checkpoint_data = load_checkpoint(
                model=model,
                checkpoint_path=best_model_path,
                device=device
            )
            log.info(f"Model loaded from: {best_model_path}")
            log.info(
                f"Model from epoch: {checkpoint_data.get('epoch', 'unknown')}"
            )
        else:
            log.info(f"Checkpoint not found at {best_model_path}")

        # 11. Test set evaluation
        log.info("Evaluating on test set...")
        model.eval()
        test_loss = 0.0
        test_metrics = {k: 0.0 for k in metrics.keys()}

        # Save some images, masks, and predictions for visualization
        sample_images = []
        sample_masks = []
        sample_preds = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if isinstance(batch, dict):
                    inputs, targets = batch['image'].to(
                        device), batch['mask'].to(device)
                else:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)

                # Ensure inputs tensor has the correct shape (B, C, H, W)
                if inputs.shape[-1] == 3:  # If channels are last dimension
                    inputs = inputs.permute(0, 3, 1, 2)  # To (B, C, H, W)

                # Ensure targets tensor has a channel dimension
                if len(targets.shape) == 3:
                    targets = targets.unsqueeze(1)  # Add channel dim

                # Forward pass
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                # Update metrics
                test_loss += loss.item()
                for k, metric_fn in metrics.items():
                    test_metrics[k] += metric_fn(outputs, targets).item()

                # Save samples for visualization
                if batch_idx == 0:
                    sample_images.extend(inputs.cpu())
                    sample_masks.extend(targets.cpu())
                    sample_preds.extend(outputs.cpu())

        # Average metrics
        test_loss /= len(test_loader)
        for k in test_metrics:
            test_metrics[k] /= len(test_loader)

        log.info(
            f"Test Loss: {test_loss:.4f}, "
            f"Metrics: {test_metrics}"
        )
        experiment_logger.log_scalar("test/loss", test_loss, 0)
        for k, v in test_metrics.items():
            experiment_logger.log_scalar(f"test/{k}", v, 0)

        # 12. Visualize results
        if sample_images:
            log.info("Generating visualizations...")
            visualize_results(
                sample_images[:4],  # Limit to 4 samples
                sample_masks[:4],
                sample_preds[:4],
                os.path.join(vis_dir, "predictions.png")
            )

        # 13. Save final metrics
        final_results = {
            'train': {
                'loss': train_loss,
                'metrics': train_metrics
            },
            'val': {
                'loss': val_loss,
                'metrics': val_metrics
            },
            'test': {
                'loss': test_loss,
                'metrics': test_metrics
            },
            'epochs': cfg.training.epochs,
            'best_metric': best_metric,
            'checkpoint_path': best_model_path
        }

        results_path = os.path.join(metrics_dir, "final_results.yaml")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            yaml.dump(final_results, f, default_flow_style=False)

        log.info(f"Checkpoint not found at {best_model_path}")
        log.info(
            f"Sample images: {sample_images[:4]}"
        )
        log.info(
            f"Results: {final_results}"
        )
        log.info(f"Final results saved to {results_path}")
        log.info("End-to-end test completed successfully")

    except Exception as e:
        log.exception(f"Error during end-to-end test: {str(e)}")
        raise
    finally:
        # Close resources
        experiment_logger.close()
        log.info("Resources released")

    return exp_dir, final_results


if __name__ == "__main__":
    experiment_dir, results = run_e2e_test()
    print(f"\n{'='*80}")
    print(" End-to-End Test Completed")
    print(f" Experiment directory: {experiment_dir}")
    print(f" Test metrics: {results['test']['metrics']}")
    print(f"{'='*80}")
