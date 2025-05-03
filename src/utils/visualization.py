import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def visualize_predictions(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    outputs: torch.Tensor,
    output_dir: str,
    num_samples: int = 5
) -> None:
    """
    Create and save visualizations of model predictions.

    Args:
        inputs: Input images
        targets: Ground truth masks
        outputs: Model predictions
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    # Limit number of samples
    num_samples = min(num_samples, len(inputs))

    # Apply threshold to outputs (assuming sigmoid activation)
    binary_outputs = (outputs > 0.5).float()

    # Create directory if it doesn't exist
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Configure plotting
    plt.figure(figsize=(15, 5 * num_samples))

    for i in range(num_samples):
        # Get current sample
        img = inputs[i].permute(1, 2, 0).numpy()
        mask = targets[i].squeeze().numpy()
        pred = outputs[i].squeeze().numpy()
        binary_pred = binary_outputs[i].squeeze().numpy()

        # Denormalize image if it's normalized to [-1, 1]
        if img.min() < 0:
            img = (img + 1) / 2
        # Clip to [0, 1] range
        img = np.clip(img, 0, 1)

        # Plot original image, ground truth, prediction, and binary prediction
        plt.subplot(num_samples, 4, i*4 + 1)
        plt.imshow(img)
        plt.title(f"Image {i+1}")
        plt.axis('off')

        plt.subplot(num_samples, 4, i*4 + 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(num_samples, 4, i*4 + 3)
        plt.imshow(pred, cmap='gray')
        plt.title("Prediction (Raw)")
        plt.axis('off')

        plt.subplot(num_samples, 4, i*4 + 4)
        plt.imshow(binary_pred, cmap='gray')
        plt.title("Prediction (Binary)")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "prediction_samples.png"), dpi=200)
    plt.close()

    # Save individual samples for detailed inspection
    for i in range(num_samples):
        img = inputs[i].permute(1, 2, 0).numpy()
        mask = targets[i].squeeze().numpy()
        pred = outputs[i].squeeze().numpy()

        # Denormalize image
        if img.min() < 0:
            img = (img + 1) / 2
        img = np.clip(img, 0, 1)

        # Create overlay visualization (prediction contour on image)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)

        # Create contour of the prediction
        masked_pred = np.ma.masked_where(pred < 0.5, pred)
        ax.imshow(masked_pred, cmap='jet', alpha=0.4)

        plt.title(f"Sample {i+1}: Image with crack overlay")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"overlay_sample_{i+1}.png"),
                    dpi=200)
        plt.close()
