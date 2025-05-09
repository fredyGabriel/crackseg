# Training Workflow Guide

This document describes the basic workflow for configuring and training a crack segmentation model in this project. It includes available architecture and component options, and a minimal configuration example.

---

## 1. Environment Setup

- Install the Conda environment:
  ```bash
  conda env create -f environment.yml
  conda activate torch
  ```
- Configure environment variables if needed:
  - Copy `.env.example` to `.env` and edit the required values.

---

## 2. Data Configuration

- Edit `configs/data/default.yaml` to set:
  - Data root path (`data_root`)
  - Split proportions (`train_split`, `val_split`, `test_split`)
  - Image size (`image_size`)
  - Batch size and number of workers (`batch_size`, `num_workers`)

---

## 3. Select Model Architecture and Components

- **Available architectures** (`configs/model/architectures/`):
  - `unet_cnn.yaml` — Classic U-Net with CNN
  - `unet_aspp.yaml` — U-Net with ASPP
  - `unet_swin.yaml` — U-Net with Swin Transformer
  - `unet_swin_base.yaml` — Base variant of U-Net+Swin
  - `unet_swin_transfer.yaml` — U-Net+Swin with transfer learning
  - `swinv2_hybrid.yaml` — SwinV2 hybrid
  - `cnn_convlstm_unet.yaml` — U-Net with CNN and ConvLSTM
  - `unet_mock.yaml` — Mock for testing

- **Encoders** (`configs/model/encoder/`):
  - `default_encoder.yaml`
  - `swin_transformer_encoder.yaml`
  - `mock_encoder.yaml`

- **Decoders** (`configs/model/decoder/`):
  - `default_decoder.yaml`
  - `mock_decoder.yaml`

- **Bottlenecks** (`configs/model/bottleneck/`):
  - `default_bottleneck.yaml`
  - `aspp_bottleneck.yaml`
  - `convlstm_bottleneck.yaml`
  - `mock_bottleneck.yaml`

> You can combine these components by editing the architecture files or referencing them in the Hydra configuration.

---

## 4. Training Configuration

- Edit `configs/training/trainer.yaml` to set:
  - Number of epochs (`epochs`)
  - Device (`device`: "auto", "cpu", "cuda")
  - Optimizer and learning rate
  - Checkpoint and early stopping strategy

- **Available loss functions** (`configs/training/loss/`):
  - `bce.yaml` — Binary Cross Entropy
  - `dice.yaml` — Dice Loss
  - `focal.yaml` — Focal Loss
  - `bce_dice.yaml` — BCE + Dice combined
  - `combined.yaml` — Custom combination (e.g., Focal + Dice)

- **Available schedulers** (`configs/training/lr_scheduler/`):
  - `step_lr.yaml`
  - `cosine.yaml`
  - `reduce_on_plateau.yaml`

- **Available metrics** (`configs/training/metric/`):
  - `iou.yaml` — Intersection over Union
  - `f1.yaml` — F1 Score
  - `precision.yaml`
  - `recall.yaml`

---

## 5. Basic Training Example

1. **Select architecture and components**  
   For example, to train a classic U-Net:
   - In your Hydra configuration (can be in `configs/config.yaml` or via the Hydra CLI), select:
     ```yaml
     model:
       architecture: unet_cnn
       encoder: default_encoder
       decoder: default_decoder
       bottleneck: default_bottleneck
     ```

2. **Select loss function and scheduler**  
   For example:
   ```yaml
   training:
     loss: dice
     lr_scheduler: step_lr
     metric: [iou, f1]
   ```

3. **Start training**  
   ```bash
   python run.py
   ```
   - Results, logs, and checkpoints will be saved in `outputs/experiments/<timestamp>/`.

---

## 6. Model Evaluation

- Once training is complete, run:
  ```bash
  python src/evaluate.py
  ```
  - This will evaluate the model on the test set and save metrics and predictions.

---

## Minimal Hydra Configuration Example (`config.yaml`)

```yaml
defaults:
  - model/architecture: unet_cnn
  - model/encoder: default_encoder
  - model/decoder: default_decoder
  - model/bottleneck: default_bottleneck
  - training/loss: dice
  - training/lr_scheduler: step_lr
  - training/metric: [iou, f1]
  - data: default
  - data/dataloader: default

# You can override any parameter here
```

---

## Final Notes

- You can easily create variants by changing values in the YAML files or using the Hydra command line.
- See the README files in each `configs/` subfolder for more details and examples.
- The system is modular: you can combine any architecture, encoder, decoder, bottleneck, loss function, scheduler, and metric. 