# Project Directory Structure (excluding .gitignore)

```txt
└── crackseg/
    ├── ${TASK_MASTER_PROJECT_ROOT}/
    ├── artifacts/
    ├── configs/
    │   ├── __pycache__/
    │   ├── archive/
    │   │   ├── config.yaml.backup
    │   │   └── experiment_quick_test.yaml.backup
    │   ├── data/
    │   │   ├── dataloader/
    │   │   │   └── default.yaml
    │   │   ├── transform/
    │   │   │   └── augmentations.yaml
    │   │   ├── default.yaml
    │   │   └── README.md
    │   ├── evaluation/
    │   │   └── default.yaml
    │   ├── experiments/
    │   │   ├── swinv2_hybrid/
    │   │   │   └── swinv2_hybrid_experiment.yaml
    │   │   ├── tutorial_02/
    │   │   │   ├── focal_loss_experiment.yaml
    │   │   │   ├── high_lr_experiment.yaml
    │   │   │   ├── low_lr_experiment.yaml
    │   │   │   ├── README.md
    │   │   │   └── swin_unet_experiment.yaml
    │   │   └── tutorial_03/
    │   │       └── smooth_l1_experiment.yaml
    │   ├── linting/
    │   │   └── config.yaml
    │   ├── model/
    │   │   ├── architecture/
    │   │   ├── architectures/
    │   │   │   ├── cnn_convlstm_unet.yaml
    │   │   │   ├── README.md
    │   │   │   ├── swinv2_hybrid.yaml
    │   │   │   ├── unet_aspp.yaml
    │   │   │   ├── unet_cnn.yaml
    │   │   │   ├── unet_swin.yaml
    │   │   │   ├── unet_swin_base.yaml
    │   │   │   └── unet_swin_transfer.yaml
    │   │   ├── bottleneck/
    │   │   │   ├── __init__.py
    │   │   │   ├── aspp_bottleneck.yaml
    │   │   │   ├── convlstm_bottleneck.yaml
    │   │   │   ├── default_bottleneck.yaml
    │   │   │   ├── mock_bottleneck.yaml
    │   │   │   └── README.md
    │   │   ├── decoder/
    │   │   │   ├── default_decoder.yaml
    │   │   │   └── mock_decoder.yaml
    │   │   ├── encoder/
    │   │   │   ├── default_encoder.yaml
    │   │   │   ├── mock_encoder.yaml
    │   │   │   └── swin_transformer_encoder.yaml
    │   │   ├── default.yaml
    │   │   └── README.md
    │   ├── testing/
    │   │   └── performance_thresholds.yaml
    │   ├── training/
    │   │   ├── logging/
    │   │   │   ├── checkpoints.yaml
    │   │   │   └── logging_base.yaml
    │   │   ├── loss/
    │   │   │   ├── bce.yaml
    │   │   │   ├── bce_dice.yaml
    │   │   │   ├── combined.yaml
    │   │   │   ├── dice.yaml
    │   │   │   ├── focal.yaml
    │   │   │   ├── focal_dice.yaml
    │   │   │   └── smooth_l1.yaml
    │   │   ├── lr_scheduler/
    │   │   │   ├── cosine.yaml
    │   │   │   ├── reduce_on_plateau.yaml
    │   │   │   └── step_lr.yaml
    │   │   ├── metric/
    │   │   │   ├── f1.yaml
    │   │   │   ├── iou.yaml
    │   │   │   ├── precision.yaml
    │   │   │   └── recall.yaml
    │   │   ├── optimizer/
    │   │   ├── default.yaml
    │   │   ├── README.md
    │   │   └── trainer.yaml
    │   ├── __init__.py
    │   ├── base.yaml
    │   ├── basic_verification.yaml
    │   ├── README.md
    │   └── simple_test.yaml
    ├── data/
    │   ├── test/
    │   │   ├── images/
    │   │   │   ├── 101.jpg
    │   │   │   ├── 102.jpg
    │   │   │   ├── 104.jpg
    │   │   │   ├── 109.jpg
    │   │   │   ├── 110.jpg
    │   │   │   ├── 114.jpg
    │   │   │   ├── 123.jpg
    │   │   │   ├── 124.jpg
    │   │   │   ├── 125.jpg
    │   │   │   ├── 127.jpg
    │   │   │   ├── 30.jpg
    │   │   │   ├── 44.jpg
    │   │   │   ├── 45.jpg
    │   │   │   ├── 5.jpg
    │   │   │   ├── 6.jpg
    │   │   │   ├── 67.jpg
    │   │   │   ├── 85.jpg
    │   │   │   ├── 88.jpg
    │   │   │   ├── 93.jpg
    │   │   │   └── 99.jpg
    │   │   └── masks/
    │   │       ├── 101.png
    │   │       ├── 102.png
    │   │       ├── 104.png
    │   │       ├── 109.png
    │   │       ├── 110.png
    │   │       ├── 114.png
    │   │       ├── 123.png
    │   │       ├── 124.png
    │   │       ├── 125.png
    │   │       ├── 127.png
    │   │       ├── 30.png
    │   │       ├── 44.png
    │   │       ├── 45.png
    │   │       ├── 5.png
    │   │       ├── 6.png
    │   │       ├── 67.png
    │   │       ├── 85.png
    │   │       ├── 88.png
    │   │       ├── 93.png
    │   │       └── 99.png
    │   ├── train/
    │   │   ├── images/
    │   │   │   ├── 10.jpg
    │   │   │   ├── 100.jpg
    │   │   │   ├── 103.jpg
    │   │   │   ├── 105.jpg
    │   │   │   ├── 106.jpg
    │   │   │   ├── 107.jpg
    │   │   │   ├── 108.jpg
    │   │   │   ├── 11.jpg
    │   │   │   ├── 111.jpg
    │   │   │   ├── 112.jpg
    │   │   │   ├── 113.jpg
    │   │   │   ├── 115.jpg
    │   │   │   ├── 116.jpg
    │   │   │   ├── 118.jpg
    │   │   │   ├── 119.jpg
    │   │   │   ├── 12.jpg
    │   │   │   ├── 120.jpg
    │   │   │   ├── 122.jpg
    │   │   │   ├── 126.jpg
    │   │   │   ├── 128.jpg
    │   │   │   ├── 129.jpg
    │   │   │   ├── 13.jpg
    │   │   │   ├── 14.jpg
    │   │   │   ├── 15.jpg
    │   │   │   ├── 16.jpg
    │   │   │   ├── 17.jpg
    │   │   │   ├── 18.jpg
    │   │   │   ├── 19.jpg
    │   │   │   ├── 2.jpg
    │   │   │   ├── 21.jpg
    │   │   │   ├── 22.jpg
    │   │   │   ├── 23.jpg
    │   │   │   ├── 24.jpg
    │   │   │   ├── 25.jpg
    │   │   │   ├── 26.jpg
    │   │   │   ├── 27.jpg
    │   │   │   ├── 28.jpg
    │   │   │   ├── 29.jpg
    │   │   │   ├── 31.jpg
    │   │   │   ├── 32.jpg
    │   │   │   ├── 34.jpg
    │   │   │   ├── 35.jpg
    │   │   │   ├── 36.jpg
    │   │   │   ├── 37.jpg
    │   │   │   ├── 38.jpg
    │   │   │   ├── 39.jpg
    │   │   │   ├── 40.jpg
    │   │   │   ├── 41.jpg
    │   │   │   ├── 42.jpg
    │   │   │   ├── 43.jpg
    │   │   │   ├── 47.jpg
    │   │   │   ├── 48.jpg
    │   │   │   ├── 49.jpg
    │   │   │   ├── 50.jpg
    │   │   │   ├── 52.jpg
    │   │   │   ├── 53.jpg
    │   │   │   ├── 54.jpg
    │   │   │   ├── 55.jpg
    │   │   │   ├── 56.jpg
    │   │   │   ├── 58.jpg
    │   │   │   ├── 59.jpg
    │   │   │   ├── 60.jpg
    │   │   │   ├── 61.jpg
    │   │   │   ├── 63.jpg
    │   │   │   ├── 65.jpg
    │   │   │   ├── 66.jpg
    │   │   │   ├── 68.jpg
    │   │   │   ├── 69.jpg
    │   │   │   ├── 7.jpg
    │   │   │   ├── 70.jpg
    │   │   │   ├── 71.jpg
    │   │   │   ├── 72.jpg
    │   │   │   ├── 73.jpg
    │   │   │   ├── 75.jpg
    │   │   │   ├── 76.jpg
    │   │   │   ├── 77.jpg
    │   │   │   ├── 78.jpg
    │   │   │   ├── 79.jpg
    │   │   │   ├── 8.jpg
    │   │   │   ├── 80.jpg
    │   │   │   ├── 81.jpg
    │   │   │   ├── 82.jpg
    │   │   │   ├── 83.jpg
    │   │   │   ├── 9.jpg
    │   │   │   ├── 90.jpg
    │   │   │   ├── 91.jpg
    │   │   │   ├── 92.jpg
    │   │   │   ├── 94.jpg
    │   │   │   ├── 96.jpg
    │   │   │   ├── 97.jpg
    │   │   │   └── 98.jpg
    │   │   └── masks/
    │   │       ├── 10.png
    │   │       ├── 100.png
    │   │       ├── 103.png
    │   │       ├── 105.png
    │   │       ├── 106.png
    │   │       ├── 107.png
    │   │       ├── 108.png
    │   │       ├── 11.png
    │   │       ├── 111.png
    │   │       ├── 112.png
    │   │       ├── 113.png
    │   │       ├── 115.png
    │   │       ├── 116.png
    │   │       ├── 118.png
    │   │       ├── 119.png
    │   │       ├── 12.png
    │   │       ├── 120.png
    │   │       ├── 122.png
    │   │       ├── 126.png
    │   │       ├── 128.png
    │   │       ├── 129.png
    │   │       ├── 13.png
    │   │       ├── 14.png
    │   │       ├── 15.png
    │   │       ├── 16.png
    │   │       ├── 17.png
    │   │       ├── 18.png
    │   │       ├── 19.png
    │   │       ├── 2.png
    │   │       ├── 21.png
    │   │       ├── 22.png
    │   │       ├── 23.png
    │   │       ├── 24.png
    │   │       ├── 25.png
    │   │       ├── 26.png
    │   │       ├── 27.png
    │   │       ├── 28.png
    │   │       ├── 29.png
    │   │       ├── 31.png
    │   │       ├── 32.png
    │   │       ├── 34.png
    │   │       ├── 35.png
    │   │       ├── 36.png
    │   │       ├── 37.png
    │   │       ├── 38.png
    │   │       ├── 39.png
    │   │       ├── 40.png
    │   │       ├── 41.png
    │   │       ├── 42.png
    │   │       ├── 43.png
    │   │       ├── 47.png
    │   │       ├── 48.png
    │   │       ├── 49.png
    │   │       ├── 50.png
    │   │       ├── 52.png
    │   │       ├── 53.png
    │   │       ├── 54.png
    │   │       ├── 55.png
    │   │       ├── 56.png
    │   │       ├── 58.png
    │   │       ├── 59.png
    │   │       ├── 60.png
    │   │       ├── 61.png
    │   │       ├── 63.png
    │   │       ├── 65.png
    │   │       ├── 66.png
    │   │       ├── 68.png
    │   │       ├── 69.png
    │   │       ├── 7.png
    │   │       ├── 70.png
    │   │       ├── 71.png
    │   │       ├── 72.png
    │   │       ├── 73.png
    │   │       ├── 75.png
    │   │       ├── 76.png
    │   │       ├── 77.png
    │   │       ├── 78.png
    │   │       ├── 79.png
    │   │       ├── 8.png
    │   │       ├── 80.png
    │   │       ├── 81.png
    │   │       ├── 82.png
    │   │       ├── 83.png
    │   │       ├── 9.png
    │   │       ├── 90.png
    │   │       ├── 91.png
    │   │       ├── 92.png
    │   │       ├── 94.png
    │   │       ├── 96.png
    │   │       ├── 97.png
    │   │       └── 98.png
    │   ├── unified/
    │   │   ├── images/
    │   │   │   ├── 1.jpg
    │   │   │   ├── 10.jpg
    │   │   │   ├── 100.jpg
    │   │   │   ├── 101.jpg
    │   │   │   ├── 102.jpg
    │   │   │   ├── 103.jpg
    │   │   │   ├── 104.jpg
    │   │   │   ├── 105.jpg
    │   │   │   ├── 106.jpg
    │   │   │   ├── 107.jpg
    │   │   │   ├── 108.jpg
    │   │   │   ├── 109.jpg
    │   │   │   ├── 11.jpg
    │   │   │   ├── 110.jpg
    │   │   │   ├── 111.jpg
    │   │   │   ├── 112.jpg
    │   │   │   ├── 113.jpg
    │   │   │   ├── 114.jpg
    │   │   │   ├── 115.jpg
    │   │   │   ├── 116.jpg
    │   │   │   ├── 117.jpg
    │   │   │   ├── 118.jpg
    │   │   │   ├── 119.jpg
    │   │   │   ├── 12.jpg
    │   │   │   ├── 120.jpg
    │   │   │   ├── 121.jpg
    │   │   │   ├── 122.jpg
    │   │   │   ├── 123.jpg
    │   │   │   ├── 124.jpg
    │   │   │   ├── 125.jpg
    │   │   │   ├── 126.jpg
    │   │   │   ├── 127.jpg
    │   │   │   ├── 128.jpg
    │   │   │   ├── 129.jpg
    │   │   │   ├── 13.jpg
    │   │   │   ├── 130.jpg
    │   │   │   ├── 14.jpg
    │   │   │   ├── 15.jpg
    │   │   │   ├── 16.jpg
    │   │   │   ├── 17.jpg
    │   │   │   ├── 18.jpg
    │   │   │   ├── 19.jpg
    │   │   │   ├── 2.jpg
    │   │   │   ├── 20.jpg
    │   │   │   ├── 21.jpg
    │   │   │   ├── 22.jpg
    │   │   │   ├── 23.jpg
    │   │   │   ├── 24.jpg
    │   │   │   ├── 25.jpg
    │   │   │   ├── 26.jpg
    │   │   │   ├── 27.jpg
    │   │   │   ├── 28.jpg
    │   │   │   ├── 29.jpg
    │   │   │   ├── 3.jpg
    │   │   │   ├── 30.jpg
    │   │   │   ├── 31.jpg
    │   │   │   ├── 32.jpg
    │   │   │   ├── 33.jpg
    │   │   │   ├── 34.jpg
    │   │   │   ├── 35.jpg
    │   │   │   ├── 36.jpg
    │   │   │   ├── 37.jpg
    │   │   │   ├── 38.jpg
    │   │   │   ├── 39.jpg
    │   │   │   ├── 4.jpg
    │   │   │   ├── 40.jpg
    │   │   │   ├── 41.jpg
    │   │   │   ├── 42.jpg
    │   │   │   ├── 43.jpg
    │   │   │   ├── 44.jpg
    │   │   │   ├── 45.jpg
    │   │   │   ├── 46.jpg
    │   │   │   ├── 47.jpg
    │   │   │   ├── 48.jpg
    │   │   │   ├── 49.jpg
    │   │   │   ├── 5.jpg
    │   │   │   ├── 50.jpg
    │   │   │   ├── 51.jpg
    │   │   │   ├── 52.jpg
    │   │   │   ├── 53.jpg
    │   │   │   ├── 54.jpg
    │   │   │   ├── 55.jpg
    │   │   │   ├── 56.jpg
    │   │   │   ├── 57.jpg
    │   │   │   ├── 58.jpg
    │   │   │   ├── 59.jpg
    │   │   │   ├── 6.jpg
    │   │   │   ├── 60.jpg
    │   │   │   ├── 61.jpg
    │   │   │   ├── 62.jpg
    │   │   │   ├── 63.jpg
    │   │   │   ├── 64.jpg
    │   │   │   ├── 65.jpg
    │   │   │   ├── 66.jpg
    │   │   │   ├── 67.jpg
    │   │   │   ├── 68.jpg
    │   │   │   ├── 69.jpg
    │   │   │   ├── 7.jpg
    │   │   │   ├── 70.jpg
    │   │   │   ├── 71.jpg
    │   │   │   ├── 72.jpg
    │   │   │   ├── 73.jpg
    │   │   │   ├── 74.jpg
    │   │   │   ├── 75.jpg
    │   │   │   ├── 76.jpg
    │   │   │   ├── 77.jpg
    │   │   │   ├── 78.jpg
    │   │   │   ├── 79.jpg
    │   │   │   ├── 8.jpg
    │   │   │   ├── 80.jpg
    │   │   │   ├── 81.jpg
    │   │   │   ├── 82.jpg
    │   │   │   ├── 83.jpg
    │   │   │   ├── 84.jpg
    │   │   │   ├── 85.jpg
    │   │   │   ├── 86.jpg
    │   │   │   ├── 87.jpg
    │   │   │   ├── 88.jpg
    │   │   │   ├── 89.jpg
    │   │   │   ├── 9.jpg
    │   │   │   ├── 90.jpg
    │   │   │   ├── 91.jpg
    │   │   │   ├── 92.jpg
    │   │   │   ├── 93.jpg
    │   │   │   ├── 94.jpg
    │   │   │   ├── 95.jpg
    │   │   │   ├── 96.jpg
    │   │   │   ├── 97.jpg
    │   │   │   ├── 98.jpg
    │   │   │   └── 99.jpg
    │   │   └── masks/
    │   │       ├── 1.png
    │   │       ├── 10.png
    │   │       ├── 100.png
    │   │       ├── 101.png
    │   │       ├── 102.png
    │   │       ├── 103.png
    │   │       ├── 104.png
    │   │       ├── 105.png
    │   │       ├── 106.png
    │   │       ├── 107.png
    │   │       ├── 108.png
    │   │       ├── 109.png
    │   │       ├── 11.png
    │   │       ├── 110.png
    │   │       ├── 111.png
    │   │       ├── 112.png
    │   │       ├── 113.png
    │   │       ├── 114.png
    │   │       ├── 115.png
    │   │       ├── 116.png
    │   │       ├── 117.png
    │   │       ├── 118.png
    │   │       ├── 119.png
    │   │       ├── 12.png
    │   │       ├── 120.png
    │   │       ├── 121.png
    │   │       ├── 122.png
    │   │       ├── 123.png
    │   │       ├── 124.png
    │   │       ├── 125.png
    │   │       ├── 126.png
    │   │       ├── 127.png
    │   │       ├── 128.png
    │   │       ├── 129.png
    │   │       ├── 13.png
    │   │       ├── 130.png
    │   │       ├── 14.png
    │   │       ├── 15.png
    │   │       ├── 16.png
    │   │       ├── 17.png
    │   │       ├── 18.png
    │   │       ├── 19.png
    │   │       ├── 2.png
    │   │       ├── 20.png
    │   │       ├── 21.png
    │   │       ├── 22.png
    │   │       ├── 23.png
    │   │       ├── 24.png
    │   │       ├── 25.png
    │   │       ├── 26.png
    │   │       ├── 27.png
    │   │       ├── 28.png
    │   │       ├── 29.png
    │   │       ├── 3.png
    │   │       ├── 30.png
    │   │       ├── 31.png
    │   │       ├── 32.png
    │   │       ├── 33.png
    │   │       ├── 34.png
    │   │       ├── 35.png
    │   │       ├── 36.png
    │   │       ├── 37.png
    │   │       ├── 38.png
    │   │       ├── 39.png
    │   │       ├── 4.png
    │   │       ├── 40.png
    │   │       ├── 41.png
    │   │       ├── 42.png
    │   │       ├── 43.png
    │   │       ├── 44.png
    │   │       ├── 45.png
    │   │       ├── 46.png
    │   │       ├── 47.png
    │   │       ├── 48.png
    │   │       ├── 49.png
    │   │       ├── 5.png
    │   │       ├── 50.png
    │   │       ├── 51.png
    │   │       ├── 52.png
    │   │       ├── 53.png
    │   │       ├── 54.png
    │   │       ├── 55.png
    │   │       ├── 56.png
    │   │       ├── 57.png
    │   │       ├── 58.png
    │   │       ├── 59.png
    │   │       ├── 6.png
    │   │       ├── 60.png
    │   │       ├── 61.png
    │   │       ├── 62.png
    │   │       ├── 63.png
    │   │       ├── 64.png
    │   │       ├── 65.png
    │   │       ├── 66.png
    │   │       ├── 67.png
    │   │       ├── 68.png
    │   │       ├── 69.png
    │   │       ├── 7.png
    │   │       ├── 70.png
    │   │       ├── 71.png
    │   │       ├── 72.png
    │   │       ├── 73.png
    │   │       ├── 74.png
    │   │       ├── 75.png
    │   │       ├── 76.png
    │   │       ├── 77.png
    │   │       ├── 78.png
    │   │       ├── 79.png
    │   │       ├── 8.png
    │   │       ├── 80.png
    │   │       ├── 81.png
    │   │       ├── 82.png
    │   │       ├── 83.png
    │   │       ├── 84.png
    │   │       ├── 85.png
    │   │       ├── 86.png
    │   │       ├── 87.png
    │   │       ├── 88.png
    │   │       ├── 89.png
    │   │       ├── 9.png
    │   │       ├── 90.png
    │   │       ├── 91.png
    │   │       ├── 92.png
    │   │       ├── 93.png
    │   │       ├── 94.png
    │   │       ├── 95.png
    │   │       ├── 96.png
    │   │       ├── 97.png
    │   │       ├── 98.png
    │   │       └── 99.png
    │   ├── val/
    │   │   ├── images/
    │   │   │   ├── 1.jpg
    │   │   │   ├── 117.jpg
    │   │   │   ├── 121.jpg
    │   │   │   ├── 130.jpg
    │   │   │   ├── 20.jpg
    │   │   │   ├── 3.jpg
    │   │   │   ├── 33.jpg
    │   │   │   ├── 4.jpg
    │   │   │   ├── 46.jpg
    │   │   │   ├── 51.jpg
    │   │   │   ├── 57.jpg
    │   │   │   ├── 62.jpg
    │   │   │   ├── 64.jpg
    │   │   │   ├── 74.jpg
    │   │   │   ├── 84.jpg
    │   │   │   ├── 86.jpg
    │   │   │   ├── 87.jpg
    │   │   │   ├── 89.jpg
    │   │   │   └── 95.jpg
    │   │   └── masks/
    │   │       ├── 1.png
    │   │       ├── 117.png
    │   │       ├── 121.png
    │   │       ├── 130.png
    │   │       ├── 20.png
    │   │       ├── 3.png
    │   │       ├── 33.png
    │   │       ├── 4.png
    │   │       ├── 46.png
    │   │       ├── 51.png
    │   │       ├── 57.png
    │   │       ├── 62.png
    │   │       ├── 64.png
    │   │       ├── 74.png
    │   │       ├── 84.png
    │   │       ├── 86.png
    │   │       ├── 87.png
    │   │       ├── 89.png
    │   │       └── 95.png
    │   └── README.md
    ├── deployments/
    │   ├── test/
    │   ├── test-crackseg-model/
    │   │   └── package/
    │   │       ├── app/
    │   │       │   ├── main.py
    │   │       │   └── streamlit_app.py
    │   │       ├── config/
    │   │       ├── docs/
    │   │       ├── scripts/
    │   │       └── tests/
    │   └── test-package/
    │       └── package/
    │           ├── app/
    │           │   ├── main.py
    │           │   └── streamlit_app.py
    │           ├── config/
    │           │   ├── app_config.json
    │           │   └── environment.json
    │           ├── docs/
    │           ├── helm/
    │           │   ├── Chart.yaml
    │           │   └── values.yaml
    │           ├── k8s/
    │           │   ├── deployment.yaml
    │           │   ├── hpa.yaml
    │           │   ├── ingress.yaml
    │           │   └── service.yaml
    │           ├── scripts/
    │           │   ├── deploy_docker.sh
    │           │   ├── deploy_kubernetes.sh
    │           │   └── health_check.py
    │           ├── tests/
    │           ├── docker-compose.yml
    │           ├── Dockerfile
    │           └── requirements.txt
    ├── docker/
    │   ├── __pycache__/
    │   ├── health_check/
    │   │   ├── analytics/
    │   │   │   ├── __init__.py
    │   │   │   ├── dashboard_generator.py
    │   │   │   ├── metrics_collector.py
    │   │   │   └── recommendation_engine.py
    │   │   ├── checkers/
    │   │   │   ├── __init__.py
    │   │   │   ├── dependency_validator.py
    │   │   │   ├── docker_checker.py
    │   │   │   └── endpoint_checker.py
    │   │   ├── cli/
    │   │   │   ├── __init__.py
    │   │   │   └── commands.py
    │   │   ├── models/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── config.py
    │   │   │   ├── enums.py
    │   │   │   └── results.py
    │   │   ├── orchestration/
    │   │   │   ├── __init__.py
    │   │   │   ├── health_orchestrator.py
    │   │   │   ├── monitoring.py
    │   │   │   └── service_registry.py
    │   │   ├── persistence/
    │   │   │   ├── __init__.py
    │   │   │   └── report_saver.py
    │   │   └── __init__.py
    │   ├── scripts/
    │   │   ├── artifact-manager.sh
    │   │   ├── browser-manager.sh
    │   │   ├── ci-setup.sh
    │   │   ├── docker-stack-manager.sh
    │   │   ├── e2e-test-orchestrator.sh
    │   │   ├── health-check-manager.sh
    │   │   ├── manage-grid.sh
    │   │   ├── network-manager.sh
    │   │   ├── run-e2e-tests.sh
    │   │   ├── run-test-runner.sh
    │   │   ├── setup-env.sh
    │   │   ├── setup-local-dev.sh
    │   │   ├── start-test-env.sh
    │   │   └── system-monitor.sh
    │   ├── browser-capabilities.json
    │   ├── docker-compose.README.md
    │   ├── docker-compose.test.yml
    │   ├── docker-entrypoint.sh
    │   ├── Dockerfile.streamlit
    │   ├── Dockerfile.test-runner
    │   ├── env-test.yml
    │   ├── env.local.template
    │   ├── env.production.template
    │   ├── env.staging.template
    │   ├── env.test.template
    │   ├── env_config.py
    │   ├── env_manager.py
    │   ├── env_utils.py
    │   ├── grid-config.json
    │   ├── health_check_system.py
    │   ├── mobile-browser-config.json
    │   ├── pytest.ini
    │   ├── README-ARCHITECTURE.md
    │   ├── README-DOCKER-TESTING.md
    │   ├── README-LOCAL-DEV.md
    │   ├── README-TROUBLESHOOTING.md
    │   ├── README-USAGE.md
    │   ├── README.artifact-management.md
    │   ├── README.cross-browser-testing.md
    │   ├── README.environment-management.md
    │   ├── README.md
    │   ├── README.network-setup.md
    │   ├── selenium-grid-guide.md
    │   ├── setup-local-dev.sh
    │   └── test-runner.config
    ├── docs/
    │   ├── analysis/
    │   │   └── singleton_pattern_analysis.md
    │   ├── api/
    │   │   ├── gui_components.md
    │   │   ├── gui_pages.md
    │   │   ├── gui_services.md
    │   │   ├── utilities.md
    │   │   └── visualization_api.md
    │   ├── designs/
    │   │   ├── logo.png
    │   │   └── loss_registry_design.md
    │   ├── guides/
    │   │   ├── architecture/
    │   │   │   ├── architectural_decisions.md
    │   │   │   ├── README.md
    │   │   │   └── TECHNICAL_ARCHITECTURE.md
    │   │   ├── cicd/
    │   │   │   ├── ci_cd_integration_guide.md
    │   │   │   ├── ci_cd_testing_integration.md
    │   │   │   └── README.md
    │   │   ├── deployment/
    │   │   │   └── DEPLOYMENT_PIPELINE_ARCHITECTURE.md
    │   │   ├── development/
    │   │   │   ├── CONTRIBUTING.md
    │   │   │   ├── gui_development_guidelines.md
    │   │   │   ├── README.md
    │   │   │   └── SYSTEM_DEPENDENCIES.md
    │   │   ├── experiment_tracker/
    │   │   │   ├── experiment_tracker_basic_usage.md
    │   │   │   ├── experiment_tracker_integration.md
    │   │   │   └── experiment_tracker_usage.md
    │   │   ├── experiments/
    │   │   │   └── README_swinv2_hybrid.md
    │   │   ├── monitoring/
    │   │   │   ├── continuous_coverage_monitoring_guide.md
    │   │   │   └── README.md
    │   │   ├── quality/
    │   │   │   ├── comprehensive_integration_test_reporting_guide.md
    │   │   │   ├── gui_testing_best_practices.md
    │   │   │   ├── gui_testing_implementation_checklist.md
    │   │   │   ├── quality_gates_guide.md
    │   │   │   ├── README.md
    │   │   │   └── test_maintenance_procedures.md
    │   │   ├── reporting/
    │   │   │   ├── experiment_reporter_architecture.md
    │   │   │   └── experiment_reporter_usage.md
    │   │   ├── specifications/
    │   │   │   ├── checkpoint_format_specification.md
    │   │   │   ├── configuration_storage_specification.md
    │   │   │   ├── performance_benchmarking_system.md
    │   │   │   ├── README.md
    │   │   │   └── traceability_data_model_specification.md
    │   │   ├── troubleshooting/
    │   │   │   ├── README.md
    │   │   │   └── TROUBLESHOOTING.md
    │   │   ├── usage/
    │   │   │   ├── focal_dice_loss_usage.md
    │   │   │   ├── loss_registry_usage.md
    │   │   │   ├── README.md
    │   │   │   └── USAGE.md
    │   │   ├── visualization/
    │   │   │   ├── visualization_customization_guide.md
    │   │   │   └── visualization_usage_examples.md
    │   │   ├── workflows/
    │   │   │   ├── CLEAN_INSTALLATION.md
    │   │   │   ├── README.md
    │   │   │   └── WORKFLOW_TRAINING.md
    │   │   ├── deployment_orchestration_api.md
    │   │   ├── experiment_tracker_usage.md
    │   │   ├── prediction_analysis_guide.md
    │   │   └── README.md
    │   ├── plans/
    │   │   └── artifact_system_development_plan.md
    │   ├── reports/
    │   │   ├── analysis/
    │   │   │   ├── basedpyright_analysis_report.md
    │   │   │   ├── consolidation-implementation-summary.md
    │   │   │   ├── duplication-mapping.md
    │   │   │   ├── final-rule-cleanup-summary.md
    │   │   │   ├── pytorch_cuda_compatibility_issue.md
    │   │   │   ├── rule-consolidation-report.md
    │   │   │   ├── rule-system-analysis.md
    │   │   │   └── tensorboard_component_refactoring_summary.md
    │   │   ├── coverage/
    │   │   │   ├── coverage_gaps_analysis.md
    │   │   │   ├── coverage_validation_report.md
    │   │   │   ├── test_coverage_analysis_report.md
    │   │   │   └── test_coverage_comparison_report.md
    │   │   ├── experiment_plots/
    │   │   │   ├── experiment_comparison_20250724_081112.csv
    │   │   │   ├── experiment_comparison_20250724_081136.csv
    │   │   │   ├── performance_radar_20250724_081112.png
    │   │   │   ├── performance_radar_20250724_081136.png
    │   │   │   ├── swinv2_hybrid_summary_20250724_081510.csv
    │   │   │   ├── swinv2_hybrid_training_curves_20250724_081510.png
    │   │   │   ├── training_curves_20250724_081112.png
    │   │   │   └── training_curves_20250724_081136.png
    │   │   ├── models/
    │   │   │   ├── model_expected_structure.json
    │   │   │   ├── model_imports_catalog.json
    │   │   │   ├── model_inventory.json
    │   │   │   ├── model_pyfiles.json
    │   │   │   └── model_structure_diff.json
    │   │   ├── project/
    │   │   │   ├── crackseg_paper.md
    │   │   │   ├── crackseg_paper_es.md
    │   │   │   ├── documentation_checklist.md
    │   │   │   ├── plan_verificacion_post_linting.md
    │   │   │   ├── project_tree.md
    │   │   │   └── technical_report.md
    │   │   ├── scripts/
    │   │   │   ├── example_prd.txt
    │   │   │   ├── hydra_examples.txt
    │   │   │   └── README.md
    │   │   ├── tasks/
    │   │   ├── testing/
    │   │   │   ├── automated_test_execution_report.md
    │   │   │   ├── gui_corrections_inventory.md
    │   │   │   ├── gui_test_coverage_analysis.md
    │   │   │   ├── next_testing_priorities.md
    │   │   │   ├── test_coverage_improvement_plan.md
    │   │   │   ├── test_fixes_validation_report.md
    │   │   │   └── test_inventory.txt
    │   │   ├── tutorial_02_plots/
    │   │   │   ├── experiment_comparison.csv
    │   │   │   ├── performance_radar.png
    │   │   │   └── training_curves.png
    │   │   ├── project_tree.md
    │   │   └── README.md
    │   ├── stylesheets/
    │   │   └── extra.css
    │   ├── testing/
    │   │   ├── artifact_testing_plan.md
    │   │   ├── test_patterns_and_best_practices.md
    │   │   └── visualization_testing_guide.md
    │   ├── tools/
    │   │   └── task-master-guide.md
    │   ├── tutorials/
    │   │   ├── cli/
    │   │   │   ├── 01_basic_training_cli.md
    │   │   │   ├── 02_custom_experiment_cli.md
    │   │   │   └── 03_extending_project_cli.md
    │   │   ├── gui/
    │   │   │   ├── 01_basic_training.md
    │   │   │   ├── 02_custom_experiment.md
    │   │   │   └── 03_extending_project.md
    │   │   └── README.md
    │   └── index.md
    ├── gui/
    │   ├── __pycache__/
    │   ├── assets/
    │   │   ├── __pycache__/
    │   │   ├── css/
    │   │   │   ├── components/
    │   │   │   │   ├── navigation.css
    │   │   │   │   └── README.md
    │   │   │   ├── global/
    │   │   │   │   ├── base.css
    │   │   │   │   └── README.md
    │   │   │   └── themes/
    │   │   │       └── README.md
    │   │   ├── fonts/
    │   │   │   └── primary/
    │   │   │       └── README.md
    │   │   ├── images/
    │   │   │   ├── backgrounds/
    │   │   │   │   └── README.md
    │   │   │   ├── icons/
    │   │   │   │   └── README.md
    │   │   │   ├── logos/
    │   │   │   │   ├── primary-logo.png
    │   │   │   │   └── README.md
    │   │   │   └── samples/
    │   │   │       └── README.md
    │   │   ├── js/
    │   │   │   └── components/
    │   │   │       └── README.md
    │   │   ├── manifest/
    │   │   │   ├── asset_registry.json
    │   │   │   └── optimization_config.json
    │   │   ├── init_assets.py
    │   │   ├── manager.py
    │   │   ├── README.md
    │   │   └── structure.md
    │   ├── components/
    │   │   ├── __pycache__/
    │   │   ├── config_editor/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── editor_core.py
    │   │   │   ├── file_browser_integration.py
    │   │   │   └── validation_panel.py
    │   │   ├── gallery/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── actions.py
    │   │   │   ├── event_handlers.py
    │   │   │   ├── renderer.py
    │   │   │   └── state_manager.py
    │   │   ├── tensorboard/
    │   │   │   ├── __pycache__/
    │   │   │   ├── recovery/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── recovery_strategies.py
    │   │   │   ├── rendering/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── diagnostics/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── action_controls.py
    │   │   │   │   │   └── diagnostic_panel.py
    │   │   │   │   ├── status_cards/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── base_card.py
    │   │   │   │   │   ├── health_card.py
    │   │   │   │   │   ├── network_card.py
    │   │   │   │   │   ├── process_card.py
    │   │   │   │   │   └── resource_card.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── advanced_status_renderer.py
    │   │   │   │   ├── control_renderer.py
    │   │   │   │   ├── error_renderer.py
    │   │   │   │   ├── iframe_renderer.py
    │   │   │   │   ├── startup_renderer.py
    │   │   │   │   └── status_renderer.py
    │   │   │   ├── state/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── progress_tracker.py
    │   │   │   │   └── session_manager.py
    │   │   │   ├── utils/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── formatters.py
    │   │   │   │   └── validators.py
    │   │   │   ├── __init__.py
    │   │   │   └── component.py
    │   │   ├── __init__.py
    │   │   ├── auto_save_manager.py
    │   │   ├── config_editor_component.py
    │   │   ├── confirmation_dialog.py
    │   │   ├── confirmation_renderer.py
    │   │   ├── confirmation_utils.py
    │   │   ├── device_detector.py
    │   │   ├── device_info.py
    │   │   ├── device_selector.py
    │   │   ├── device_selector_ui.py
    │   │   ├── error_console.py
    │   │   ├── file_browser.py
    │   │   ├── file_browser_component.py
    │   │   ├── file_upload_component.py
    │   │   ├── header_component.py
    │   │   ├── loading_spinner.py
    │   │   ├── loading_spinner_optimized.py
    │   │   ├── log_viewer.py
    │   │   ├── logo_component.py
    │   │   ├── metrics_viewer.py
    │   │   ├── page_router.py
    │   │   ├── progress_bar.py
    │   │   ├── progress_bar_optimized.py
    │   │   ├── results_display.py
    │   │   ├── results_gallery_component.py
    │   │   ├── sidebar_component.py
    │   │   ├── tensorboard_component.py
    │   │   └── theme_component.py
    │   ├── docs/
    │   │   ├── error_messaging_system.md
    │   │   ├── file_upload_guide.md
    │   │   └── tensorboard_integration_summary.md
    │   ├── pages/
    │   │   ├── __pycache__/
    │   │   ├── architecture/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── config_section.py
    │   │   │   ├── info_section.py
    │   │   │   ├── model_section.py
    │   │   │   ├── utils.py
    │   │   │   └── visualization_section.py
    │   │   ├── results/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── comparison_section.py
    │   │   │   ├── config_section.py
    │   │   │   ├── gallery_section.py
    │   │   │   ├── metrics_section.py
    │   │   │   ├── setup_section.py
    │   │   │   ├── tensorboard_section.py
    │   │   │   └── utils.py
    │   │   ├── __init__.py
    │   │   ├── advanced_config_page.py
    │   │   ├── architecture_page.py
    │   │   ├── config_page.py
    │   │   ├── home_page.py
    │   │   ├── page_train.py
    │   │   ├── results_page.py
    │   │   ├── results_page_new.py
    │   │   ├── train_page.py
    │   │   └── train_page.py.backup
    │   ├── services/
    │   │   ├── __pycache__/
    │   │   ├── __init__.py
    │   │   ├── gallery_export_service.py
    │   │   ├── gallery_scanner_service.py
    │   │   └── gpu_monitor.py
    │   ├── styles/
    │   │   └── main.css
    │   ├── utils/
    │   │   ├── __pycache__/
    │   │   ├── config/
    │   │   │   ├── __pycache__/
    │   │   │   ├── error_reporting/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── core_reporter.py
    │   │   │   │   ├── formatters.py
    │   │   │   │   ├── report_models.py
    │   │   │   │   └── utils.py
    │   │   │   ├── io/
    │   │   │   ├── parsing/
    │   │   │   ├── schema/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── constraint_validator.py
    │   │   │   │   ├── core_validator.py
    │   │   │   │   ├── type_validator.py
    │   │   │   │   └── utils.py
    │   │   │   ├── validation/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── error_categorizer.py
    │   │   │   │   └── yaml_engine.py
    │   │   │   ├── __init__.py
    │   │   │   ├── cache.py
    │   │   │   ├── config_loader.py
    │   │   │   ├── error_reporter.py
    │   │   │   ├── exceptions.py
    │   │   │   ├── formatters.py
    │   │   │   ├── io.py
    │   │   │   ├── parsing_engine.py
    │   │   │   ├── schema_validator.py
    │   │   │   └── templates.py
    │   │   ├── parsing/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── exceptions.py
    │   │   │   └── override_parser.py
    │   │   ├── process/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── abort_system.py
    │   │   │   ├── core.py
    │   │   │   ├── error_handling.py
    │   │   │   ├── log_integration.py
    │   │   │   ├── manager_backup.py
    │   │   │   ├── monitoring.py
    │   │   │   ├── override_parser.py
    │   │   │   └── states.py
    │   │   ├── reports/
    │   │   │   └── models.py
    │   │   ├── results/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── advanced_validation.py
    │   │   │   ├── cache.py
    │   │   │   ├── core.py
    │   │   │   ├── demo.py
    │   │   │   ├── demo_advanced_validation.py
    │   │   │   ├── demo_reactive.py
    │   │   │   ├── demo_streamlit_integration.py
    │   │   │   ├── events.py
    │   │   │   ├── results_validator.py
    │   │   │   ├── scanner.py
    │   │   │   └── validation.py
    │   │   ├── results_scanning/
    │   │   │   └── __init__.py
    │   │   ├── run_manager/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── abort_api.py
    │   │   │   ├── orchestrator.py
    │   │   │   ├── session_api.py
    │   │   │   ├── status_integration.py
    │   │   │   ├── status_updates.py
    │   │   │   ├── streaming_api.py
    │   │   │   ├── ui_integration.py
    │   │   │   └── ui_status_helpers.py
    │   │   ├── streaming/
    │   │   │   ├── __pycache__/
    │   │   │   ├── sources/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── file_watcher.py
    │   │   │   │   └── stdout_reader.py
    │   │   │   ├── __init__.py
    │   │   │   ├── core.py
    │   │   │   └── exceptions.py
    │   │   ├── tensorboard/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── core.py
    │   │   │   ├── demo_refactored.py
    │   │   │   ├── lifecycle_manager.py
    │   │   │   ├── manager.py
    │   │   │   ├── port_management.py
    │   │   │   └── process_manager.py
    │   │   ├── threading/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── cancellation.py
    │   │   │   ├── coordinator.py
    │   │   │   ├── progress_tracking.py
    │   │   │   ├── task_results.py
    │   │   │   ├── task_status.py
    │   │   │   ├── ui_responsive_backup.py
    │   │   │   └── ui_wrapper.py
    │   │   ├── __init__.py
    │   │   ├── architecture_viewer.py
    │   │   ├── auto_save.py
    │   │   ├── config_io.py
    │   │   ├── data_stats.py
    │   │   ├── demo_tensorboard.py
    │   │   ├── error_state.py
    │   │   ├── export_manager.py
    │   │   ├── gui_config.py
    │   │   ├── log_parser.py
    │   │   ├── override_examples.py
    │   │   ├── performance_optimizer.py
    │   │   ├── process_manager.py
    │   │   ├── save_dialog.py
    │   │   ├── session_state.py
    │   │   ├── session_sync.py
    │   │   ├── streaming_examples.py
    │   │   ├── styling.py
    │   │   ├── tb_manager.py
    │   │   ├── theme.py
    │   │   └── training_state.py
    │   ├── utils(results/
    │   ├── __init__.py
    │   ├── app.py
    │   ├── debug_page_rendering.py
    │   └── README.md
    ├── logs/
    ├── outputs/
    │   ├── checkpoints/
    │   │   ├── checkpoint_last.pth
    │   │   └── model_best.pth.tar
    │   ├── configurations/
    │   │   └── default_experiment/
    │   │       ├── config_epoch_0001.yaml
    │   │       ├── config_epoch_0001_validation.json
    │   │       ├── config_epoch_0002.yaml
    │   │       ├── config_epoch_0002_validation.json
    │   │       ├── config_epoch_0003.yaml
    │   │       ├── config_epoch_0003_validation.json
    │   │       ├── config_epoch_0004.yaml
    │   │       ├── config_epoch_0004_validation.json
    │   │       ├── config_epoch_0005.yaml
    │   │       ├── config_epoch_0005_validation.json
    │   │       ├── config_epoch_0006.yaml
    │   │       ├── config_epoch_0006_validation.json
    │   │       ├── config_epoch_0007.yaml
    │   │       ├── config_epoch_0007_validation.json
    │   │       ├── config_epoch_0008.yaml
    │   │       ├── config_epoch_0008_validation.json
    │   │       ├── config_epoch_0009.yaml
    │   │       ├── config_epoch_0009_validation.json
    │   │       ├── config_epoch_0010.yaml
    │   │       ├── config_epoch_0010_validation.json
    │   │       ├── config_epoch_0011.yaml
    │   │       ├── config_epoch_0011_validation.json
    │   │       ├── config_epoch_0012.yaml
    │   │       ├── config_epoch_0012_validation.json
    │   │       ├── config_epoch_0013.yaml
    │   │       ├── config_epoch_0013_validation.json
    │   │       ├── config_epoch_0014.yaml
    │   │       ├── config_epoch_0014_validation.json
    │   │       ├── config_epoch_0015.yaml
    │   │       ├── config_epoch_0015_validation.json
    │   │       ├── config_epoch_0016.yaml
    │   │       ├── config_epoch_0016_validation.json
    │   │       ├── config_epoch_0017.yaml
    │   │       ├── config_epoch_0017_validation.json
    │   │       ├── config_epoch_0018.yaml
    │   │       ├── config_epoch_0018_validation.json
    │   │       ├── config_epoch_0019.yaml
    │   │       ├── config_epoch_0019_validation.json
    │   │       ├── config_epoch_0020.yaml
    │   │       ├── config_epoch_0020_validation.json
    │   │       ├── config_epoch_0021.yaml
    │   │       ├── config_epoch_0021_validation.json
    │   │       ├── config_epoch_0022.yaml
    │   │       ├── config_epoch_0022_validation.json
    │   │       ├── config_epoch_0023.yaml
    │   │       ├── config_epoch_0023_validation.json
    │   │       ├── config_epoch_0024.yaml
    │   │       ├── config_epoch_0024_validation.json
    │   │       ├── config_epoch_0025.yaml
    │   │       ├── config_epoch_0025_validation.json
    │   │       ├── config_epoch_0026.yaml
    │   │       ├── config_epoch_0026_validation.json
    │   │       ├── config_epoch_0027.yaml
    │   │       ├── config_epoch_0027_validation.json
    │   │       ├── config_epoch_0028.yaml
    │   │       ├── config_epoch_0028_validation.json
    │   │       ├── config_epoch_0029.yaml
    │   │       ├── config_epoch_0029_validation.json
    │   │       ├── config_epoch_0030.yaml
    │   │       ├── config_epoch_0030_validation.json
    │   │       ├── config_epoch_0031.yaml
    │   │       ├── config_epoch_0031_validation.json
    │   │       ├── config_epoch_0032.yaml
    │   │       ├── config_epoch_0032_validation.json
    │   │       ├── config_epoch_0033.yaml
    │   │       ├── config_epoch_0033_validation.json
    │   │       ├── config_epoch_0034.yaml
    │   │       ├── config_epoch_0034_validation.json
    │   │       ├── config_epoch_0035.yaml
    │   │       ├── config_epoch_0035_validation.json
    │   │       ├── config_epoch_0036.yaml
    │   │       ├── config_epoch_0036_validation.json
    │   │       ├── config_epoch_0037.yaml
    │   │       ├── config_epoch_0037_validation.json
    │   │       ├── config_epoch_0038.yaml
    │   │       ├── config_epoch_0038_validation.json
    │   │       ├── config_epoch_0039.yaml
    │   │       ├── config_epoch_0039_validation.json
    │   │       ├── config_epoch_0040.yaml
    │   │       ├── config_epoch_0040_validation.json
    │   │       ├── config_epoch_0041.yaml
    │   │       ├── config_epoch_0041_validation.json
    │   │       ├── config_epoch_0042.yaml
    │   │       ├── config_epoch_0042_validation.json
    │   │       ├── config_epoch_0043.yaml
    │   │       ├── config_epoch_0043_validation.json
    │   │       ├── config_epoch_0044.yaml
    │   │       ├── config_epoch_0044_validation.json
    │   │       ├── config_epoch_0045.yaml
    │   │       ├── config_epoch_0045_validation.json
    │   │       ├── config_epoch_0046.yaml
    │   │       ├── config_epoch_0046_validation.json
    │   │       ├── config_epoch_0047.yaml
    │   │       ├── config_epoch_0047_validation.json
    │   │       ├── config_epoch_0048.yaml
    │   │       ├── config_epoch_0048_validation.json
    │   │       ├── config_epoch_0049.yaml
    │   │       ├── config_epoch_0049_validation.json
    │   │       ├── config_epoch_0050.yaml
    │   │       ├── config_epoch_0050_validation.json
    │   │       ├── config_epoch_0051.yaml
    │   │       ├── config_epoch_0051_validation.json
    │   │       ├── config_epoch_0052.yaml
    │   │       ├── config_epoch_0052_validation.json
    │   │       ├── config_epoch_0053.yaml
    │   │       ├── config_epoch_0053_validation.json
    │   │       ├── config_epoch_0054.yaml
    │   │       ├── config_epoch_0054_validation.json
    │   │       ├── config_epoch_0055.yaml
    │   │       ├── config_epoch_0055_validation.json
    │   │       ├── config_epoch_0056.yaml
    │   │       ├── config_epoch_0056_validation.json
    │   │       ├── config_epoch_0057.yaml
    │   │       ├── config_epoch_0057_validation.json
    │   │       ├── config_epoch_0058.yaml
    │   │       ├── config_epoch_0058_validation.json
    │   │       ├── config_epoch_0059.yaml
    │   │       ├── config_epoch_0059_validation.json
    │   │       ├── config_epoch_0060.yaml
    │   │       ├── config_epoch_0060_validation.json
    │   │       ├── config_epoch_0061.yaml
    │   │       ├── config_epoch_0061_validation.json
    │   │       ├── config_epoch_0062.yaml
    │   │       ├── config_epoch_0062_validation.json
    │   │       ├── config_epoch_0063.yaml
    │   │       ├── config_epoch_0063_validation.json
    │   │       ├── config_epoch_0064.yaml
    │   │       ├── config_epoch_0064_validation.json
    │   │       ├── config_epoch_0065.yaml
    │   │       ├── config_epoch_0065_validation.json
    │   │       ├── config_epoch_0066.yaml
    │   │       ├── config_epoch_0066_validation.json
    │   │       ├── config_epoch_0067.yaml
    │   │       ├── config_epoch_0067_validation.json
    │   │       ├── config_epoch_0068.yaml
    │   │       ├── config_epoch_0068_validation.json
    │   │       ├── config_epoch_0069.yaml
    │   │       ├── config_epoch_0069_validation.json
    │   │       ├── config_epoch_0070.yaml
    │   │       ├── config_epoch_0070_validation.json
    │   │       ├── config_epoch_0071.yaml
    │   │       ├── config_epoch_0071_validation.json
    │   │       ├── config_epoch_0072.yaml
    │   │       ├── config_epoch_0072_validation.json
    │   │       ├── config_epoch_0073.yaml
    │   │       ├── config_epoch_0073_validation.json
    │   │       ├── config_epoch_0074.yaml
    │   │       ├── config_epoch_0074_validation.json
    │   │       ├── config_epoch_0075.yaml
    │   │       ├── config_epoch_0075_validation.json
    │   │       ├── config_epoch_0076.yaml
    │   │       ├── config_epoch_0076_validation.json
    │   │       ├── config_epoch_0077.yaml
    │   │       ├── config_epoch_0077_validation.json
    │   │       ├── config_epoch_0078.yaml
    │   │       ├── config_epoch_0078_validation.json
    │   │       ├── config_epoch_0079.yaml
    │   │       ├── config_epoch_0079_validation.json
    │   │       ├── config_epoch_0080.yaml
    │   │       ├── config_epoch_0080_validation.json
    │   │       ├── config_epoch_0081.yaml
    │   │       ├── config_epoch_0081_validation.json
    │   │       ├── config_epoch_0082.yaml
    │   │       ├── config_epoch_0082_validation.json
    │   │       ├── config_epoch_0083.yaml
    │   │       ├── config_epoch_0083_validation.json
    │   │       ├── config_epoch_0084.yaml
    │   │       ├── config_epoch_0084_validation.json
    │   │       ├── config_epoch_0085.yaml
    │   │       ├── config_epoch_0085_validation.json
    │   │       ├── config_epoch_0086.yaml
    │   │       ├── config_epoch_0086_validation.json
    │   │       ├── config_epoch_0087.yaml
    │   │       ├── config_epoch_0087_validation.json
    │   │       ├── config_epoch_0088.yaml
    │   │       ├── config_epoch_0088_validation.json
    │   │       ├── config_epoch_0089.yaml
    │   │       ├── config_epoch_0089_validation.json
    │   │       ├── config_epoch_0090.yaml
    │   │       ├── config_epoch_0090_validation.json
    │   │       ├── config_epoch_0091.yaml
    │   │       ├── config_epoch_0091_validation.json
    │   │       ├── config_epoch_0092.yaml
    │   │       ├── config_epoch_0092_validation.json
    │   │       ├── config_epoch_0093.yaml
    │   │       ├── config_epoch_0093_validation.json
    │   │       ├── config_epoch_0094.yaml
    │   │       ├── config_epoch_0094_validation.json
    │   │       ├── config_epoch_0095.yaml
    │   │       ├── config_epoch_0095_validation.json
    │   │       ├── config_epoch_0096.yaml
    │   │       ├── config_epoch_0096_validation.json
    │   │       ├── config_epoch_0097.yaml
    │   │       ├── config_epoch_0097_validation.json
    │   │       ├── config_epoch_0098.yaml
    │   │       ├── config_epoch_0098_validation.json
    │   │       ├── config_epoch_0099.yaml
    │   │       ├── config_epoch_0099_validation.json
    │   │       ├── config_epoch_0100.yaml
    │   │       ├── config_epoch_0100_validation.json
    │   │       ├── config_epoch_0101.yaml
    │   │       ├── config_epoch_0101_validation.json
    │   │       ├── config_epoch_0102.yaml
    │   │       ├── config_epoch_0102_validation.json
    │   │       ├── config_epoch_0103.yaml
    │   │       ├── config_epoch_0103_validation.json
    │   │       ├── config_epoch_0104.yaml
    │   │       ├── config_epoch_0104_validation.json
    │   │       ├── config_epoch_0105.yaml
    │   │       ├── config_epoch_0105_validation.json
    │   │       ├── config_epoch_0106.yaml
    │   │       ├── config_epoch_0106_validation.json
    │   │       ├── config_epoch_0107.yaml
    │   │       ├── config_epoch_0107_validation.json
    │   │       ├── config_epoch_0108.yaml
    │   │       ├── config_epoch_0108_validation.json
    │   │       ├── config_epoch_0109.yaml
    │   │       ├── config_epoch_0109_validation.json
    │   │       ├── config_epoch_0110.yaml
    │   │       ├── config_epoch_0110_validation.json
    │   │       ├── config_epoch_0111.yaml
    │   │       ├── config_epoch_0111_validation.json
    │   │       ├── config_epoch_0112.yaml
    │   │       ├── config_epoch_0112_validation.json
    │   │       ├── config_epoch_0113.yaml
    │   │       ├── config_epoch_0113_validation.json
    │   │       ├── config_epoch_0114.yaml
    │   │       ├── config_epoch_0114_validation.json
    │   │       ├── config_epoch_0115.yaml
    │   │       ├── config_epoch_0115_validation.json
    │   │       ├── config_epoch_0116.yaml
    │   │       ├── config_epoch_0116_validation.json
    │   │       ├── config_epoch_0117.yaml
    │   │       ├── config_epoch_0117_validation.json
    │   │       ├── config_epoch_0118.yaml
    │   │       ├── config_epoch_0118_validation.json
    │   │       ├── config_epoch_0119.yaml
    │   │       ├── config_epoch_0119_validation.json
    │   │       ├── config_epoch_0120.yaml
    │   │       ├── config_epoch_0120_validation.json
    │   │       ├── config_epoch_0121.yaml
    │   │       ├── config_epoch_0121_validation.json
    │   │       ├── config_epoch_0122.yaml
    │   │       ├── config_epoch_0122_validation.json
    │   │       ├── config_epoch_0123.yaml
    │   │       ├── config_epoch_0123_validation.json
    │   │       ├── config_epoch_0124.yaml
    │   │       ├── config_epoch_0124_validation.json
    │   │       ├── config_epoch_0125.yaml
    │   │       ├── config_epoch_0125_validation.json
    │   │       ├── config_epoch_0126.yaml
    │   │       ├── config_epoch_0126_validation.json
    │   │       ├── config_epoch_0127.yaml
    │   │       ├── config_epoch_0127_validation.json
    │   │       ├── config_epoch_0128.yaml
    │   │       ├── config_epoch_0128_validation.json
    │   │       ├── config_epoch_0129.yaml
    │   │       ├── config_epoch_0129_validation.json
    │   │       ├── config_epoch_0130.yaml
    │   │       ├── config_epoch_0130_validation.json
    │   │       ├── config_epoch_0131.yaml
    │   │       ├── config_epoch_0131_validation.json
    │   │       ├── config_epoch_0132.yaml
    │   │       ├── config_epoch_0132_validation.json
    │   │       ├── config_epoch_0133.yaml
    │   │       ├── config_epoch_0133_validation.json
    │   │       ├── config_epoch_0134.yaml
    │   │       ├── config_epoch_0134_validation.json
    │   │       ├── config_epoch_0135.yaml
    │   │       ├── config_epoch_0135_validation.json
    │   │       ├── config_epoch_0136.yaml
    │   │       ├── config_epoch_0136_validation.json
    │   │       ├── config_epoch_0137.yaml
    │   │       ├── config_epoch_0137_validation.json
    │   │       ├── config_epoch_0138.yaml
    │   │       ├── config_epoch_0138_validation.json
    │   │       ├── config_epoch_0139.yaml
    │   │       ├── config_epoch_0139_validation.json
    │   │       ├── config_epoch_0140.yaml
    │   │       ├── config_epoch_0140_validation.json
    │   │       ├── config_epoch_0141.yaml
    │   │       ├── config_epoch_0141_validation.json
    │   │       ├── config_epoch_0142.yaml
    │   │       ├── config_epoch_0142_validation.json
    │   │       ├── config_epoch_0143.yaml
    │   │       ├── config_epoch_0143_validation.json
    │   │       ├── config_epoch_0144.yaml
    │   │       ├── config_epoch_0144_validation.json
    │   │       ├── config_epoch_0145.yaml
    │   │       ├── config_epoch_0145_validation.json
    │   │       ├── config_epoch_0146.yaml
    │   │       ├── config_epoch_0146_validation.json
    │   │       ├── config_epoch_0147.yaml
    │   │       ├── config_epoch_0147_validation.json
    │   │       ├── config_epoch_0148.yaml
    │   │       ├── config_epoch_0148_validation.json
    │   │       ├── config_epoch_0149.yaml
    │   │       ├── config_epoch_0149_validation.json
    │   │       ├── config_epoch_0150.yaml
    │   │       ├── config_epoch_0150_validation.json
    │   │       ├── config_epoch_0151.yaml
    │   │       ├── config_epoch_0151_validation.json
    │   │       ├── config_epoch_0152.yaml
    │   │       ├── config_epoch_0152_validation.json
    │   │       ├── config_epoch_0153.yaml
    │   │       ├── config_epoch_0153_validation.json
    │   │       ├── config_epoch_0154.yaml
    │   │       ├── config_epoch_0154_validation.json
    │   │       ├── config_epoch_0155.yaml
    │   │       ├── config_epoch_0155_validation.json
    │   │       ├── config_epoch_0156.yaml
    │   │       ├── config_epoch_0156_validation.json
    │   │       ├── config_epoch_0157.yaml
    │   │       ├── config_epoch_0157_validation.json
    │   │       ├── config_epoch_0158.yaml
    │   │       ├── config_epoch_0158_validation.json
    │   │       ├── config_epoch_0159.yaml
    │   │       ├── config_epoch_0159_validation.json
    │   │       ├── config_epoch_0160.yaml
    │   │       ├── config_epoch_0160_validation.json
    │   │       ├── config_epoch_0161.yaml
    │   │       ├── config_epoch_0161_validation.json
    │   │       ├── config_epoch_0162.yaml
    │   │       ├── config_epoch_0162_validation.json
    │   │       ├── config_epoch_0163.yaml
    │   │       ├── config_epoch_0163_validation.json
    │   │       ├── config_epoch_0164.yaml
    │   │       ├── config_epoch_0164_validation.json
    │   │       ├── config_epoch_0165.yaml
    │   │       ├── config_epoch_0165_validation.json
    │   │       ├── config_epoch_0166.yaml
    │   │       ├── config_epoch_0166_validation.json
    │   │       ├── config_epoch_0167.yaml
    │   │       ├── config_epoch_0167_validation.json
    │   │       ├── config_epoch_0168.yaml
    │   │       ├── config_epoch_0168_validation.json
    │   │       ├── config_epoch_0169.yaml
    │   │       ├── config_epoch_0169_validation.json
    │   │       ├── config_epoch_0170.yaml
    │   │       ├── config_epoch_0170_validation.json
    │   │       ├── config_epoch_0171.yaml
    │   │       ├── config_epoch_0171_validation.json
    │   │       ├── config_epoch_0172.yaml
    │   │       ├── config_epoch_0172_validation.json
    │   │       ├── config_epoch_0173.yaml
    │   │       ├── config_epoch_0173_validation.json
    │   │       ├── config_epoch_0174.yaml
    │   │       ├── config_epoch_0174_validation.json
    │   │       ├── config_epoch_0175.yaml
    │   │       ├── config_epoch_0175_validation.json
    │   │       ├── config_epoch_0176.yaml
    │   │       ├── config_epoch_0176_validation.json
    │   │       ├── config_epoch_0177.yaml
    │   │       ├── config_epoch_0177_validation.json
    │   │       ├── config_epoch_0178.yaml
    │   │       ├── config_epoch_0178_validation.json
    │   │       ├── config_epoch_0179.yaml
    │   │       ├── config_epoch_0179_validation.json
    │   │       ├── config_epoch_0180.yaml
    │   │       ├── config_epoch_0180_validation.json
    │   │       ├── config_epoch_0181.yaml
    │   │       ├── config_epoch_0181_validation.json
    │   │       ├── config_epoch_0182.yaml
    │   │       ├── config_epoch_0182_validation.json
    │   │       ├── config_epoch_0183.yaml
    │   │       ├── config_epoch_0183_validation.json
    │   │       ├── config_epoch_0184.yaml
    │   │       ├── config_epoch_0184_validation.json
    │   │       ├── config_epoch_0185.yaml
    │   │       ├── config_epoch_0185_validation.json
    │   │       ├── config_epoch_0186.yaml
    │   │       ├── config_epoch_0186_validation.json
    │   │       ├── config_epoch_0187.yaml
    │   │       ├── config_epoch_0187_validation.json
    │   │       ├── config_epoch_0188.yaml
    │   │       ├── config_epoch_0188_validation.json
    │   │       ├── config_epoch_0189.yaml
    │   │       ├── config_epoch_0189_validation.json
    │   │       ├── config_epoch_0190.yaml
    │   │       ├── config_epoch_0190_validation.json
    │   │       ├── config_epoch_0191.yaml
    │   │       ├── config_epoch_0191_validation.json
    │   │       ├── config_epoch_0192.yaml
    │   │       ├── config_epoch_0192_validation.json
    │   │       ├── config_epoch_0193.yaml
    │   │       ├── config_epoch_0193_validation.json
    │   │       ├── config_epoch_0194.yaml
    │   │       ├── config_epoch_0194_validation.json
    │   │       ├── config_epoch_0195.yaml
    │   │       ├── config_epoch_0195_validation.json
    │   │       ├── config_epoch_0196.yaml
    │   │       ├── config_epoch_0196_validation.json
    │   │       ├── config_epoch_0197.yaml
    │   │       ├── config_epoch_0197_validation.json
    │   │       ├── config_epoch_0198.yaml
    │   │       ├── config_epoch_0198_validation.json
    │   │       ├── config_epoch_0199.yaml
    │   │       ├── config_epoch_0199_validation.json
    │   │       ├── config_epoch_0200.yaml
    │   │       ├── config_epoch_0200_validation.json
    │   │       ├── config_epoch_0201.yaml
    │   │       ├── config_epoch_0201_validation.json
    │   │       ├── config_epoch_0202.yaml
    │   │       ├── config_epoch_0202_validation.json
    │   │       ├── config_epoch_0203.yaml
    │   │       ├── config_epoch_0203_validation.json
    │   │       ├── config_epoch_0204.yaml
    │   │       ├── config_epoch_0204_validation.json
    │   │       ├── config_epoch_0205.yaml
    │   │       ├── config_epoch_0205_validation.json
    │   │       ├── config_epoch_0206.yaml
    │   │       ├── config_epoch_0206_validation.json
    │   │       ├── config_epoch_0207.yaml
    │   │       ├── config_epoch_0207_validation.json
    │   │       ├── config_epoch_0208.yaml
    │   │       ├── config_epoch_0208_validation.json
    │   │       ├── config_epoch_0209.yaml
    │   │       ├── config_epoch_0209_validation.json
    │   │       ├── config_epoch_0210.yaml
    │   │       ├── config_epoch_0210_validation.json
    │   │       ├── config_epoch_0211.yaml
    │   │       ├── config_epoch_0211_validation.json
    │   │       ├── config_epoch_0212.yaml
    │   │       ├── config_epoch_0212_validation.json
    │   │       ├── config_epoch_0213.yaml
    │   │       ├── config_epoch_0213_validation.json
    │   │       ├── config_epoch_0214.yaml
    │   │       ├── config_epoch_0214_validation.json
    │   │       ├── config_epoch_0215.yaml
    │   │       ├── config_epoch_0215_validation.json
    │   │       ├── config_epoch_0216.yaml
    │   │       ├── config_epoch_0216_validation.json
    │   │       ├── config_epoch_0217.yaml
    │   │       ├── config_epoch_0217_validation.json
    │   │       ├── config_epoch_0218.yaml
    │   │       ├── config_epoch_0218_validation.json
    │   │       ├── config_epoch_0219.yaml
    │   │       ├── config_epoch_0219_validation.json
    │   │       ├── config_epoch_0220.yaml
    │   │       ├── config_epoch_0220_validation.json
    │   │       ├── config_epoch_0221.yaml
    │   │       ├── config_epoch_0221_validation.json
    │   │       ├── config_epoch_0222.yaml
    │   │       ├── config_epoch_0222_validation.json
    │   │       ├── config_epoch_0223.yaml
    │   │       ├── config_epoch_0223_validation.json
    │   │       ├── config_epoch_0224.yaml
    │   │       ├── config_epoch_0224_validation.json
    │   │       ├── config_epoch_0225.yaml
    │   │       ├── config_epoch_0225_validation.json
    │   │       ├── config_epoch_0226.yaml
    │   │       ├── config_epoch_0226_validation.json
    │   │       ├── config_epoch_0227.yaml
    │   │       ├── config_epoch_0227_validation.json
    │   │       ├── config_epoch_0228.yaml
    │   │       ├── config_epoch_0228_validation.json
    │   │       ├── config_epoch_0229.yaml
    │   │       ├── config_epoch_0229_validation.json
    │   │       ├── config_epoch_0230.yaml
    │   │       ├── config_epoch_0230_validation.json
    │   │       ├── config_epoch_0231.yaml
    │   │       ├── config_epoch_0231_validation.json
    │   │       ├── config_epoch_0232.yaml
    │   │       ├── config_epoch_0232_validation.json
    │   │       ├── config_epoch_0233.yaml
    │   │       ├── config_epoch_0233_validation.json
    │   │       ├── config_epoch_0234.yaml
    │   │       ├── config_epoch_0234_validation.json
    │   │       ├── config_epoch_0235.yaml
    │   │       ├── config_epoch_0235_validation.json
    │   │       ├── config_epoch_0236.yaml
    │   │       ├── config_epoch_0236_validation.json
    │   │       ├── config_epoch_0237.yaml
    │   │       ├── config_epoch_0237_validation.json
    │   │       ├── config_epoch_0238.yaml
    │   │       ├── config_epoch_0238_validation.json
    │   │       ├── config_epoch_0239.yaml
    │   │       ├── config_epoch_0239_validation.json
    │   │       ├── config_epoch_0240.yaml
    │   │       ├── config_epoch_0240_validation.json
    │   │       ├── config_epoch_0241.yaml
    │   │       ├── config_epoch_0241_validation.json
    │   │       ├── config_epoch_0242.yaml
    │   │       ├── config_epoch_0242_validation.json
    │   │       ├── config_epoch_0243.yaml
    │   │       ├── config_epoch_0243_validation.json
    │   │       ├── config_epoch_0244.yaml
    │   │       ├── config_epoch_0244_validation.json
    │   │       ├── config_epoch_0245.yaml
    │   │       ├── config_epoch_0245_validation.json
    │   │       ├── config_epoch_0246.yaml
    │   │       ├── config_epoch_0246_validation.json
    │   │       ├── config_epoch_0247.yaml
    │   │       ├── config_epoch_0247_validation.json
    │   │       ├── config_epoch_0248.yaml
    │   │       ├── config_epoch_0248_validation.json
    │   │       ├── config_epoch_0249.yaml
    │   │       ├── config_epoch_0249_validation.json
    │   │       ├── config_epoch_0250.yaml
    │   │       ├── config_epoch_0250_validation.json
    │   │       ├── config_epoch_0251.yaml
    │   │       ├── config_epoch_0251_validation.json
    │   │       ├── config_epoch_0252.yaml
    │   │       ├── config_epoch_0252_validation.json
    │   │       ├── config_epoch_0253.yaml
    │   │       ├── config_epoch_0253_validation.json
    │   │       ├── config_epoch_0254.yaml
    │   │       ├── config_epoch_0254_validation.json
    │   │       ├── config_epoch_0255.yaml
    │   │       ├── config_epoch_0255_validation.json
    │   │       ├── config_epoch_0256.yaml
    │   │       ├── config_epoch_0256_validation.json
    │   │       ├── config_epoch_0257.yaml
    │   │       ├── config_epoch_0257_validation.json
    │   │       ├── config_epoch_0258.yaml
    │   │       ├── config_epoch_0258_validation.json
    │   │       ├── config_epoch_0259.yaml
    │   │       ├── config_epoch_0259_validation.json
    │   │       ├── config_epoch_0260.yaml
    │   │       ├── config_epoch_0260_validation.json
    │   │       ├── config_epoch_0261.yaml
    │   │       ├── config_epoch_0261_validation.json
    │   │       ├── config_epoch_0262.yaml
    │   │       ├── config_epoch_0262_validation.json
    │   │       ├── config_epoch_0263.yaml
    │   │       ├── config_epoch_0263_validation.json
    │   │       ├── config_epoch_0264.yaml
    │   │       ├── config_epoch_0264_validation.json
    │   │       ├── config_epoch_0265.yaml
    │   │       ├── config_epoch_0265_validation.json
    │   │       ├── config_epoch_0266.yaml
    │   │       ├── config_epoch_0266_validation.json
    │   │       ├── config_epoch_0267.yaml
    │   │       ├── config_epoch_0267_validation.json
    │   │       ├── config_epoch_0268.yaml
    │   │       ├── config_epoch_0268_validation.json
    │   │       ├── config_epoch_0269.yaml
    │   │       ├── config_epoch_0269_validation.json
    │   │       ├── config_epoch_0270.yaml
    │   │       ├── config_epoch_0270_validation.json
    │   │       ├── config_epoch_0271.yaml
    │   │       ├── config_epoch_0271_validation.json
    │   │       ├── config_epoch_0272.yaml
    │   │       ├── config_epoch_0272_validation.json
    │   │       ├── config_epoch_0273.yaml
    │   │       ├── config_epoch_0273_validation.json
    │   │       ├── config_epoch_0274.yaml
    │   │       ├── config_epoch_0274_validation.json
    │   │       ├── config_epoch_0275.yaml
    │   │       ├── config_epoch_0275_validation.json
    │   │       ├── config_epoch_0276.yaml
    │   │       ├── config_epoch_0276_validation.json
    │   │       ├── config_epoch_0277.yaml
    │   │       ├── config_epoch_0277_validation.json
    │   │       ├── config_epoch_0278.yaml
    │   │       ├── config_epoch_0278_validation.json
    │   │       ├── config_epoch_0279.yaml
    │   │       ├── config_epoch_0279_validation.json
    │   │       ├── config_epoch_0280.yaml
    │   │       ├── config_epoch_0280_validation.json
    │   │       ├── config_epoch_0281.yaml
    │   │       ├── config_epoch_0281_validation.json
    │   │       ├── config_epoch_0282.yaml
    │   │       ├── config_epoch_0282_validation.json
    │   │       ├── config_epoch_0283.yaml
    │   │       ├── config_epoch_0283_validation.json
    │   │       ├── config_epoch_0284.yaml
    │   │       ├── config_epoch_0284_validation.json
    │   │       ├── config_epoch_0285.yaml
    │   │       ├── config_epoch_0285_validation.json
    │   │       ├── config_epoch_0286.yaml
    │   │       ├── config_epoch_0286_validation.json
    │   │       ├── config_epoch_0287.yaml
    │   │       ├── config_epoch_0287_validation.json
    │   │       ├── config_epoch_0288.yaml
    │   │       ├── config_epoch_0288_validation.json
    │   │       ├── config_epoch_0289.yaml
    │   │       ├── config_epoch_0289_validation.json
    │   │       ├── config_epoch_0290.yaml
    │   │       ├── config_epoch_0290_validation.json
    │   │       ├── config_epoch_0291.yaml
    │   │       ├── config_epoch_0291_validation.json
    │   │       ├── config_epoch_0292.yaml
    │   │       ├── config_epoch_0292_validation.json
    │   │       ├── config_epoch_0293.yaml
    │   │       ├── config_epoch_0293_validation.json
    │   │       ├── config_epoch_0294.yaml
    │   │       ├── config_epoch_0294_validation.json
    │   │       ├── config_epoch_0295.yaml
    │   │       ├── config_epoch_0295_validation.json
    │   │       ├── config_epoch_0296.yaml
    │   │       ├── config_epoch_0296_validation.json
    │   │       ├── config_epoch_0297.yaml
    │   │       ├── config_epoch_0297_validation.json
    │   │       ├── config_epoch_0298.yaml
    │   │       ├── config_epoch_0298_validation.json
    │   │       ├── config_epoch_0299.yaml
    │   │       ├── config_epoch_0299_validation.json
    │   │       ├── config_epoch_0300.yaml
    │   │       ├── config_epoch_0300_validation.json
    │   │       ├── config_epoch_0301.yaml
    │   │       ├── config_epoch_0301_validation.json
    │   │       ├── config_epoch_0302.yaml
    │   │       ├── config_epoch_0302_validation.json
    │   │       ├── config_epoch_0303.yaml
    │   │       ├── config_epoch_0303_validation.json
    │   │       ├── config_epoch_0304.yaml
    │   │       ├── config_epoch_0304_validation.json
    │   │       ├── config_epoch_0305.yaml
    │   │       ├── config_epoch_0305_validation.json
    │   │       ├── config_epoch_0306.yaml
    │   │       ├── config_epoch_0306_validation.json
    │   │       ├── config_epoch_0307.yaml
    │   │       ├── config_epoch_0307_validation.json
    │   │       ├── config_epoch_0308.yaml
    │   │       ├── config_epoch_0308_validation.json
    │   │       ├── config_epoch_0309.yaml
    │   │       ├── config_epoch_0309_validation.json
    │   │       ├── config_epoch_0310.yaml
    │   │       ├── config_epoch_0310_validation.json
    │   │       ├── config_epoch_0311.yaml
    │   │       ├── config_epoch_0311_validation.json
    │   │       ├── config_epoch_0312.yaml
    │   │       ├── config_epoch_0312_validation.json
    │   │       ├── config_epoch_0313.yaml
    │   │       ├── config_epoch_0313_validation.json
    │   │       ├── config_epoch_0314.yaml
    │   │       ├── config_epoch_0314_validation.json
    │   │       ├── config_epoch_0315.yaml
    │   │       ├── config_epoch_0315_validation.json
    │   │       ├── config_epoch_0316.yaml
    │   │       ├── config_epoch_0316_validation.json
    │   │       ├── config_epoch_0317.yaml
    │   │       ├── config_epoch_0317_validation.json
    │   │       ├── config_epoch_0318.yaml
    │   │       ├── config_epoch_0318_validation.json
    │   │       ├── config_epoch_0319.yaml
    │   │       ├── config_epoch_0319_validation.json
    │   │       ├── config_epoch_0320.yaml
    │   │       ├── config_epoch_0320_validation.json
    │   │       ├── config_epoch_0321.yaml
    │   │       ├── config_epoch_0321_validation.json
    │   │       ├── config_epoch_0322.yaml
    │   │       ├── config_epoch_0322_validation.json
    │   │       ├── config_epoch_0323.yaml
    │   │       ├── config_epoch_0323_validation.json
    │   │       ├── config_epoch_0324.yaml
    │   │       ├── config_epoch_0324_validation.json
    │   │       ├── config_epoch_0325.yaml
    │   │       ├── config_epoch_0325_validation.json
    │   │       ├── config_epoch_0326.yaml
    │   │       ├── config_epoch_0326_validation.json
    │   │       ├── config_epoch_0327.yaml
    │   │       ├── config_epoch_0327_validation.json
    │   │       ├── config_epoch_0328.yaml
    │   │       ├── config_epoch_0328_validation.json
    │   │       ├── config_epoch_0329.yaml
    │   │       ├── config_epoch_0329_validation.json
    │   │       ├── config_epoch_0330.yaml
    │   │       ├── config_epoch_0330_validation.json
    │   │       ├── config_epoch_0331.yaml
    │   │       ├── config_epoch_0331_validation.json
    │   │       ├── config_epoch_0332.yaml
    │   │       ├── config_epoch_0332_validation.json
    │   │       ├── config_epoch_0333.yaml
    │   │       ├── config_epoch_0333_validation.json
    │   │       ├── config_epoch_0334.yaml
    │   │       ├── config_epoch_0334_validation.json
    │   │       ├── config_epoch_0335.yaml
    │   │       ├── config_epoch_0335_validation.json
    │   │       ├── config_epoch_0336.yaml
    │   │       ├── config_epoch_0336_validation.json
    │   │       ├── config_epoch_0337.yaml
    │   │       ├── config_epoch_0337_validation.json
    │   │       ├── config_epoch_0338.yaml
    │   │       ├── config_epoch_0338_validation.json
    │   │       ├── config_epoch_0339.yaml
    │   │       ├── config_epoch_0339_validation.json
    │   │       ├── config_epoch_0340.yaml
    │   │       ├── config_epoch_0340_validation.json
    │   │       ├── config_epoch_0341.yaml
    │   │       ├── config_epoch_0341_validation.json
    │   │       ├── config_epoch_0342.yaml
    │   │       ├── config_epoch_0342_validation.json
    │   │       ├── config_epoch_0343.yaml
    │   │       ├── config_epoch_0343_validation.json
    │   │       ├── config_epoch_0344.yaml
    │   │       ├── config_epoch_0344_validation.json
    │   │       ├── config_epoch_0345.yaml
    │   │       ├── config_epoch_0345_validation.json
    │   │       ├── config_epoch_0346.yaml
    │   │       ├── config_epoch_0346_validation.json
    │   │       ├── config_epoch_0347.yaml
    │   │       ├── config_epoch_0347_validation.json
    │   │       ├── config_epoch_0348.yaml
    │   │       ├── config_epoch_0348_validation.json
    │   │       ├── config_epoch_0349.yaml
    │   │       ├── config_epoch_0349_validation.json
    │   │       ├── config_epoch_0350.yaml
    │   │       ├── config_epoch_0350_validation.json
    │   │       ├── config_epoch_0351.yaml
    │   │       ├── config_epoch_0351_validation.json
    │   │       ├── config_epoch_0352.yaml
    │   │       ├── config_epoch_0352_validation.json
    │   │       ├── config_epoch_0353.yaml
    │   │       ├── config_epoch_0353_validation.json
    │   │       ├── config_epoch_0354.yaml
    │   │       ├── config_epoch_0354_validation.json
    │   │       ├── config_epoch_0355.yaml
    │   │       ├── config_epoch_0355_validation.json
    │   │       ├── config_epoch_0356.yaml
    │   │       ├── config_epoch_0356_validation.json
    │   │       ├── config_epoch_0357.yaml
    │   │       ├── config_epoch_0357_validation.json
    │   │       ├── config_epoch_0358.yaml
    │   │       ├── config_epoch_0358_validation.json
    │   │       ├── config_epoch_0359.yaml
    │   │       ├── config_epoch_0359_validation.json
    │   │       ├── config_epoch_0360.yaml
    │   │       ├── config_epoch_0360_validation.json
    │   │       ├── config_epoch_0361.yaml
    │   │       ├── config_epoch_0361_validation.json
    │   │       ├── config_epoch_0362.yaml
    │   │       ├── config_epoch_0362_validation.json
    │   │       ├── config_epoch_0363.yaml
    │   │       ├── config_epoch_0363_validation.json
    │   │       ├── config_epoch_0364.yaml
    │   │       ├── config_epoch_0364_validation.json
    │   │       ├── config_epoch_0365.yaml
    │   │       ├── config_epoch_0365_validation.json
    │   │       ├── config_epoch_0366.yaml
    │   │       ├── config_epoch_0366_validation.json
    │   │       ├── config_epoch_0367.yaml
    │   │       ├── config_epoch_0367_validation.json
    │   │       ├── config_epoch_0368.yaml
    │   │       ├── config_epoch_0368_validation.json
    │   │       ├── config_epoch_0369.yaml
    │   │       ├── config_epoch_0369_validation.json
    │   │       ├── config_epoch_0370.yaml
    │   │       ├── config_epoch_0370_validation.json
    │   │       ├── config_epoch_0371.yaml
    │   │       ├── config_epoch_0371_validation.json
    │   │       ├── config_epoch_0372.yaml
    │   │       ├── config_epoch_0372_validation.json
    │   │       ├── config_epoch_0373.yaml
    │   │       ├── config_epoch_0373_validation.json
    │   │       ├── config_epoch_0374.yaml
    │   │       ├── config_epoch_0374_validation.json
    │   │       ├── config_epoch_0375.yaml
    │   │       ├── config_epoch_0375_validation.json
    │   │       ├── config_epoch_0376.yaml
    │   │       ├── config_epoch_0376_validation.json
    │   │       ├── config_epoch_0377.yaml
    │   │       ├── config_epoch_0377_validation.json
    │   │       ├── config_epoch_0378.yaml
    │   │       ├── config_epoch_0378_validation.json
    │   │       ├── config_epoch_0379.yaml
    │   │       ├── config_epoch_0379_validation.json
    │   │       ├── config_epoch_0380.yaml
    │   │       ├── config_epoch_0380_validation.json
    │   │       ├── config_epoch_0381.yaml
    │   │       ├── config_epoch_0381_validation.json
    │   │       ├── config_epoch_0382.yaml
    │   │       ├── config_epoch_0382_validation.json
    │   │       ├── config_epoch_0383.yaml
    │   │       ├── config_epoch_0383_validation.json
    │   │       ├── config_epoch_0384.yaml
    │   │       ├── config_epoch_0384_validation.json
    │   │       ├── config_epoch_0385.yaml
    │   │       ├── config_epoch_0385_validation.json
    │   │       ├── config_epoch_0386.yaml
    │   │       ├── config_epoch_0386_validation.json
    │   │       ├── config_epoch_0387.yaml
    │   │       ├── config_epoch_0387_validation.json
    │   │       ├── config_epoch_0388.yaml
    │   │       ├── config_epoch_0388_validation.json
    │   │       ├── config_epoch_0389.yaml
    │   │       ├── config_epoch_0389_validation.json
    │   │       ├── config_epoch_0390.yaml
    │   │       ├── config_epoch_0390_validation.json
    │   │       ├── config_epoch_0391.yaml
    │   │       ├── config_epoch_0391_validation.json
    │   │       ├── config_epoch_0392.yaml
    │   │       ├── config_epoch_0392_validation.json
    │   │       ├── config_epoch_0393.yaml
    │   │       ├── config_epoch_0393_validation.json
    │   │       ├── config_epoch_0394.yaml
    │   │       ├── config_epoch_0394_validation.json
    │   │       ├── config_epoch_0395.yaml
    │   │       ├── config_epoch_0395_validation.json
    │   │       ├── config_epoch_0396.yaml
    │   │       ├── config_epoch_0396_validation.json
    │   │       ├── config_epoch_0397.yaml
    │   │       ├── config_epoch_0397_validation.json
    │   │       ├── config_epoch_0398.yaml
    │   │       ├── config_epoch_0398_validation.json
    │   │       ├── config_epoch_0399.yaml
    │   │       ├── config_epoch_0399_validation.json
    │   │       ├── config_epoch_0400.yaml
    │   │       ├── config_epoch_0400_validation.json
    │   │       ├── config_epoch_0401.yaml
    │   │       ├── config_epoch_0401_validation.json
    │   │       ├── config_epoch_0402.yaml
    │   │       ├── config_epoch_0402_validation.json
    │   │       ├── config_epoch_0403.yaml
    │   │       ├── config_epoch_0403_validation.json
    │   │       ├── config_epoch_0404.yaml
    │   │       ├── config_epoch_0404_validation.json
    │   │       ├── config_epoch_0405.yaml
    │   │       ├── config_epoch_0405_validation.json
    │   │       ├── config_epoch_0406.yaml
    │   │       ├── config_epoch_0406_validation.json
    │   │       ├── config_epoch_0407.yaml
    │   │       ├── config_epoch_0407_validation.json
    │   │       ├── config_epoch_0408.yaml
    │   │       ├── config_epoch_0408_validation.json
    │   │       ├── config_epoch_0409.yaml
    │   │       ├── config_epoch_0409_validation.json
    │   │       ├── config_epoch_0410.yaml
    │   │       ├── config_epoch_0410_validation.json
    │   │       ├── config_epoch_0411.yaml
    │   │       ├── config_epoch_0411_validation.json
    │   │       ├── config_epoch_0412.yaml
    │   │       ├── config_epoch_0412_validation.json
    │   │       ├── config_epoch_0413.yaml
    │   │       ├── config_epoch_0413_validation.json
    │   │       ├── config_epoch_0414.yaml
    │   │       ├── config_epoch_0414_validation.json
    │   │       ├── config_epoch_0415.yaml
    │   │       ├── config_epoch_0415_validation.json
    │   │       ├── config_epoch_0416.yaml
    │   │       ├── config_epoch_0416_validation.json
    │   │       ├── config_epoch_0417.yaml
    │   │       ├── config_epoch_0417_validation.json
    │   │       ├── config_epoch_0418.yaml
    │   │       ├── config_epoch_0418_validation.json
    │   │       ├── config_epoch_0419.yaml
    │   │       ├── config_epoch_0419_validation.json
    │   │       ├── config_epoch_0420.yaml
    │   │       ├── config_epoch_0420_validation.json
    │   │       ├── config_epoch_0421.yaml
    │   │       ├── config_epoch_0421_validation.json
    │   │       ├── config_epoch_0422.yaml
    │   │       ├── config_epoch_0422_validation.json
    │   │       ├── config_epoch_0423.yaml
    │   │       ├── config_epoch_0423_validation.json
    │   │       ├── config_epoch_0424.yaml
    │   │       ├── config_epoch_0424_validation.json
    │   │       ├── config_epoch_0425.yaml
    │   │       ├── config_epoch_0425_validation.json
    │   │       ├── config_epoch_0426.yaml
    │   │       ├── config_epoch_0426_validation.json
    │   │       ├── config_epoch_0427.yaml
    │   │       ├── config_epoch_0427_validation.json
    │   │       ├── config_epoch_0428.yaml
    │   │       ├── config_epoch_0428_validation.json
    │   │       ├── config_epoch_0429.yaml
    │   │       ├── config_epoch_0429_validation.json
    │   │       ├── config_epoch_0430.yaml
    │   │       ├── config_epoch_0430_validation.json
    │   │       ├── config_epoch_0431.yaml
    │   │       ├── config_epoch_0431_validation.json
    │   │       ├── config_epoch_0432.yaml
    │   │       ├── config_epoch_0432_validation.json
    │   │       ├── config_epoch_0433.yaml
    │   │       ├── config_epoch_0433_validation.json
    │   │       ├── config_epoch_0434.yaml
    │   │       ├── config_epoch_0434_validation.json
    │   │       ├── config_epoch_0435.yaml
    │   │       ├── config_epoch_0435_validation.json
    │   │       ├── config_epoch_0436.yaml
    │   │       ├── config_epoch_0436_validation.json
    │   │       ├── config_epoch_0437.yaml
    │   │       ├── config_epoch_0437_validation.json
    │   │       ├── config_epoch_0438.yaml
    │   │       ├── config_epoch_0438_validation.json
    │   │       ├── config_epoch_0439.yaml
    │   │       ├── config_epoch_0439_validation.json
    │   │       ├── config_epoch_0440.yaml
    │   │       ├── config_epoch_0440_validation.json
    │   │       ├── config_epoch_0441.yaml
    │   │       ├── config_epoch_0441_validation.json
    │   │       ├── config_epoch_0442.yaml
    │   │       ├── config_epoch_0442_validation.json
    │   │       ├── config_epoch_0443.yaml
    │   │       ├── config_epoch_0443_validation.json
    │   │       ├── config_epoch_0444.yaml
    │   │       ├── config_epoch_0444_validation.json
    │   │       ├── config_epoch_0445.yaml
    │   │       ├── config_epoch_0445_validation.json
    │   │       ├── config_epoch_0446.yaml
    │   │       ├── config_epoch_0446_validation.json
    │   │       ├── config_epoch_0447.yaml
    │   │       ├── config_epoch_0447_validation.json
    │   │       ├── config_epoch_0448.yaml
    │   │       ├── config_epoch_0448_validation.json
    │   │       ├── config_epoch_0449.yaml
    │   │       ├── config_epoch_0449_validation.json
    │   │       ├── config_epoch_0450.yaml
    │   │       ├── config_epoch_0450_validation.json
    │   │       ├── config_epoch_0451.yaml
    │   │       ├── config_epoch_0451_validation.json
    │   │       ├── config_epoch_0452.yaml
    │   │       ├── config_epoch_0452_validation.json
    │   │       ├── config_epoch_0453.yaml
    │   │       ├── config_epoch_0453_validation.json
    │   │       ├── config_epoch_0454.yaml
    │   │       ├── config_epoch_0454_validation.json
    │   │       ├── config_epoch_0455.yaml
    │   │       ├── config_epoch_0455_validation.json
    │   │       ├── config_epoch_0456.yaml
    │   │       ├── config_epoch_0456_validation.json
    │   │       ├── config_epoch_0457.yaml
    │   │       ├── config_epoch_0457_validation.json
    │   │       ├── config_epoch_0458.yaml
    │   │       ├── config_epoch_0458_validation.json
    │   │       ├── config_epoch_0459.yaml
    │   │       ├── config_epoch_0459_validation.json
    │   │       ├── config_epoch_0460.yaml
    │   │       ├── config_epoch_0460_validation.json
    │   │       ├── config_epoch_0461.yaml
    │   │       ├── config_epoch_0461_validation.json
    │   │       ├── config_epoch_0462.yaml
    │   │       ├── config_epoch_0462_validation.json
    │   │       ├── config_epoch_0463.yaml
    │   │       ├── config_epoch_0463_validation.json
    │   │       ├── config_epoch_0464.yaml
    │   │       ├── config_epoch_0464_validation.json
    │   │       ├── config_epoch_0465.yaml
    │   │       ├── config_epoch_0465_validation.json
    │   │       ├── config_epoch_0466.yaml
    │   │       ├── config_epoch_0466_validation.json
    │   │       ├── config_epoch_0467.yaml
    │   │       ├── config_epoch_0467_validation.json
    │   │       ├── config_epoch_0468.yaml
    │   │       ├── config_epoch_0468_validation.json
    │   │       ├── config_epoch_0469.yaml
    │   │       ├── config_epoch_0469_validation.json
    │   │       ├── config_epoch_0470.yaml
    │   │       ├── config_epoch_0470_validation.json
    │   │       ├── config_epoch_0471.yaml
    │   │       ├── config_epoch_0471_validation.json
    │   │       ├── config_epoch_0472.yaml
    │   │       ├── config_epoch_0472_validation.json
    │   │       ├── config_epoch_0473.yaml
    │   │       ├── config_epoch_0473_validation.json
    │   │       ├── config_epoch_0474.yaml
    │   │       ├── config_epoch_0474_validation.json
    │   │       ├── config_epoch_0475.yaml
    │   │       ├── config_epoch_0475_validation.json
    │   │       ├── config_epoch_0476.yaml
    │   │       ├── config_epoch_0476_validation.json
    │   │       ├── config_epoch_0477.yaml
    │   │       ├── config_epoch_0477_validation.json
    │   │       ├── config_epoch_0478.yaml
    │   │       ├── config_epoch_0478_validation.json
    │   │       ├── config_epoch_0479.yaml
    │   │       ├── config_epoch_0479_validation.json
    │   │       ├── config_epoch_0480.yaml
    │   │       ├── config_epoch_0480_validation.json
    │   │       ├── config_epoch_0481.yaml
    │   │       ├── config_epoch_0481_validation.json
    │   │       ├── config_epoch_0482.yaml
    │   │       ├── config_epoch_0482_validation.json
    │   │       ├── config_epoch_0483.yaml
    │   │       ├── config_epoch_0483_validation.json
    │   │       ├── config_epoch_0484.yaml
    │   │       ├── config_epoch_0484_validation.json
    │   │       ├── config_epoch_0485.yaml
    │   │       ├── config_epoch_0485_validation.json
    │   │       ├── config_epoch_0486.yaml
    │   │       ├── config_epoch_0486_validation.json
    │   │       ├── config_epoch_0487.yaml
    │   │       ├── config_epoch_0487_validation.json
    │   │       ├── config_epoch_0488.yaml
    │   │       ├── config_epoch_0488_validation.json
    │   │       ├── config_epoch_0489.yaml
    │   │       ├── config_epoch_0489_validation.json
    │   │       ├── config_epoch_0490.yaml
    │   │       ├── config_epoch_0490_validation.json
    │   │       ├── config_epoch_0491.yaml
    │   │       ├── config_epoch_0491_validation.json
    │   │       ├── config_epoch_0492.yaml
    │   │       ├── config_epoch_0492_validation.json
    │   │       ├── config_epoch_0493.yaml
    │   │       ├── config_epoch_0493_validation.json
    │   │       ├── config_epoch_0494.yaml
    │   │       ├── config_epoch_0494_validation.json
    │   │       ├── config_epoch_0495.yaml
    │   │       ├── config_epoch_0495_validation.json
    │   │       ├── config_epoch_0496.yaml
    │   │       ├── config_epoch_0496_validation.json
    │   │       ├── config_epoch_0497.yaml
    │   │       ├── config_epoch_0497_validation.json
    │   │       ├── config_epoch_0498.yaml
    │   │       ├── config_epoch_0498_validation.json
    │   │       ├── config_epoch_0499.yaml
    │   │       ├── config_epoch_0499_validation.json
    │   │       ├── config_epoch_0500.yaml
    │   │       ├── config_epoch_0500_validation.json
    │   │       ├── training_config.yaml
    │   │       └── training_config_validation.json
    │   ├── demo_experiment/
    │   │   ├── demo_experiment/
    │   │   │   ├── configs/
    │   │   │   ├── logs/
    │   │   │   ├── metrics/
    │   │   │   ├── models/
    │   │   │   ├── predictions/
    │   │   │   ├── reports/
    │   │   │   ├── visualizations/
    │   │   │   │   ├── learning_rate_analysis_learning_rate_analysis
    │   │   │   │   └── parameter_distributions_parameter_distributions
    │   │   │   └── metadata.json
    │   │   ├── experiment_data/
    │   │   │   ├── metrics/
    │   │   │   │   ├── complete_summary.json
    │   │   │   │   └── metrics.jsonl
    │   │   │   ├── config.yaml
    │   │   │   └── model_best.pth
    │   │   └── visualizations/
    │   │       ├── learning_rate_analysis.html
    │   │       ├── learning_rate_analysis.png
    │   │       ├── parameter_distributions.html
    │   │       ├── parameter_distributions.png
    │   │       └── training_curves.html
    │   ├── demo_interactive/
    │   │   ├── 3d_confidence_map.html
    │   │   ├── 3d_confidence_map.pdf
    │   │   ├── 3d_confidence_map.png
    │   │   ├── error_analysis.html
    │   │   ├── error_analysis.pdf
    │   │   ├── error_analysis.png
    │   │   ├── prediction_grid.html
    │   │   ├── prediction_grid.pdf
    │   │   ├── prediction_grid.png
    │   │   ├── real_time_dashboard.html
    │   │   ├── real_time_dashboard.pdf
    │   │   ├── real_time_dashboard.png
    │   │   ├── template_prediction_grid.html
    │   │   ├── template_prediction_grid.pdf
    │   │   ├── template_prediction_grid.png
    │   │   ├── template_training_curves.html
    │   │   ├── template_training_curves.png
    │   │   ├── training_curves.html
    │   │   ├── training_curves.pdf
    │   │   └── training_curves.png
    │   ├── demo_prediction/
    │   │   ├── comparison_grid.png
    │   │   ├── confidence_map.png
    │   │   ├── custom_styled_grid.png
    │   │   ├── error_analysis.png
    │   │   ├── sample_image_0.jpg
    │   │   ├── sample_image_1.jpg
    │   │   ├── sample_image_2.jpg
    │   │   ├── sample_image_3.jpg
    │   │   ├── sample_image_4.jpg
    │   │   ├── sample_image_5.jpg
    │   │   ├── segmentation_overlay.png
    │   │   └── tabular_comparison.png
    │   ├── metrics/
    │   │   ├── complete_summary.json
    │   │   ├── summary.json
    │   │   └── validation_metrics.jsonl
    │   └── template_demo/
    │       ├── base_template_demo.png
    │       ├── customized_template_demo.png
    │       ├── prediction_template_demo.png
    │       └── training_template_demo.png
    ├── reports/
    ├── scripts/
    │   ├── __pycache__/
    │   ├── archive/
    │   │   ├── limpieza_scripts_summary.md
    │   │   └── README.md
    │   ├── debug/
    │   │   ├── __init__.py
    │   │   ├── artifact_diagnostics.py
    │   │   ├── artifact_fixer.py
    │   │   ├── checkpoint_validator.py
    │   │   ├── main.py
    │   │   ├── mass_git_restore.py
    │   │   ├── syntax_scanner.py
    │   │   └── utils.py
    │   ├── examples/
    │   │   ├── advanced_orchestration_demo.py
    │   │   ├── advanced_prediction_viz_demo.py
    │   │   ├── advanced_training_viz_demo.py
    │   │   ├── deployment_orchestration_example.py
    │   │   ├── factory_registry_integration.py
    │   │   ├── interactive_plotly_demo.py
    │   │   ├── prediction_analysis_demo.py
    │   │   ├── production_readiness_validation_example.py
    │   │   ├── template_system_demo.py
    │   │   ├── tensorboard_port_management_demo.py
    │   │   ├── validation_pipeline_demo.py
    │   │   └── validation_reporting_demo.py
    │   ├── experiments/
    │   │   ├── e2e/
    │   │   │   ├── modules/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── checkpointing.py
    │   │   │   │   ├── config.py
    │   │   │   │   ├── data.py
    │   │   │   │   ├── dataclasses.py
    │   │   │   │   ├── evaluation.py
    │   │   │   │   ├── setup.py
    │   │   │   │   ├── training.py
    │   │   │   │   └── utils.py
    │   │   │   ├── __init__.py
    │   │   │   ├── README.md
    │   │   │   └── test_pipeline_e2e.py
    │   │   ├── swinv2_hybrid/
    │   │   │   ├── __init__.py
    │   │   │   ├── README.md
    │   │   │   ├── run_swinv2_hybrid_experiment.py
    │   │   │   ├── swinv2_hybrid_analysis.py
    │   │   │   └── test_swinv2_hybrid_setup.py
    │   │   ├── tutorial_02/
    │   │   │   ├── tutorial_02_batch.ps1
    │   │   │   ├── tutorial_02_compare.py
    │   │   │   └── tutorial_02_visualize.py
    │   │   ├── automated_comparison.py
    │   │   ├── benchmark_aspp.py
    │   │   ├── debug_swin_params.py
    │   │   ├── hybrid_registry_demo.py
    │   │   ├── README.md
    │   │   └── registry_demo.py
    │   ├── monitoring/
    │   │   └── continuous_coverage.py
    │   ├── performance/
    │   │   ├── __pycache__/
    │   │   ├── __init__.py
    │   │   ├── base_executor.py
    │   │   ├── baseline_updater.py
    │   │   ├── cleanup_validator.py
    │   │   ├── health_checker.py
    │   │   ├── maintenance_manager.py
    │   │   └── utils.py
    │   ├── reports/
    │   │   ├── autofix_backups/
    │   │   ├── compare_model_structure.py
    │   │   ├── model_imports_autofix.py
    │   │   ├── model_imports_catalog.py
    │   │   ├── model_imports_cycles.py
    │   │   └── model_imports_validation.py
    │   ├── utils/
    │   │   ├── test_suite_refinement/
    │   │   │   ├── add_reproducibility_score.py
    │   │   │   ├── categorize_tests_status.py
    │   │   │   ├── generate_executive_report.py
    │   │   │   ├── generate_test_inventory.py
    │   │   │   ├── report_environment_issues.py
    │   │   │   ├── report_manual_intervention.py
    │   │   │   ├── report_slow_tests.py
    │   │   │   ├── run_coverage_report.py
    │   │   │   ├── tag_test_priority.py
    │   │   │   └── update_test_inventory_status.py
    │   │   ├── audit_rules_checklist.py
    │   │   ├── check_updates.py
    │   │   ├── clean_workspace.py
    │   │   ├── example_override.py
    │   │   ├── generate_project_tree.py
    │   │   ├── inventory_training_imports.py
    │   │   ├── model_summary.py
    │   │   ├── organize_reports.py
    │   │   ├── organize_reports_plan.md
    │   │   ├── reorganize_legacy_folders_plan.md
    │   │   ├── unet_diagram.py
    │   │   ├── validate-rule-references.py
    │   │   └── verify_setup.py
    │   ├── __init__.py
    │   ├── artifact_selection_example.py
    │   ├── benchmark_tests.py
    │   ├── check_test_files.py
    │   ├── coverage_check.sh
    │   ├── debug_artifacts.py
    │   ├── deployment_example.py
    │   ├── orchestration_example.py
    │   ├── packaging_example.py
    │   ├── performance_maintenance.py
    │   ├── predict_image.py
    │   ├── README.md
    │   ├── run_tests_phased.py
    │   ├── simple_final_report.py
    │   ├── simple_install_check.sh
    │   ├── tutorial_03_verification.py
    │   └── validate_test_quality.py
    ├── src/
    │   ├── __pycache__/
    │   ├── crackseg/
    │   │   ├── __pycache__/
    │   │   ├── data/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── collate.py
    │   │   │   ├── dataloader.py
    │   │   │   ├── dataloader.py.backup
    │   │   │   ├── dataset.py
    │   │   │   ├── dataset.py.backup
    │   │   │   ├── distributed.py
    │   │   │   ├── factory.py
    │   │   │   ├── memory.py
    │   │   │   ├── README.md
    │   │   │   ├── sampler.py
    │   │   │   ├── splitting.py
    │   │   │   ├── transforms.py
    │   │   │   └── validation.py
    │   │   ├── evaluation/
    │   │   │   ├── __pycache__/
    │   │   │   ├── cli/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── prediction_cli.py
    │   │   │   ├── core/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── analyzer.py
    │   │   │   │   ├── image_processor.py
    │   │   │   │   └── model_loader.py
    │   │   │   ├── metrics/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── batch_processor.py
    │   │   │   │   └── calculator.py
    │   │   │   ├── visualization/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── interactive_plotly/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── core.py
    │   │   │   │   │   ├── export_handlers.py
    │   │   │   │   │   └── metadata_handlers.py
    │   │   │   │   ├── templates/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── base_template.py
    │   │   │   │   │   ├── prediction_template.py
    │   │   │   │   │   └── training_template.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── advanced_prediction_viz.py
    │   │   │   │   ├── advanced_training_viz.py
    │   │   │   │   ├── architecture.md
    │   │   │   │   ├── experiment_viz.py
    │   │   │   │   ├── learning_rate_analysis.py
    │   │   │   │   ├── parameter_analysis.py
    │   │   │   │   ├── prediction_viz.py
    │   │   │   │   └── training_curves.py
    │   │   │   ├── __init__.py
    │   │   │   ├── __main__.py
    │   │   │   ├── core.py
    │   │   │   ├── data.py
    │   │   │   ├── ensemble.py
    │   │   │   ├── loading.py
    │   │   │   ├── README.md
    │   │   │   ├── results.py
    │   │   │   └── setup.py
    │   │   ├── integration/
    │   │   ├── model/
    │   │   │   ├── __pycache__/
    │   │   │   ├── architectures/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── cnn_convlstm_unet.py
    │   │   │   │   ├── registry.py
    │   │   │   │   ├── simple_unet.py
    │   │   │   │   ├── swinv2_cnn_aspp_unet.py
    │   │   │   │   └── unet.py
    │   │   │   ├── base/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── abstract.py
    │   │   │   │   └── abstract.py.backup
    │   │   │   ├── bottleneck/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── cnn_bottleneck.py
    │   │   │   ├── common/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── visualization/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── matplotlib/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── graphviz_renderer.py
    │   │   │   │   │   ├── main.py
    │   │   │   │   │   └── matplotlib_renderer.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── spatial_utils.py
    │   │   │   │   ├── utils.py
    │   │   │   │   └── utils.py.backup
    │   │   │   ├── components/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── aspp.py
    │   │   │   │   ├── attention_decorator.py
    │   │   │   │   ├── cbam.py
    │   │   │   │   ├── convlstm.py
    │   │   │   │   └── registry_support.py
    │   │   │   ├── config/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── core.py
    │   │   │   │   ├── factory.py
    │   │   │   │   ├── instantiation.py
    │   │   │   │   ├── schemas.py
    │   │   │   │   └── validation.py
    │   │   │   ├── core/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── unet.py
    │   │   │   ├── decoder/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── common/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   └── channel_utils.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── cnn_decoder.py
    │   │   │   ├── encoder/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── swin/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── config.py
    │   │   │   │   │   ├── core.py
    │   │   │   │   │   ├── initialization.py
    │   │   │   │   │   ├── preprocessing.py
    │   │   │   │   │   └── transfer_learning.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── cnn_encoder.py
    │   │   │   │   ├── feature_info_utils.py
    │   │   │   │   ├── swin_transformer_encoder.py
    │   │   │   │   └── swin_v2_adapter.py
    │   │   │   ├── factory/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── config.py
    │   │   │   │   ├── config_schema.py
    │   │   │   │   ├── factory.py
    │   │   │   │   ├── factory_utils.py
    │   │   │   │   ├── hybrid_registry.py
    │   │   │   │   ├── registry.py
    │   │   │   │   └── registry_setup.py
    │   │   │   ├── __init__.py
    │   │   │   └── README.md
    │   │   ├── outputs/
    │   │   │   ├── experiments/
    │   │   │   │   ├── 20250602-041102-basic_verification/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   ├── error_log.txt
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250602-041159-basic_verification/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   ├── error_log.txt
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250602-041325-basic_verification/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   ├── error_log.txt
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250602-041432-basic_verification/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── configurations/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250603-010344-basic_verification/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── configurations/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250722-235914-default/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   ├── error_log.txt
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250723-000037-default/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   ├── error_log.txt
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250723-001601-default/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   └── results/
    │   │   │   │   ├── 20250723-002338-default/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   ├── error_log.txt
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250723-002515-default/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   ├── error_log.txt
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250723-002758-default/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   ├── error_log.txt
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250723-002957-default/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   ├── error_log.txt
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250723-003357-default/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   ├── error_log.txt
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250723-003548-default/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── configurations/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   ├── error_log.txt
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250723-003640-default/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── configurations/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   ├── error_log.txt
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250723-003740-default/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250723-003741-default/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── configurations/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   ├── error_log.txt
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250723-003829-default/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── configurations/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250723-005521-default/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── configurations/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250723-005704-default/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── configurations/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250723-010032-default/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── configurations/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250723-172135-default/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   ├── error_log.txt
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250723-172227-default/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   ├── error_log.txt
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250723-172329-default/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── configurations/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   └── 20250723-173339-default/
    │   │   │   │       ├── checkpoints/
    │   │   │   │       ├── configurations/
    │   │   │   │       ├── logs/
    │   │   │   │       ├── metrics/
    │   │   │   │       ├── results/
    │   │   │   │       ├── config.json
    │   │   │   │       └── experiment_info.json
    │   │   │   ├── shared/
    │   │   │   └── experiment_registry.json
    │   │   ├── reporting/
    │   │   │   ├── __pycache__/
    │   │   │   ├── comparison/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── engine.py
    │   │   │   ├── figures/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── publication_figure_generator.py
    │   │   │   ├── performance/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── analyzer.py
    │   │   │   │   ├── anomaly_detector.py
    │   │   │   │   ├── metric_evaluator.py
    │   │   │   │   ├── recommendation_engine.py
    │   │   │   │   └── training_analyzer.py
    │   │   │   ├── recommendations/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── analyzers/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── hyperparameters.py
    │   │   │   │   │   ├── performance.py
    │   │   │   │   │   └── training_patterns.py
    │   │   │   │   ├── identifiers/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── architecture.py
    │   │   │   │   │   └── opportunities.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── engine.py
    │   │   │   │   └── thresholds.py
    │   │   │   ├── templates/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── html_templates.py
    │   │   │   │   ├── latex_templates.py
    │   │   │   │   ├── markdown_templates.py
    │   │   │   │   └── template_manager.py
    │   │   │   ├── __init__.py
    │   │   │   ├── config.py
    │   │   │   ├── core.py
    │   │   │   ├── data_loader.py
    │   │   │   └── interfaces.py
    │   │   ├── training/
    │   │   │   ├── __pycache__/
    │   │   │   ├── losses/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── combinators/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── base_combinator.py
    │   │   │   │   │   ├── enhanced_product.py
    │   │   │   │   │   ├── enhanced_weighted_sum.py
    │   │   │   │   │   ├── product.py
    │   │   │   │   │   └── weighted_sum.py
    │   │   │   │   ├── factory/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── config_parser.py
    │   │   │   │   │   ├── config_validator.py
    │   │   │   │   │   └── recursive_factory.py
    │   │   │   │   ├── interfaces/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   └── loss_interface.py
    │   │   │   │   ├── registry/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── clean_registry.py
    │   │   │   │   │   ├── enhanced_registry.py
    │   │   │   │   │   └── setup_losses.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── base_loss.py
    │   │   │   │   ├── bce_dice_loss.py
    │   │   │   │   ├── bce_loss.py
    │   │   │   │   ├── combined_loss.py
    │   │   │   │   ├── dice_loss.py
    │   │   │   │   ├── focal_dice_loss.py
    │   │   │   │   ├── focal_loss.py
    │   │   │   │   ├── loss_registry_setup.py
    │   │   │   │   ├── recursive_factory.py
    │   │   │   │   └── smooth_l1_loss.py
    │   │   │   ├── optimizers/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── custom_adam.py
    │   │   │   │   └── registry.py
    │   │   │   ├── __init__.py
    │   │   │   ├── batch_processing.py
    │   │   │   ├── config_validation.py
    │   │   │   ├── factory.py
    │   │   │   ├── metrics.py
    │   │   │   ├── README.md
    │   │   │   ├── trainer.py
    │   │   │   └── trainer.py.backup
    │   │   ├── utils/
    │   │   │   ├── __pycache__/
    │   │   │   ├── artifact_manager/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── core.py
    │   │   │   │   ├── metadata.py
    │   │   │   │   ├── storage.py
    │   │   │   │   ├── validation.py
    │   │   │   │   └── versioning.py
    │   │   │   ├── checkpointing/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── core.py
    │   │   │   │   ├── helpers.py
    │   │   │   │   └── setup.py
    │   │   │   ├── config/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── env.py
    │   │   │   │   ├── init.py
    │   │   │   │   ├── override.py
    │   │   │   │   ├── schema.py
    │   │   │   │   ├── standardized_storage.py
    │   │   │   │   └── validation.py
    │   │   │   ├── core/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── device.py
    │   │   │   │   ├── exceptions.py
    │   │   │   │   ├── paths.py
    │   │   │   │   └── seeds.py
    │   │   │   ├── deployment/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── packaging/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── config.py
    │   │   │   │   │   ├── containerization.py
    │   │   │   │   │   ├── core.py
    │   │   │   │   │   ├── dependencies.py
    │   │   │   │   │   ├── docker_compose.py
    │   │   │   │   │   ├── file_generators.py
    │   │   │   │   │   ├── helm.py
    │   │   │   │   │   ├── kubernetes.py
    │   │   │   │   │   ├── manifests.py
    │   │   │   │   │   ├── metrics.py
    │   │   │   │   │   └── security.py
    │   │   │   │   ├── templates/
    │   │   │   │   │   └── validation_report.md.j2
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── artifact_optimizer.py
    │   │   │   │   ├── artifact_selector.py
    │   │   │   │   ├── config.py
    │   │   │   │   ├── deployment_manager.py
    │   │   │   │   ├── environment_configurator.py
    │   │   │   │   ├── monitoring_system.py
    │   │   │   │   ├── orchestration.py
    │   │   │   │   ├── production_readiness_validator.py
    │   │   │   │   ├── validation_pipeline.py
    │   │   │   │   └── validation_reporter.py
    │   │   │   ├── experiment/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── tracker/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── tracker_artifacts.py
    │   │   │   │   │   ├── tracker_config.py
    │   │   │   │   │   ├── tracker_git.py
    │   │   │   │   │   └── tracker_lifecycle.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── experiment.py
    │   │   │   │   ├── manager.py
    │   │   │   │   ├── metadata.py
    │   │   │   │   └── tracker.py
    │   │   │   ├── factory/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── cache.py
    │   │   │   │   └── factory.py
    │   │   │   ├── integrity/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── artifact_verifier.py
    │   │   │   │   ├── checkpoint_verifier.py
    │   │   │   │   ├── config_verifier.py
    │   │   │   │   ├── core.py
    │   │   │   │   └── experiment_verifier.py
    │   │   │   ├── logging/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── base.py
    │   │   │   │   ├── experiment.py
    │   │   │   │   ├── metrics_manager.py
    │   │   │   │   ├── setup.py
    │   │   │   │   └── training.py
    │   │   │   ├── monitoring/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── alert_types.py
    │   │   │   │   ├── alerting_system.py
    │   │   │   │   ├── callbacks.py
    │   │   │   │   ├── coverage_monitor.py
    │   │   │   │   ├── exceptions.py
    │   │   │   │   ├── gpu_callbacks.py
    │   │   │   │   ├── manager.py
    │   │   │   │   ├── resource_monitor.py
    │   │   │   │   ├── resource_snapshot.py
    │   │   │   │   ├── retention.py
    │   │   │   │   ├── system_callbacks.py
    │   │   │   │   ├── threshold_checker.py
    │   │   │   │   └── threshold_config.py
    │   │   │   ├── traceability/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── access_control.py
    │   │   │   │   ├── compliance.py
    │   │   │   │   ├── entities.py
    │   │   │   │   ├── enums.py
    │   │   │   │   ├── integration_manager.py
    │   │   │   │   ├── lineage_manager.py
    │   │   │   │   ├── metadata_manager.py
    │   │   │   │   ├── models.py
    │   │   │   │   ├── queries.py
    │   │   │   │   ├── query_interface.py
    │   │   │   │   └── storage.py
    │   │   │   ├── training/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── amp_utils.py
    │   │   │   │   ├── early_stopping.py
    │   │   │   │   ├── early_stopping_setup.py
    │   │   │   │   └── scheduler_helper.py
    │   │   │   ├── visualization/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── plots.py
    │   │   │   ├── __init__.py
    │   │   │   ├── artifact_manager.py
    │   │   │   ├── component_cache.py
    │   │   │   ├── exceptions.py
    │   │   │   ├── README.md
    │   │   │   └── torchvision_compat.py
    │   │   ├── __init__.py
    │   │   ├── __main__.py
    │   │   ├── dataclasses.py
    │   │   ├── evaluate.py
    │   │   └── README.md
    │   ├── crackseg.egg-info/
    │   └── main.py
    ├── tests/
    │   ├── __pycache__/
    │   ├── docker/
    │   ├── e2e/
    │   │   ├── __pycache__/
    │   │   ├── capture/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── screenshot.py
    │   │   │   ├── storage.py
    │   │   │   ├── video.py
    │   │   │   └── visual_regression.py
    │   │   ├── cleanup/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── audit_trail.py
    │   │   │   ├── ci_integration.py
    │   │   │   ├── cleanup_manager.py
    │   │   │   ├── environment_readiness.py
    │   │   │   ├── post_cleanup_validator.py
    │   │   │   ├── resource_cleanup.py
    │   │   │   ├── validation_reporter.py
    │   │   │   └── validation_system.py
    │   │   ├── config/
    │   │   │   ├── __pycache__/
    │   │   │   ├── viewport_config/
    │   │   │   │   ├── devices/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── core.py
    │   │   │   │   │   ├── defaults.py
    │   │   │   │   │   └── factories.py
    │   │   │   │   ├── matrix/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── core.py
    │   │   │   │   │   └── presets.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── core.py
    │   │   │   ├── __init__.py
    │   │   │   ├── browser_capabilities.py
    │   │   │   ├── browser_config_manager.py
    │   │   │   ├── browser_matrix_config.py
    │   │   │   ├── cross_browser_test.py
    │   │   │   ├── execution_strategies.py
    │   │   │   ├── parallel_execution_config.py
    │   │   │   ├── parallel_performance_integration.py
    │   │   │   ├── performance_thresholds.py
    │   │   │   ├── pytest_markers.py
    │   │   │   ├── resource_manager.py
    │   │   │   ├── test_parallel_framework_validation.py
    │   │   │   └── threshold_validator.py
    │   │   ├── data/
    │   │   │   ├── factories/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── base.py
    │   │   │   │   ├── config_factory.py
    │   │   │   │   ├── coordinator.py
    │   │   │   │   ├── image_factory.py
    │   │   │   │   └── model_factory.py
    │   │   │   ├── provisioning/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── core.py
    │   │   │   │   ├── database.py
    │   │   │   │   └── suites.py
    │   │   │   ├── __init__.py
    │   │   │   └── isolation.py
    │   │   ├── drivers/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── config.py
    │   │   │   ├── driver_factory.py
    │   │   │   ├── driver_manager.py
    │   │   │   └── exceptions.py
    │   │   ├── helpers/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── api_integration.py
    │   │   │   ├── performance_monitoring.py
    │   │   │   ├── setup_teardown.py
    │   │   │   └── test_coordination.py
    │   │   ├── maintenance/
    │   │   │   ├── __init__.py
    │   │   │   ├── config.py
    │   │   │   ├── core.py
    │   │   │   ├── health_monitor.py
    │   │   │   └── models.py
    │   │   ├── mixins/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── capture_mixin.py
    │   │   │   ├── logging_mixin.py
    │   │   │   ├── performance_mixin.py
    │   │   │   ├── retry_mixin.py
    │   │   │   └── streamlit_mixin.py
    │   │   ├── pages/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── advanced_config_page.py
    │   │   │   ├── architecture_page.py
    │   │   │   ├── base_page.py
    │   │   │   ├── config_page.py
    │   │   │   ├── locators.py
    │   │   │   ├── results_page.py
    │   │   │   └── train_page.py
    │   │   ├── performance/
    │   │   │   ├── __pycache__/
    │   │   │   ├── reporting/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── analysis.py
    │   │   │   │   ├── comparison_charts.py
    │   │   │   │   ├── config.py
    │   │   │   │   ├── core.py
    │   │   │   │   ├── data_processor.py
    │   │   │   │   ├── factory_functions.py
    │   │   │   │   ├── formats.py
    │   │   │   │   ├── historical_manager.py
    │   │   │   │   ├── html_formatter.py
    │   │   │   │   ├── insights_generator.py
    │   │   │   │   ├── json_formatter.py
    │   │   │   │   ├── metric_extractor.py
    │   │   │   │   ├── pdf_formatter.py
    │   │   │   │   ├── regression_analyzer.py
    │   │   │   │   ├── summary_charts.py
    │   │   │   │   ├── templates.py
    │   │   │   │   ├── trend_analyzer.py
    │   │   │   │   ├── trend_charts.py
    │   │   │   │   └── visualizations.py
    │   │   │   ├── __init__.py
    │   │   │   ├── benchmark_runner.py
    │   │   │   ├── benchmark_suite.py
    │   │   │   ├── ci_integration.py
    │   │   │   ├── endurance_test.py
    │   │   │   ├── load_test.py
    │   │   │   ├── metrics_collector.py
    │   │   │   ├── regression_alerting_system.py
    │   │   │   └── stress_test.py
    │   │   ├── reporting/
    │   │   │   ├── analysis/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── failure_analyzer.py
    │   │   │   │   └── trend_analyzer.py
    │   │   │   ├── __init__.py
    │   │   │   ├── config.py
    │   │   │   ├── core.py
    │   │   │   ├── exporters.py
    │   │   │   ├── generator.py
    │   │   │   └── models.py
    │   │   ├── session/
    │   │   │   ├── __init__.py
    │   │   │   ├── cookie_manager.py
    │   │   │   ├── mixins.py
    │   │   │   ├── state_manager.py
    │   │   │   ├── storage_manager.py
    │   │   │   └── streamlit_session.py
    │   │   ├── test_data/
    │   │   │   └── invalid_configs/
    │   │   │       ├── conflicting_dependencies.yaml
    │   │   │       ├── invalid_syntax.yaml
    │   │   │       └── missing_required_fields.yaml
    │   │   ├── tests/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── test_capture_demonstration.py
    │   │   │   ├── test_cross_browser_compatibility.py
    │   │   │   ├── test_edge_cases.py
    │   │   │   ├── test_edge_cases_boundary.py
    │   │   │   ├── test_edge_cases_concurrent.py
    │   │   │   ├── test_edge_cases_corruption.py
    │   │   │   ├── test_edge_cases_interactions.py
    │   │   │   ├── test_edge_cases_performance.py
    │   │   │   ├── test_edge_cases_resources.py
    │   │   │   ├── test_error_scenarios.py
    │   │   │   ├── test_happy_path.py
    │   │   │   └── test_performance_integration.py
    │   │   ├── utils/
    │   │   │   ├── __pycache__/
    │   │   │   ├── responsive/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── layout.py
    │   │   │   │   ├── testing.py
    │   │   │   │   ├── touch.py
    │   │   │   │   └── viewport.py
    │   │   │   ├── __init__.py
    │   │   │   ├── browser_validation.py
    │   │   │   ├── config.py
    │   │   │   ├── data.py
    │   │   │   ├── debugging.py
    │   │   │   ├── element.py
    │   │   │   ├── file.py
    │   │   │   ├── README_TestEnvironmentManager.md
    │   │   │   ├── streamlit.py
    │   │   │   ├── string.py
    │   │   │   ├── test_environment_fixtures.py
    │   │   │   ├── test_environment_manager.py
    │   │   │   └── time.py
    │   │   ├── waits/
    │   │   │   ├── __init__.py
    │   │   │   ├── conditions.py
    │   │   │   └── strategies.py
    │   │   ├── __init__.py
    │   │   ├── base_test.py
    │   │   ├── conftest.py
    │   │   ├── README.md
    │   │   ├── test_cleanup_integration.py
    │   │   ├── test_driver_integration.py
    │   │   ├── test_environment_setup_demo.py
    │   │   ├── test_fixture_usage_example.py
    │   │   ├── test_streamlit_basic.py
    │   │   └── test_workflow_regression_4_4.py
    │   ├── examples/
    │   │   ├── __pycache__/
    │   │   ├── enhanced_gui_testing_demo.py
    │   │   └── visual_regression_demo.py
    │   ├── fixtures/
    │   │   └── mocks/
    │   │       ├── experiment_manager/
    │   │       │   ├── mock.experiment_manager.experiment_manager.experiment_dir/
    │   │       │   │   ├── 2254896334992/
    │   │       │   │   ├── 2254896338784/
    │   │       │   │   ├── 2254920579888/
    │   │       │   │   ├── 2254920693232/
    │   │       │   │   ├── 2254921007024/
    │   │       │   │   ├── 2254921628848/
    │   │       │   │   ├── 2254921775968/
    │   │       │   │   ├── 2254921841792/
    │   │       │   │   ├── 2254921913536/
    │   │       │   │   ├── 2254922611312/
    │   │       │   │   ├── 2256368976960/
    │   │       │   │   ├── 2399501198368/
    │   │       │   │   ├── 2399501899136/
    │   │       │   │   ├── 2399501909216/
    │   │       │   │   ├── 2399502072480/
    │   │       │   │   ├── 2399502175984/
    │   │       │   │   ├── 2399502348320/
    │   │       │   │   ├── 2399502532384/
    │   │       │   │   ├── 2399503130960/
    │   │       │   │   ├── 2399503481312/
    │   │       │   │   ├── 2399503869728/
    │   │       │   │   ├── 2789862072848/
    │   │       │   │   ├── 2789862077936/
    │   │       │   │   ├── 2789862548272/
    │   │       │   │   ├── 2789864181296/
    │   │       │   │   ├── 2789864185664/
    │   │       │   │   ├── 2789864403744/
    │   │       │   │   ├── 2789864459584/
    │   │       │   │   ├── 2789864648544/
    │   │       │   │   ├── 2789864714032/
    │   │       │   │   └── 2789865142272/
    │   │       │   └── mock.experiment_manager.experiment_manager.experiment_dir.__truediv__()/
    │   │       │       ├── 2254896338256/
    │   │       │       ├── 2254896346752/
    │   │       │       ├── 2254920643024/
    │   │       │       ├── 2254920935152/
    │   │       │       ├── 2254921165824/
    │   │       │       ├── 2254921294208/
    │   │       │       ├── 2254921768912/
    │   │       │       ├── 2254922315728/
    │   │       │       ├── 2254922618864/
    │   │       │       ├── 2256368910512/
    │   │       │       └── 2256371459392/
    │   │       └── README.md
    │   ├── gui/
    │   │   ├── __pycache__/
    │   │   ├── test_auto_save.py
    │   │   ├── test_confirmation_dialog.py
    │   │   ├── test_device_selector.py
    │   │   ├── test_error_state.py
    │   │   ├── test_loading_spinner.py
    │   │   ├── test_performance_optimization.py
    │   │   ├── test_progress_bar.py
    │   │   └── test_results_gallery_component.py
    │   ├── integration/
    │   │   ├── __pycache__/
    │   │   ├── config/
    │   │   │   ├── __pycache__/
    │   │   │   └── test_hydra_config.py
    │   │   ├── data/
    │   │   │   ├── __pycache__/
    │   │   │   ├── test_data_loading_pipeline.py
    │   │   │   └── test_data_pipeline.py
    │   │   ├── end_to_end/
    │   │   ├── evaluation/
    │   │   │   ├── __pycache__/
    │   │   │   └── test_evaluation_pipeline.py
    │   │   ├── gui/
    │   │   │   ├── __pycache__/
    │   │   │   ├── automation/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── reporting/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── analysis/
    │   │   │   │   │   ├── data_collectors/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── analysis_engine.py
    │   │   │   │   │   ├── content_generators.py
    │   │   │   │   │   ├── csv_export.py
    │   │   │   │   │   ├── data_aggregation.py
    │   │   │   │   │   ├── export_manager.py
    │   │   │   │   │   ├── html_export.py
    │   │   │   │   │   ├── integration_test_reporting.py
    │   │   │   │   │   ├── json_export.py
    │   │   │   │   │   ├── metrics_compiler.py
    │   │   │   │   │   ├── regression_detection.py
    │   │   │   │   │   ├── stakeholder_reporting.py
    │   │   │   │   │   ├── trend_analysis.py
    │   │   │   │   │   ├── trend_analyzers.py
    │   │   │   │   │   ├── trend_predictions.py
    │   │   │   │   │   └── validation_utils.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── automation_orchestrator.py
    │   │   │   │   ├── automation_protocols.py
    │   │   │   │   ├── ci_integration.py
    │   │   │   │   ├── performance_benchmarking.py
    │   │   │   │   ├── resource_cleanup_monitoring.py
    │   │   │   │   ├── resource_cleanup_protocols.py
    │   │   │   │   ├── resource_cleanup_validation.py
    │   │   │   │   ├── run_performance_benchmarking_tests.py
    │   │   │   │   ├── test_automation_execution.py
    │   │   │   │   ├── test_data_automation.py
    │   │   │   │   ├── test_performance_benchmarking.py
    │   │   │   │   ├── test_resource_cleanup_validation.py
    │   │   │   │   └── workflow_automation.py
    │   │   │   ├── concurrent_tests/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── test_multi_user_operations.py
    │   │   │   │   ├── test_resource_contention.py
    │   │   │   │   └── test_system_stability.py
    │   │   │   ├── helpers/
    │   │   │   ├── workflow_components/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── concurrent/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── base.py
    │   │   │   │   │   ├── data_integrity.py
    │   │   │   │   │   ├── multi_user.py
    │   │   │   │   │   ├── resource_contention.py
    │   │   │   │   │   ├── stability.py
    │   │   │   │   │   └── synchronization.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── component_interaction_workflow.py
    │   │   │   │   ├── config_error_component.py
    │   │   │   │   ├── config_workflow.py
    │   │   │   │   ├── error_scenario_mixin.py
    │   │   │   │   ├── session_state_mixin.py
    │   │   │   │   ├── training_error_component.py
    │   │   │   │   └── training_workflow.py
    │   │   │   ├── __init__.py
    │   │   │   ├── test_advanced_workflows.py
    │   │   │   ├── test_base.py
    │   │   │   ├── test_basic_workflows.py
    │   │   │   ├── test_component_interactions.py
    │   │   │   ├── test_concurrent_operations.py
    │   │   │   ├── test_config_editor_component.py
    │   │   │   ├── test_config_io.py
    │   │   │   ├── test_error_scenarios.py
    │   │   │   ├── test_file_browser_component.py
    │   │   │   ├── test_session_state_simple.py
    │   │   │   ├── test_session_state_verification.py
    │   │   │   ├── test_specialized_config.py
    │   │   │   ├── test_specialized_parsing.py
    │   │   │   ├── test_specialized_run_manager.py
    │   │   │   ├── test_specialized_streaming.py
    │   │   │   ├── test_specialized_tensorboard.py
    │   │   │   ├── test_specialized_threading.py
    │   │   │   ├── test_workflow_performance.py
    │   │   │   ├── test_workflow_scenarios.py
    │   │   │   ├── test_yaml_validation.py
    │   │   │   └── workflow_scenarios.py
    │   │   ├── model/
    │   │   │   ├── __pycache__/
    │   │   │   ├── factory/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   └── test_registry_integration.py
    │   │   │   ├── conftest.py
    │   │   │   ├── test_cbam_integration.py
    │   │   │   ├── test_cnn_convlstm_unet.py
    │   │   │   ├── test_config_validation.py
    │   │   │   ├── test_factory_config.py
    │   │   │   ├── test_factory_instantiation_flow.py
    │   │   │   ├── test_integration.py
    │   │   │   ├── test_model_factory.py
    │   │   │   ├── test_swin_integration.py
    │   │   │   ├── test_swin_transfer_learning.py
    │   │   │   ├── test_swin_unet_integration.py
    │   │   │   └── test_unet_aspp_integration.py
    │   │   ├── monitoring/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   └── test_monitoring_integration.py
    │   │   ├── reporting/
    │   │   │   ├── __pycache__/
    │   │   │   ├── analysis/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── test_data_aggregation.py
    │   │   │   │   └── test_trend_analysis.py
    │   │   │   ├── __init__.py
    │   │   │   ├── test_automated_comparison.py
    │   │   │   ├── test_end_to_end_reporting.py
    │   │   │   └── test_sample_report_generation.py
    │   │   ├── training/
    │   │   │   ├── __pycache__/
    │   │   │   ├── test_artifacts_performance_regression.py
    │   │   │   ├── test_config_parser_validation.py
    │   │   │   ├── test_enhanced_combinators_validation.py
    │   │   │   ├── test_enhanced_registry_validation.py
    │   │   │   ├── test_loss_factory_integration.py
    │   │   │   ├── test_standardized_config_integration.py
    │   │   │   ├── test_trainer_integration.py
    │   │   │   ├── test_training_artifacts_integration.py
    │   │   │   └── test_training_loop.py
    │   │   ├── utils/
    │   │   │   ├── __pycache__/
    │   │   │   ├── conftest.py
    │   │   │   ├── test_orchestration.py
    │   │   │   ├── test_packaging_system.py
    │   │   │   ├── test_traceability_access.py
    │   │   │   ├── test_traceability_advanced_workflows.py
    │   │   │   ├── test_traceability_bulk_operations.py
    │   │   │   ├── test_traceability_operations.py
    │   │   │   └── test_traceability_workflows.py
    │   │   ├── test_backward_compatibility.py
    │   │   └── test_visualization_integration.py
    │   ├── tools/
    │   │   ├── analysis/
    │   │   │   ├── __pycache__/
    │   │   │   ├── comprehensive_failure_analysis.py
    │   │   │   ├── failure_data.json
    │   │   │   ├── pytest_executor.py
    │   │   │   ├── pytest_output_parser.py
    │   │   │   ├── report_generator.py
    │   │   │   ├── test_failure_analysis.py
    │   │   │   ├── test_failure_categorization.py
    │   │   │   └── test_priority_matrix_creator.py
    │   │   ├── testing/
    │   │   │   ├── __pycache__/
    │   │   │   └── test_config_system.py
    │   │   └── utilities/
    │   │       └── temp_storage.py
    │   ├── unit/
    │   │   ├── __pycache__/
    │   │   ├── data/
    │   │   │   ├── __pycache__/
    │   │   │   ├── test_dataloader.py
    │   │   │   ├── test_dataset_pipeline.py
    │   │   │   ├── test_distributed.py
    │   │   │   ├── test_factory.py
    │   │   │   ├── test_memory.py
    │   │   │   └── test_sampler.py
    │   │   ├── deployment/
    │   │   │   ├── test_orchestration.py
    │   │   │   └── test_production_readiness_validator.py
    │   │   ├── docker/
    │   │   │   ├── __pycache__/
    │   │   │   ├── test_env_manager.py
    │   │   │   └── test_health_check_system.py
    │   │   ├── e2e/
    │   │   │   ├── capture/
    │   │   │   ├── cleanup/
    │   │   │   ├── config/
    │   │   │   ├── performance/
    │   │   │   │   └── reporting/
    │   │   │   └── waits/
    │   │   ├── evaluation/
    │   │   │   ├── __pycache__/
    │   │   │   ├── test_advanced_training_viz.py
    │   │   │   ├── test_core.py
    │   │   │   ├── test_data.py
    │   │   │   ├── test_ensemble.py
    │   │   │   ├── test_evaluate.py
    │   │   │   ├── test_evaluation_main.py
    │   │   │   ├── test_loading.py
    │   │   │   └── test_results.py
    │   │   ├── gui/
    │   │   │   ├── __pycache__/
    │   │   │   ├── components/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── conftest.py
    │   │   │   │   ├── test_component_base.py
    │   │   │   │   ├── test_file_browser_component.py
    │   │   │   │   ├── test_file_upload_component.py
    │   │   │   │   ├── test_logo_component.py
    │   │   │   │   ├── test_page_router.py
    │   │   │   │   ├── test_results_display.py
    │   │   │   │   ├── test_sidebar_component.py
    │   │   │   │   └── test_theme_component.py
    │   │   │   ├── pages/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── test_advanced_config_page.py
    │   │   │   │   ├── test_config_page.py
    │   │   │   │   ├── test_home_page.py
    │   │   │   │   ├── test_pages_smoke.py
    │   │   │   │   └── test_train_page.py
    │   │   │   ├── utils/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── config/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── validation/
    │   │   │   │   │   ├── test_cache.py
    │   │   │   │   │   ├── test_formatters.py
    │   │   │   │   │   ├── test_io.py
    │   │   │   │   │   └── test_templates.py
    │   │   │   │   ├── test_export_manager.py
    │   │   │   │   ├── test_gui_config.py
    │   │   │   │   ├── test_performance_optimizer.py
    │   │   │   │   └── test_session_state.py
    │   │   │   ├── __init__.py
    │   │   │   ├── test_critical_coverage_paths.py
    │   │   │   ├── test_edge_cases.py
    │   │   │   ├── test_enhanced_abort.py
    │   │   │   ├── test_error_console.py
    │   │   │   ├── test_error_console_simple.py
    │   │   │   ├── test_essential_coverage.py
    │   │   │   ├── test_file_upload.py
    │   │   │   ├── test_session_state_updates.py
    │   │   │   ├── test_tensorboard_coverage.py
    │   │   │   └── test_threading_integration.py
    │   │   ├── integration/
    │   │   │   └── gui/
    │   │   │       └── automation/
    │   │   │           └── reporting/
    │   │   ├── model/
    │   │   │   ├── __pycache__/
    │   │   │   ├── config/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   └── test_instantiation.py
    │   │   │   ├── decoder/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── test_channel_utils.py
    │   │   │   │   ├── test_cnn_decoder_channel_handling.py
    │   │   │   │   ├── test_cnn_decoder_error_handling.py
    │   │   │   │   ├── test_cnn_decoder_forward_pass.py
    │   │   │   │   ├── test_cnn_decoder_initialization.py
    │   │   │   │   └── test_cnn_decoder_special_features.py
    │   │   │   ├── factory/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   └── test_registry.py
    │   │   │   ├── architectures
    │   │   │   ├── conftest.py
    │   │   │   ├── test_aspp.py
    │   │   │   ├── test_base.py
    │   │   │   ├── test_bottleneckblock.py
    │   │   │   ├── test_cbam.py
    │   │   │   ├── test_cbam_config.py
    │   │   │   ├── test_cnn_encoder.py
    │   │   │   ├── test_convlstm.py
    │   │   │   ├── test_decoderblock.py
    │   │   │   ├── test_encoderblock.py
    │   │   │   ├── test_exports.py
    │   │   │   ├── test_factory_utils.py
    │   │   │   ├── test_feature_info_utils.py
    │   │   │   ├── test_hybrid_registry.py
    │   │   │   ├── test_import_compat.py
    │   │   │   ├── test_registry.py
    │   │   │   ├── test_swin_basic.py
    │   │   │   ├── test_swin_encoder.py
    │   │   │   ├── test_swin_transfer_learning_script.py
    │   │   │   ├── test_swin_transformer_encoder.py
    │   │   │   ├── test_swin_unet.py
    │   │   │   ├── test_thread_safety.py
    │   │   │   ├── test_unet.py
    │   │   │   └── test_utils.py
    │   │   ├── reporting/
    │   │   │   ├── __init__.py
    │   │   │   ├── test_configurable_templates.py
    │   │   │   ├── test_publication_figures.py
    │   │   │   └── test_recommendation_engine.py
    │   │   ├── training/
    │   │   │   ├── __pycache__/
    │   │   │   ├── losses/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── test_clean_factory.py
    │   │   │   │   ├── test_config_parser.py
    │   │   │   │   ├── test_enhanced_combinators.py
    │   │   │   │   ├── test_focal_dice_loss.py
    │   │   │   │   ├── test_isolated_clean_factory.py
    │   │   │   │   ├── test_loss_factory.py
    │   │   │   │   ├── test_loss_registry.py
    │   │   │   │   ├── test_recursive_factory.py
    │   │   │   │   ├── test_recursive_factory_basic.py
    │   │   │   │   ├── test_recursive_factory_combinations.py
    │   │   │   │   ├── test_recursive_factory_config.py
    │   │   │   │   ├── test_recursive_factory_errors.py
    │   │   │   │   ├── test_recursive_factory_performance.py
    │   │   │   │   └── test_recursive_factory_regression.py
    │   │   │   ├── test_losses.py
    │   │   │   ├── test_lr_scheduler_factory.py
    │   │   │   ├── test_metrics.py
    │   │   │   ├── test_reproducibility.py
    │   │   │   ├── test_trainer.py
    │   │   │   ├── test_trainer_initialization.py
    │   │   │   └── test_trainer_training.py
    │   │   ├── utils/
    │   │   │   ├── __pycache__/
    │   │   │   ├── artifacts/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.cpython-312.pyc
    │   │   │   │   │   ├── test_artifact_manager.cpython-312-pytest-8.4.1.pyc
    │   │   │   │   │   ├── test_artifact_versioner.cpython-312-pytest-8.4.1.pyc
    │   │   │   │   │   └── test_checkpointing.cpython-312-pytest-8.4.1.pyc
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── test_artifact_manager.py
    │   │   │   │   ├── test_artifact_versioner.py
    │   │   │   │   └── test_checkpointing.py
    │   │   │   ├── config/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── test_env.py
    │   │   │   │   ├── test_override.py
    │   │   │   │   ├── test_schema.py
    │   │   │   │   ├── test_standardized_storage.py
    │   │   │   │   └── test_validation.py
    │   │   │   ├── data/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── test_dataset.py
    │   │   │   │   └── test_splitting.py
    │   │   │   ├── experiment/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── test_experiment_metadata.py
    │   │   │   │   ├── test_experiment_tracker.py
    │   │   │   │   ├── test_experiment_tracker_artifacts.py
    │   │   │   │   └── test_experiment_tracker_lifecycle.py
    │   │   │   ├── integrity/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── test_artifact_verifier.py
    │   │   │   │   ├── test_checkpoint_verifier.py
    │   │   │   │   ├── test_config_verifier.py
    │   │   │   │   ├── test_experiment_verifier.py
    │   │   │   │   └── test_integrity_core.py
    │   │   │   ├── logging/
    │   │   │   │   └── __init__.py
    │   │   │   ├── monitoring/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── test_callbacks.py
    │   │   │   │   ├── test_monitoring_manager.py
    │   │   │   │   └── test_retention.py
    │   │   │   ├── monitoring_logging/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── test_logging.py
    │   │   │   │   └── test_metrics_manager.py
    │   │   │   ├── traceability/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── test_integration_manager.py
    │   │   │   │   ├── test_integration_manager_bulk.py
    │   │   │   │   ├── test_integration_manager_compliance.py
    │   │   │   │   ├── test_integration_manager_search.py
    │   │   │   │   ├── test_lineage_manager.py
    │   │   │   │   ├── test_query_interface.py
    │   │   │   │   └── test_storage.py
    │   │   │   └── training/
    │   │   │       ├── __pycache__/
    │   │   │       ├── __init__.py
    │   │   │       └── test_early_stopping.py
    │   │   ├── __init__.py
    │   │   ├── test_data_loader.py
    │   │   ├── test_interactive_plotly.py
    │   │   ├── test_main_data.py
    │   │   ├── test_main_environment.py
    │   │   ├── test_main_integration.py
    │   │   ├── test_main_model.py
    │   │   ├── test_main_training.py
    │   │   └── test_performance_analyzer.py
    │   ├── utils/
    │   │   ├── __pycache__/
    │   │   ├── unified_testing/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── __init__.py.backup
    │   │   │   ├── core.py
    │   │   │   ├── core.py.backup
    │   │   │   ├── helpers.py
    │   │   │   ├── helpers.py.backup
    │   │   │   ├── mocking.py
    │   │   │   ├── mocking.py.backup
    │   │   │   ├── performance.py
    │   │   │   ├── performance.py.backup
    │   │   │   └── visual.py
    │   │   ├── __init__.py
    │   │   ├── performance_optimizer.py
    │   │   ├── pytest_performance_plugin.py
    │   │   ├── test_benchmark.py
    │   │   └── visual_regression_benchmarks.py
    │   ├── __init__.py
    │   ├── conftest.py
    │   ├── README.md
    │   └── requirements-testing.txt
    ├── CHANGELOG.md
    ├── codecov.yml
    ├── debug_output.txt
    ├── environment.yml
    ├── mkdocs.yml
    ├── pyproject.toml
    ├── pyrightconfig.json
    ├── README.md
    ├── requirements.txt
    └── run.py
```
