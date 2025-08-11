<!-- markdownlint-disable-file -->
# Hydra & Registry Alignment Report

## Architectures

Summary | Count
:-- | --:
Hydra entries | 7
Registered | 1
Only in Hydra | 7
Only in Registry | 1

### In Hydra but not Registered
- `cnn_convlstm_unet`
- `swinv2_hybrid`
- `unet_aspp`
- `unet_cnn`
- `unet_swin`
- `unet_swin_base`
- `unet_swin_transfer`

### Registered but no Hydra config
- `SwinV2CnnAsppUNet`

## Encoder

Summary | Count
:-- | --:
Hydra entries | 3
Registered | 4
Only in Hydra | 3
Only in Registry | 4

### In Hydra but not Registered
- `default_encoder`
- `mock_encoder`
- `swin_transformer_encoder`

### Registered but no Hydra config
- `CNNEncoder`
- `EncoderBlock`
- `EncoderBlockAlias`
- `ResNetEncoder`

## Decoder

Summary | Count
:-- | --:
Hydra entries | 2
Registered | 1
Only in Hydra | 2
Only in Registry | 1

### In Hydra but not Registered
- `default_decoder`
- `mock_decoder`

### Registered but no Hydra config
- `CNNDecoder`

## Bottleneck

Summary | Count
:-- | --:
Hydra entries | 4
Registered | 2
Only in Hydra | 4
Only in Registry | 2

### In Hydra but not Registered
- `aspp_bottleneck`
- `convlstm_bottleneck`
- `default_bottleneck`
- `mock_bottleneck`

### Registered but no Hydra config
- `ASPPModule`
- `BottleneckBlock`
