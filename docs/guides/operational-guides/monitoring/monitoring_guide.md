# Monitoring Guide

## Overview

This guide covers monitoring and observability for the CrackSeg project, including training
monitoring, system health checks, and performance tracking.

## 📊 Training Monitoring

### TensorBoard Setup

**Start TensorBoard**:

```bash
# Start monitoring
tensorboard --logdir artifacts/experiments/tensorboard

# Access at: http://localhost:6006
```

**Key Metrics to Monitor**:

- **Training Loss**: Should decrease over epochs
- **Validation Loss**: Should decrease and not diverge from training
- **IoU**: Should increase over time
- **Dice Score**: Should increase over time
- **Precision/Recall**: Balance for crack detection

### Log Monitoring

**Log Locations**:

```bash
# Training logs
ls artifacts/experiments/*/logs/

# Checkpoint logs
ls artifacts/experiments/*/checkpoints/

# Configuration logs
ls artifacts/experiments/*/config.yaml
```

**Expected Log Pattern**:

```bash
Epoch 1: val_loss: 0.9075 | val_iou: 0.0902 | val_dice: 0.1623
Epoch 2: val_loss: 0.8833 | val_iou: 0.2220 | val_dice: 0.3543
```

## 🔍 System Health Monitoring

### GPU Monitoring

**Monitor GPU Usage**:

```bash
# Real-time GPU monitoring
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi
```

**Key GPU Metrics**:

- **Memory Usage**: Should stay under 8GB for RTX 3070 Ti
- **GPU Utilization**: Should be high during training
- **Temperature**: Should stay under 80°C
- **Power Usage**: Monitor for efficiency

### System Resource Monitoring

**CPU and Memory**:

```bash
# Linux/macOS
htop

# Windows
taskmgr

# Memory usage
free -h  # Linux
vm_stat  # macOS
```

**Key System Metrics**:

- **CPU Usage**: Should not be bottleneck
- **RAM Usage**: Should have sufficient free memory
- **Disk Space**: Monitor for checkpoint storage
- **Network**: If using remote datasets

## 📈 Performance Monitoring

### Training Performance

**Metrics to Track**:

- **Training Speed**: Epochs per hour
- **Memory Efficiency**: VRAM usage vs batch size
- **Convergence Rate**: Loss reduction speed
- **Validation Performance**: Metrics improvement

**Performance Baselines**:

```txt
SwinV2 360x360 (Crack500):
- Training Time: ~2-3 hours for 50 epochs
- Memory Usage: ~6-7GB VRAM
- Convergence: 30-40 epochs

SwinV2 320x320 (PY-CrackDB):
- Training Time: ~2-3 hours for 50 epochs
- Memory Usage: ~6-7GB VRAM
- Convergence: 30-40 epochs
```

### Model Performance

**Key Performance Indicators**:

- **IoU**: Target > 0.75 for production
- **Dice Score**: Target > 0.82 for production
- **Precision**: Balance with recall
- **Recall**: Should be high for crack detection

**Performance Tracking**:

```python
# Example monitoring script
import matplotlib.pyplot as plt

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('Loss Over Time')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(val_iou, label='IoU')
plt.title('IoU Over Time')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(val_dice, label='Dice')
plt.title('Dice Over Time')
plt.legend()

plt.tight_layout()
plt.show()
```

## 🚨 Alert Monitoring

### Critical Alerts

**Training Alerts**:

- **Loss Divergence**: Training loss increases significantly
- **Memory Overflow**: GPU memory exceeds 7.5GB
- **NaN Values**: Loss or metrics become NaN
- **Stalled Training**: No progress for 10+ epochs

**System Alerts**:

- **High Temperature**: GPU > 80°C
- **Low Disk Space**: < 10GB free
- **High Memory Usage**: > 90% RAM usage
- **Network Issues**: Dataset loading failures

### Alert Setup

**Basic Monitoring Script**:

```python
import psutil
import GPUtil
import time

def monitor_system():
    while True:
        # GPU monitoring
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            if gpu.memoryUtil > 0.9:  # 90% memory usage
                print(f"⚠️ High GPU memory usage: {gpu.memoryUtil*100:.1f}%")

            if gpu.temperature > 80:
                print(f"⚠️ High GPU temperature: {gpu.temperature}°C")

        # System monitoring
        if psutil.virtual_memory().percent > 90:
            print(f"⚠️ High RAM usage: {psutil.virtual_memory().percent}%")

        if psutil.disk_usage('/').percent > 90:
            print(f"⚠️ Low disk space: {psutil.disk_usage('/').percent}%")

        time.sleep(60)  # Check every minute
```

## 📊 Metrics Collection

### Training Metrics

**Automatic Collection**:

- **TensorBoard**: Automatic metric logging
- **Checkpoint Logs**: Saved in experiment directories
- **Configuration Logs**: Complete experiment state

**Manual Collection**:

```bash
# Extract metrics from logs
grep "val_loss\|val_iou\|val_dice" artifacts/experiments/*/logs/*.log

# Monitor specific experiment
tail -f artifacts/experiments/py_crackdb_swinv2/logs/training.log
```

### System Metrics

**Resource Monitoring**:

```bash
# GPU metrics
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv

# System metrics
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1
free -m | awk 'NR==2{printf "%.2f%%", $3*100/$2}'
df -h | awk '$NF=="/"{printf "%s", $5}'
```

## 🔧 Monitoring Tools

### Built-in Tools

**TensorBoard**:

- **Scalars**: Loss and metrics over time
- **Images**: Training samples and predictions
- **Graphs**: Model architecture visualization
- **Histograms**: Weight distributions

**PyTorch Profiler**:

```python
# Enable profiling
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    # Training code here
    pass

# Analyze results
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### External Tools

**System Monitoring**:

- **htop/top**: Process monitoring
- **iotop**: Disk I/O monitoring
- **nethogs**: Network usage monitoring
- **glances**: Comprehensive system monitoring

**GPU Monitoring**:

- **nvidia-smi**: NVIDIA GPU monitoring
- **nvtop**: Interactive GPU monitoring
- **gpustat**: Simple GPU status

## 📈 Reporting

### Daily Monitoring Report

**Template**:

```txt
=== CrackSeg Daily Monitoring Report ===
Date: [Date]
Environment: [Development/Production]

Training Status:
- Active Experiments: [List]
- Completed Today: [List]
- Failed: [List]

System Health:
- GPU Memory: [Usage/Total]
- System Memory: [Usage/Total]
- Disk Space: [Usage/Total]
- Temperature: [Current]

Performance Metrics:
- Average Training Time: [Hours]
- Best IoU Today: [Value]
- Best Dice Today: [Value]

Issues:
- [List any problems]
- [Resolution status]

Next Actions:
- [Planned activities]
```

### Weekly Performance Report

**Metrics to Include**:

- **Training Efficiency**: Experiments completed
- **Model Performance**: Best metrics achieved
- **System Utilization**: Resource usage patterns
- **Issues Resolved**: Problems and solutions
- **Improvements**: Optimizations implemented

## 🔄 Continuous Monitoring

### Automated Monitoring

**Scheduled Checks**:

```bash
# Create monitoring script
cat > monitor_crackseg.sh << 'EOF'
#!/bin/bash
# Check GPU status
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | \
awk -F',' '{printf "GPU Memory: %s/%s MB\n", $1, $2}'

# Check disk space
df -h | grep '/$' | awk '{print "Disk Usage: " $5}'

# Check running processes
ps aux | grep python | grep run.py | wc -l | \
awk '{print "Active Training Jobs: " $1}'
EOF

chmod +x monitor_crackseg.sh

# Schedule monitoring (every 5 minutes)
crontab -e
# Add: */5 * * * * /path/to/monitor_crackseg.sh >> /var/log/crackseg_monitoring.log
```

### Real-time Alerts

**Setup Alerts**:

```bash
# Email alerts for critical issues
echo "Subject: CrackSeg Alert - High GPU Memory Usage" | \
mail -s "Alert" your-email@domain.com

# Slack/Discord webhooks for team notifications
curl -X POST -H 'Content-type: application/json' \
--data '{"text":"⚠️ CrackSeg Alert: High GPU memory usage detected"}' \
https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

## 📚 References

- **Successful Experiments**: `../successful_experiments_guide.md`
- **Training Workflow**: `../workflows/training_workflow_guide.md`
- **Deployment Guide**: `../deployment/deployment_guide.md`
- **Troubleshooting**: `../../user-guides/troubleshooting.md`

---

**Last Updated**: December 2024
**Status**: Active - Basic monitoring implemented
**Next Steps**: Implement advanced monitoring with Prometheus/Grafana
