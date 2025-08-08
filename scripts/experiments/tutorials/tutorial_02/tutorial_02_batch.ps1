# Tutorial 02 Batch Experiment Execution Script
#
# This script runs multiple experiments created in Tutorial 02:
# "Creating Custom Experiments (CLI Only)"
#
# Reference: docs/tutorials/02_custom_experiment_cli.md
#
# Usage: .\scripts\experiments\tutorial_02_batch.ps1

# List of experiments to run (Tutorial 02 experiments)
$experiments = @(
    @{
        name   = "high_lr_experiment"
        params = @("training.learning_rate=0.001", "training.epochs=50", "data.dataloader.batch_size=8")
    },
    @{
        name   = "low_lr_experiment"
        params = @("training.learning_rate=0.00001", "training.epochs=100", "data.dataloader.batch_size=16")
    },
    @{
        name   = "medium_lr_experiment"
        params = @("training.learning_rate=0.0001", "training.epochs=75", "data.dataloader.batch_size=12")
    }
)

Write-Host "Tutorial 02 - Batch Experiment Execution" -ForegroundColor Green
Write-Host "Reference: docs/tutorials/02_custom_experiment_cli.md" -ForegroundColor Yellow
Write-Host "=" * 60 -ForegroundColor Cyan

# Activate conda environment
conda activate crackseg

# Run each experiment
foreach ($exp in $experiments) {
    Write-Host "Running experiment: $($exp.name)" -ForegroundColor Green
    Write-Host "Parameters: $($exp.params -join ' ')" -ForegroundColor Yellow

    $command = "python run.py --config-name $($exp.name)"
    Write-Host "Command: $command" -ForegroundColor Cyan

    Invoke-Expression $command

    # Check if experiment completed successfully
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $($exp.name) completed successfully" -ForegroundColor Green
    }
    else {
        Write-Host "❌ $($exp.name) failed" -ForegroundColor Red
    }

    Write-Host "---"
}

Write-Host "All Tutorial 02 experiments completed!" -ForegroundColor Green
Write-Host "Run comparison: python scripts/experiments/tutorial_02/tutorial_02_compare.py" -ForegroundColor Yellow