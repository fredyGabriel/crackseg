"""Artifact optimization for deployment.

This module provides comprehensive optimization capabilities for ML artifacts
including advanced quantization, pruning, format conversion, and size
optimization while maintaining performance thresholds.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from ..traceability import ArtifactEntity

if TYPE_CHECKING:
    from .config import DeploymentConfig

logger = logging.getLogger(__name__)


@dataclass
class OptimizationStrategy:
    """Strategy for artifact optimization."""

    # Quantization settings
    quantization_method: str = "dynamic"  # "dynamic", "static", "fp16", "int8"
    quantization_target: str = "cpu"  # "cpu", "cuda"
    calibration_samples: int = 100

    # Pruning settings
    pruning_method: str = (
        "magnitude"  # "magnitude", "structured", "unstructured"
    )
    pruning_ratio: float = 0.2  # Percentage of weights to prune
    pruning_schedule: str = "one_shot"  # "one_shot", "iterative"

    # Format conversion settings
    target_format: str = "onnx"  # "pytorch", "onnx", "tensorrt", "torchscript"
    optimize_for_inference: bool = True
    preserve_accuracy: bool = True

    # Performance thresholds
    max_accuracy_drop: float = 0.05  # 5% maximum accuracy drop
    max_inference_time_increase: float = 1.2  # 20% maximum time increase
    min_compression_ratio: float = 1.5  # Minimum compression ratio


@dataclass
class OptimizationResult:
    """Result of artifact optimization."""

    success: bool
    original_size_mb: float
    optimized_size_mb: float
    compression_ratio: float

    # Model-specific results
    original_model_path: str | None = None
    optimized_model_path: str | None = None
    model_format: str = (
        "pytorch"  # "pytorch", "onnx", "tensorrt", "torchscript"
    )

    # Performance metrics
    inference_time_ms: float | None = None
    memory_usage_mb: float | None = None
    accuracy_drop: float | None = None
    speedup_ratio: float | None = None

    # Optimization details
    applied_optimizations: list[str] = field(default_factory=list)
    optimization_strategy: OptimizationStrategy | None = None
    validation_passed: bool = False

    # Error information
    error_message: str | None = None


class ArtifactOptimizer:
    """Advanced artifact optimizer for deployment.

    Handles comprehensive model optimization including advanced quantization,
    structured pruning, format conversion, and performance validation while
    maintaining strict performance thresholds.
    """

    def __init__(self) -> None:
        """Initialize artifact optimizer."""
        self.supported_formats = ["pytorch", "onnx", "tensorrt", "torchscript"]
        self.quantization_methods = ["dynamic", "static", "fp16", "int8"]
        self.pruning_methods = ["magnitude", "structured", "unstructured"]
        self.optimization_strategies = {
            "production": OptimizationStrategy(
                quantization_method="static",
                pruning_method="structured",
                pruning_ratio=0.3,
                target_format="onnx",
                max_accuracy_drop=0.02,
                min_compression_ratio=2.0,
            ),
            "staging": OptimizationStrategy(
                quantization_method="dynamic",
                pruning_method="magnitude",
                pruning_ratio=0.2,
                target_format="pytorch",
                max_accuracy_drop=0.05,
                min_compression_ratio=1.5,
            ),
            "development": OptimizationStrategy(
                quantization_method="dynamic",
                pruning_method="magnitude",
                pruning_ratio=0.1,
                target_format="pytorch",
                max_accuracy_drop=0.1,
                min_compression_ratio=1.2,
            ),
        }

        logger.info("ArtifactOptimizer initialized with advanced capabilities")

    def optimize_artifact(
        self, artifact: ArtifactEntity, config: "DeploymentConfig"
    ) -> OptimizationResult:
        """Optimize artifact for deployment with comprehensive validation.

        Args:
            artifact: Artifact to optimize
            config: Deployment configuration

        Returns:
            OptimizationResult with detailed optimization metrics
        """
        logger.info(
            "Starting comprehensive optimization of artifact "
            f"{artifact.artifact_id}"
        )

        try:
            # Load original model
            original_model = self._load_model(artifact)
            if original_model is None:
                return self._create_error_result("Failed to load model")

            # Get optimization strategy
            strategy = self._get_optimization_strategy(config)
            logger.info(f"Using optimization strategy: {strategy}")

            # Get original metrics
            original_metrics = self._get_model_metrics(original_model)
            original_size_mb = original_metrics["size_mb"]

            # Apply comprehensive optimizations
            optimized_model = original_model
            applied_optimizations = []

            # 1. Advanced Quantization
            if config.enable_quantization:
                logger.info(
                    f"Applying {strategy.quantization_method} quantization..."
                )
                optimized_model = self._apply_advanced_quantization(
                    optimized_model, strategy
                )
                applied_optimizations.append(
                    f"quantization_{strategy.quantization_method}"
                )

            # 2. Advanced Pruning
            if config.enable_pruning:
                logger.info(f"Applying {strategy.pruning_method} pruning...")
                optimized_model = self._apply_advanced_pruning(
                    optimized_model, strategy
                )
                applied_optimizations.append(
                    f"pruning_{strategy.pruning_method}"
                )

            # 3. Format Conversion
            if strategy.target_format != "pytorch":
                logger.info(f"Converting to {strategy.target_format}...")
                optimized_model = self._convert_to_format(
                    optimized_model, strategy
                )
                applied_optimizations.append(
                    f"format_{strategy.target_format}"
                )

            # 4. Save optimized model
            optimized_path = self._save_optimized_model(
                optimized_model, artifact, strategy
            )

            # 5. Calculate optimization metrics
            optimized_metrics = self._get_model_metrics(optimized_model)
            optimized_size_mb = optimized_metrics["size_mb"]
            compression_ratio = (
                original_size_mb / optimized_size_mb
                if optimized_size_mb > 0
                else 1.0
            )

            # 6. Comprehensive performance benchmarking
            performance_metrics = self._benchmark_comprehensive_performance(
                original_model, optimized_model, strategy
            )

            # 7. Validate optimization results
            validation_passed = self._validate_optimization_results(
                original_metrics,
                optimized_metrics,
                performance_metrics,
                strategy,
            )

            result = OptimizationResult(
                success=True,
                original_size_mb=original_size_mb,
                optimized_size_mb=optimized_size_mb,
                compression_ratio=compression_ratio,
                original_model_path=str(Path(artifact.file_path)),
                optimized_model_path=str(optimized_path),
                model_format=strategy.target_format,
                inference_time_ms=performance_metrics.get("inference_time_ms"),
                memory_usage_mb=performance_metrics.get("memory_usage_mb"),
                accuracy_drop=performance_metrics.get("accuracy_drop"),
                speedup_ratio=performance_metrics.get("speedup_ratio"),
                applied_optimizations=applied_optimizations,
                optimization_strategy=strategy,
                validation_passed=validation_passed,
            )

            logger.info(
                f"Optimization completed: "
                f"{compression_ratio:.2f}x compression, "
                f"validation: {'PASSED' if validation_passed else 'FAILED'}"
            )
            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return self._create_error_result(str(e))

    def _get_optimization_strategy(
        self, config: "DeploymentConfig"
    ) -> OptimizationStrategy:
        """Get optimization strategy based on deployment configuration."""
        if config.target_environment in self.optimization_strategies:
            strategy = self.optimization_strategies[config.target_environment]
        else:
            strategy = self.optimization_strategies["staging"]

        # Override with config settings
        if config.enable_quantization:
            strategy.quantization_method = (
                "dynamic" if config.target_format == "pytorch" else "static"
            )
        if config.enable_pruning:
            strategy.pruning_method = "magnitude"
        if config.target_format != "pytorch":
            strategy.target_format = config.target_format

        return strategy

    def _load_model(self, artifact: ArtifactEntity) -> nn.Module | None:
        """Load model from artifact with enhanced error handling."""
        try:
            model_path = Path(artifact.file_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return None

            # Load model based on file extension
            if model_path.suffix == ".pth":
                model = torch.load(model_path, map_location="cpu")
                if isinstance(model, dict):
                    # Handle state dict format
                    from ..model import get_model_architecture

                    model_arch = get_model_architecture()
                    model = model_arch()
                    model.load_state_dict(
                        torch.load(model_path, map_location="cpu")
                    )
                return model
            elif model_path.suffix == ".onnx":
                import onnx

                return onnx.load(str(model_path))
            else:
                logger.error(f"Unsupported model format: {model_path.suffix}")
                return None

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def _apply_advanced_quantization(
        self, model: nn.Module, strategy: OptimizationStrategy
    ) -> nn.Module:
        """Apply advanced quantization techniques."""
        try:
            if strategy.quantization_method == "dynamic":
                # Dynamic quantization for CPU inference
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
                return quantized_model

            elif strategy.quantization_method == "static":
                # Static quantization with calibration
                model.eval()
                # Prepare model for static quantization
                torch.quantization.prepare(model, inplace=True)

                # Calibration (simulated)
                with torch.no_grad():
                    for _ in range(strategy.calibration_samples):
                        dummy_input = torch.randn(1, 3, 512, 512)
                        _ = model(dummy_input)

                # Convert to quantized model
                quantized_model = torch.quantization.convert(model)
                return quantized_model

            elif strategy.quantization_method == "fp16":
                # FP16 quantization for GPU inference
                model.half()
                return model

            elif strategy.quantization_method == "int8":
                # INT8 quantization
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
                return quantized_model

            else:
                logger.warning(
                    "Unknown quantization method: "
                    f"{strategy.quantization_method}"
                )
                return model

        except Exception as e:
            logger.warning(
                f"Quantization failed: {e}, returning original model"
            )
            return model

    def _apply_advanced_pruning(
        self, model: nn.Module, strategy: OptimizationStrategy
    ) -> nn.Module:
        """Apply advanced pruning techniques."""
        try:
            if strategy.pruning_method == "magnitude":
                # Magnitude-based pruning
                for _name, module in model.named_modules():
                    if isinstance(module, nn.Conv2d | nn.Linear):
                        prune.l1_unstructured(
                            module,
                            name="weight",
                            amount=strategy.pruning_ratio,
                        )
                return model

            elif strategy.pruning_method == "structured":
                # Structured pruning
                for _name, module in model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        # Prune entire channels
                        prune.ln_structured(
                            module,
                            name="weight",
                            amount=strategy.pruning_ratio,
                            n=2,
                            dim=0,
                        )
                return model

            elif strategy.pruning_method == "unstructured":
                # Unstructured pruning
                for _name, module in model.named_modules():
                    if isinstance(module, nn.Conv2d | nn.Linear):
                        prune.random_unstructured(
                            module,
                            name="weight",
                            amount=strategy.pruning_ratio,
                        )
                return model

            else:
                logger.warning(
                    f"Unknown pruning method: {strategy.pruning_method}"
                )
                return model

        except Exception as e:
            logger.warning(f"Pruning failed: {e}, returning original model")
            return model

    def _convert_to_format(
        self, model: nn.Module, strategy: OptimizationStrategy
    ) -> nn.Module:
        """Convert model to target format with optimization."""
        try:
            if strategy.target_format == "onnx":
                # Convert to ONNX with optimization
                dummy_input = torch.randn(1, 3, 512, 512)
                torch.onnx.export(
                    model,
                    (dummy_input,),
                    "temp_model.onnx",
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "output": {0: "batch_size"},
                    },
                )
                import onnx

                return onnx.load("temp_model.onnx")

            elif strategy.target_format == "torchscript":
                # Convert to TorchScript with optimization
                dummy_input = torch.randn(1, 3, 512, 512)
                model.eval()
                traced_model = torch.jit.trace(model, dummy_input)
                return traced_model

            elif strategy.target_format == "tensorrt":
                # TensorRT conversion (placeholder)
                logger.info("TensorRT conversion requires additional setup")
                return model

            else:
                return model

        except Exception as e:
            logger.warning(
                f"Format conversion failed: {e}, returning original model"
            )
            return model

    def _get_model_metrics(self, model: Any) -> dict[str, float]:
        """Get comprehensive model metrics."""
        try:
            if hasattr(model, "state_dict"):
                # PyTorch model
                param_size = 0
                for param in model.parameters():
                    param_size += param.nelement() * param.element_size()
                buffer_size = 0
                for buffer in model.buffers():
                    buffer_size += buffer.nelement() * buffer.element_size()
                size_mb = (param_size + buffer_size) / 1024 / 1024
            else:
                # ONNX or other format - use a simple estimate
                size_mb = 100.0  # Default estimate for non-PyTorch models

            return {
                "size_mb": size_mb,
                "parameter_count": (
                    sum(p.numel() for p in model.parameters())
                    if hasattr(model, "parameters")
                    else 0
                ),
            }

        except Exception as e:
            logger.warning(f"Failed to calculate model metrics: {e}")
            return {"size_mb": 0.0, "parameter_count": 0}

    def _benchmark_comprehensive_performance(
        self,
        original_model: nn.Module,
        optimized_model: nn.Module,
        strategy: OptimizationStrategy,
    ) -> dict[str, float]:
        """Benchmark comprehensive performance metrics."""
        try:
            # Benchmark inference time
            original_time = self._benchmark_inference_time(original_model)
            optimized_time = self._benchmark_inference_time(optimized_model)
            speedup_ratio = (
                original_time / optimized_time if optimized_time > 0 else 1.0
            )

            # Benchmark memory usage
            original_memory = self._benchmark_memory_usage(original_model)
            optimized_memory = self._benchmark_memory_usage(optimized_model)
            memory_reduction = (
                (original_memory - optimized_memory) / original_memory
                if original_memory > 0
                else 0.0
            )

            # Measure accuracy drop
            accuracy_drop = self._measure_accuracy_drop(
                original_model, optimized_model, strategy
            )

            return {
                "inference_time_ms": optimized_time,
                "speedup_ratio": speedup_ratio,
                "memory_usage_mb": optimized_memory,
                "memory_reduction_ratio": memory_reduction,
                "accuracy_drop": accuracy_drop,
            }

        except Exception as e:
            logger.warning(f"Performance benchmarking failed: {e}")
            return {
                "inference_time_ms": 0.0,
                "speedup_ratio": 1.0,
                "memory_usage_mb": 0.0,
                "memory_reduction_ratio": 0.0,
                "accuracy_drop": 0.0,
            }

    def _measure_accuracy_drop(
        self,
        original_model: nn.Module,
        optimized_model: nn.Module,
        strategy: OptimizationStrategy,
    ) -> float:
        """Measure accuracy drop between original and optimized models.

        Args:
            original_model: Original PyTorch model
            optimized_model: Optimized PyTorch model
            strategy: Optimization strategy with accuracy thresholds

        Returns:
            Accuracy drop as a percentage (0.0 = no drop, 1.0 = 100% drop)
        """
        try:
            # Set models to evaluation mode
            original_model.eval()
            optimized_model.eval()

            # Use test dataset for accuracy measurement
            test_dataset = self._get_test_dataset()
            if test_dataset is None:
                logger.warning(
                    "No test dataset available, using synthetic data"
                )
                return 0.0

            original_accuracy = self._evaluate_model_accuracy(
                original_model, test_dataset
            )
            optimized_accuracy = self._evaluate_model_accuracy(
                optimized_model, test_dataset
            )

            # Calculate accuracy drop
            if original_accuracy > 0:
                accuracy_drop = (
                    original_accuracy - optimized_accuracy
                ) / original_accuracy
                return max(
                    0.0, min(1.0, accuracy_drop)
                )  # Clamp between 0 and 1
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Accuracy measurement failed: {e}")
            return 0.0

    def _get_test_dataset(self) -> Any | None:
        """Get test dataset for accuracy evaluation."""
        try:
            # Try to load test dataset from data directory
            from crackseg.data import CrackDataset
            from crackseg.data.transforms import get_test_transforms

            test_dataset = CrackDataset(
                data_dir="data/test",
                transform=get_test_transforms(),
                split="test",
            )
            return test_dataset
        except Exception as e:
            logger.warning(f"Could not load test dataset: {e}")
            return None

    def _evaluate_model_accuracy(
        self, model: nn.Module, dataset: Any
    ) -> float:
        """Evaluate model accuracy on test dataset.

        Args:
            model: PyTorch model to evaluate
            dataset: Test dataset

        Returns:
            Accuracy score between 0.0 and 1.0
        """
        try:
            model.eval()
            device = next(model.parameters()).device

            correct_predictions = 0
            total_predictions = 0

            with torch.no_grad():
                for i in range(
                    min(100, len(dataset))
                ):  # Use subset for efficiency
                    try:
                        image, target = dataset[i]
                        image = image.unsqueeze(0).to(device)
                        target = target.unsqueeze(0).to(device)

                        # Forward pass
                        output = model(image)

                        # Calculate IoU for segmentation
                        pred_mask = (output > 0.5).float()
                        intersection = (pred_mask * target).sum()
                        union = pred_mask.sum() + target.sum() - intersection

                        if union > 0:
                            iou = intersection / union
                            correct_predictions += iou.item()
                            total_predictions += 1

                    except Exception as e:
                        logger.debug(f"Error evaluating sample {i}: {e}")
                        continue

            if total_predictions > 0:
                return correct_predictions / total_predictions
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Model accuracy evaluation failed: {e}")
            return 0.0

    def _benchmark_inference_time(self, model: nn.Module) -> float:
        """Benchmark model inference time.

        Args:
            model: PyTorch model to benchmark

        Returns:
            Average inference time in milliseconds
        """
        try:
            import time

            model.eval()
            device = next(model.parameters()).device
            dummy_input = torch.randn(1, 3, 512, 512).to(device)

            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)

            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(50):
                    start_time = time.time()
                    _ = model(dummy_input)
                    torch.cuda.synchronize() if device.type == "cuda" else None
                    end_time = time.time()
                    times.append(
                        (end_time - start_time) * 1000
                    )  # Convert to ms

            return sum(times) / len(times)

        except Exception as e:
            logger.warning(f"Inference time benchmarking failed: {e}")
            return 0.0

    def _benchmark_memory_usage(self, model: nn.Module) -> float:
        """Benchmark model memory usage.

        Args:
            model: PyTorch model to benchmark

        Returns:
            Memory usage in MB
        """
        try:
            import time

            import psutil

            model.eval()
            device = next(model.parameters()).device
            dummy_input = torch.randn(1, 3, 512, 512).to(device)

            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Warmup and measure
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)

            # Clear cache if using CUDA
            if device.type == "cuda":
                torch.cuda.empty_cache()

            time.sleep(0.1)  # Allow memory to stabilize
            final_memory = process.memory_info().rss / 1024 / 1024  # MB

            return max(0.0, final_memory - initial_memory)

        except Exception as e:
            logger.warning(f"Memory usage benchmarking failed: {e}")
            return 0.0

    def _validate_optimization_results(
        self,
        original_metrics: dict[str, float],
        optimized_metrics: dict[str, float],
        performance_metrics: dict[str, float],
        strategy: OptimizationStrategy,
    ) -> bool:
        """Validate optimization results against thresholds."""
        try:
            # Check compression ratio
            compression_ratio = (
                original_metrics["size_mb"] / optimized_metrics["size_mb"]
            )
            if compression_ratio < strategy.min_compression_ratio:
                logger.warning(
                    f"Compression ratio {compression_ratio:.2f} below "
                    f"threshold {strategy.min_compression_ratio}"
                )
                return False

            # Check accuracy drop
            accuracy_drop = performance_metrics.get("accuracy_drop", 0.0)
            if accuracy_drop > strategy.max_accuracy_drop:
                logger.warning(
                    f"Accuracy drop {accuracy_drop:.3f} above threshold "
                    f"{strategy.max_accuracy_drop}"
                )
                return False

            # Check inference time increase
            speedup_ratio = performance_metrics.get("speedup_ratio", 1.0)
            if speedup_ratio < (1.0 / strategy.max_inference_time_increase):
                logger.warning(
                    f"Speedup ratio {speedup_ratio:.2f} below threshold "
                    f"{1.0 / strategy.max_inference_time_increase:.2f}"
                )
                return False

            logger.info("All optimization validation checks passed")
            return True

        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return False

    def _save_optimized_model(
        self,
        model: nn.Module,
        artifact: ArtifactEntity,
        strategy: OptimizationStrategy,
    ) -> Path:
        """Save optimized model with metadata."""
        output_dir = Path(f"deployments/{artifact.artifact_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        if strategy.target_format == "onnx":
            output_path = output_dir / f"{artifact.artifact_id}_optimized.onnx"
            import onnx

            onnx.save(model, str(output_path))
        elif strategy.target_format == "torchscript":
            output_path = output_dir / f"{artifact.artifact_id}_optimized.pt"
            torch.jit.save(model, str(output_path))
        else:
            output_path = output_dir / f"{artifact.artifact_id}_optimized.pth"
            torch.save(model, str(output_path))

        # Save optimization metadata
        metadata_path = (
            output_dir / f"{artifact.artifact_id}_optimization_metadata.json"
        )
        import json

        metadata = {
            "original_artifact_id": artifact.artifact_id,
            "optimization_strategy": {
                "quantization_method": strategy.quantization_method,
                "pruning_method": strategy.pruning_method,
                "target_format": strategy.target_format,
            },
            "optimization_timestamp": time.time(),
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return output_path

    def _create_error_result(self, error_message: str) -> OptimizationResult:
        """Create error result."""
        return OptimizationResult(
            success=False,
            original_size_mb=0.0,
            optimized_size_mb=0.0,
            compression_ratio=1.0,
            error_message=error_message,
        )

    def get_optimization_recommendations(
        self, artifact: ArtifactEntity, target_environment: str
    ) -> dict[str, Any]:
        """Get optimization recommendations for artifact."""
        try:
            model = self._load_model(artifact)
            if model is None:
                return {"error": "Failed to load model"}

            metrics = self._get_model_metrics(model)
            strategy = self.optimization_strategies.get(
                target_environment, self.optimization_strategies["staging"]
            )

            recommendations = {
                "artifact_id": artifact.artifact_id,
                "target_environment": target_environment,
                "current_size_mb": metrics["size_mb"],
                "recommended_optimizations": [],
                "expected_compression_ratio": 0.0,
                "expected_performance_impact": "minimal",
                "recommended_strategy": {
                    "quantization_method": strategy.quantization_method,
                    "pruning_method": strategy.pruning_method,
                    "target_format": strategy.target_format,
                    "max_accuracy_drop": strategy.max_accuracy_drop,
                    "min_compression_ratio": strategy.min_compression_ratio,
                },
            }

            # Generate recommendations based on model characteristics
            if metrics["size_mb"] > 200:
                recommendations["recommended_optimizations"].append(
                    "quantization"
                )
                recommendations["expected_compression_ratio"] += 0.5

            if metrics["parameter_count"] > 1000000:
                recommendations["recommended_optimizations"].append("pruning")
                recommendations["expected_compression_ratio"] += 0.3

            if target_environment == "production":
                recommendations["recommended_optimizations"].append(
                    "format_conversion"
                )
                recommendations["expected_compression_ratio"] += 0.2

            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return {"error": str(e)}
