"""
Test data provisioning suites. This module provides various test data
provisioning suites for different testing scenarios.
"""

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import ProvisioningResult, TestDataProvisioner


def provision_basic_suite(self: "TestDataProvisioner") -> "ProvisioningResult":
    """
    Provision a basic test data suite. Args: self: TestDataProvisioner
    instance Returns: ProvisioningResult with operation details
    """
    start_time = time.time()
    provisioned_items = []
    errors = []

    try:
        # Generate basic configuration
        config_data = self.data_factory.generate_config(config_type="basic")
        self.provisioned_data["basic_config"] = config_data
        provisioned_items.append("basic_config")

        # Generate crack image
        crack_image = self.data_factory.generate_image(
            image_type="crack", width=512, height=512, crack_density=0.05
        )
        self.provisioned_data["crack_image"] = crack_image
        provisioned_items.append("crack_image")

        # Generate clean image
        clean_image = self.data_factory.generate_image(image_type="no_crack")
        self.provisioned_data["clean_image"] = clean_image
        provisioned_items.append("clean_image")

        # Generate simple model
        model_data = self.data_factory.generate_model(model_type="simple")
        self.provisioned_data["simple_model"] = model_data
        provisioned_items.append("simple_model")

        duration = time.time() - start_time

        self.logger.info(
            f"Basic suite provisioned in {duration:.2f}s: "
            f"{len(provisioned_items)} items"
        )

        return {
            "success": True,
            "duration": duration,
            "provisioned_items": provisioned_items,
            "errors": errors,
            "metadata": {
                "suite_type": "basic",
                "item_count": len(provisioned_items),
            },
        }

    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Basic suite provisioning failed: {e}"
        self.logger.error(error_msg)
        errors.append(error_msg)

        return {
            "success": False,
            "duration": duration,
            "provisioned_items": provisioned_items,
            "errors": errors,
            "metadata": {"suite_type": "basic", "failure_reason": str(e)},
        }


def provision_comprehensive_suite(
    self: "TestDataProvisioner",
) -> "ProvisioningResult":
    """
    Provision a comprehensive test data suite with multiple data types.
    Args: self: TestDataProvisioner instance Returns: ProvisioningResult
    with operation details
    """
    start_time = time.time()
    provisioned_items = []
    errors = []

    try:
        # Generate multiple configurations
        for config_type in ["basic", "advanced"]:
            config_data = self.data_factory.generate_config(
                config_type=config_type
            )
            key = f"{config_type}_config"
            self.provisioned_data[key] = config_data
            provisioned_items.append(key)

        # Generate invalid configuration for error testing
        invalid_config = self.data_factory.generate_config(
            config_type="basic", invalid=True, missing_keys=["model"]
        )
        self.provisioned_data["invalid_config"] = invalid_config
        provisioned_items.append("invalid_config")

        # Generate various image types
        for image_type in ["crack", "no_crack"]:
            for size in [(256, 256), (512, 512)]:
                image_data = self.data_factory.generate_image(
                    image_type=image_type, width=size[0], height=size[1]
                )
                key = f"{image_type}_image_{size[0]}x{size[1]}"
                self.provisioned_data[key] = image_data
                provisioned_items.append(key)

        # Generate different model checkpoints
        for model_type in ["simple", "complex"]:
            for include_opt in [True, False]:
                model_data = self.data_factory.generate_model(
                    model_type=model_type, include_optimizer=include_opt
                )
                key = f"{model_type}_model_opt_{include_opt}"
                self.provisioned_data[key] = model_data
                provisioned_items.append(key)

        # Generate corrupted model for error testing
        corrupted_model = self.data_factory.generate_model(corrupt_data=True)
        self.provisioned_data["corrupted_model"] = corrupted_model
        provisioned_items.append("corrupted_model")

        duration = time.time() - start_time

        self.logger.info(
            f"Comprehensive suite provisioned in {duration:.2f}s: "
            f"{len(provisioned_items)} items"
        )

        return {
            "success": True,
            "duration": duration,
            "provisioned_items": provisioned_items,
            "errors": errors,
            "metadata": {
                "suite_type": "comprehensive",
                "item_count": len(provisioned_items),
            },
        }

    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Comprehensive suite provisioning failed: {e}"
        self.logger.error(error_msg)
        errors.append(error_msg)

        return {
            "success": False,
            "duration": duration,
            "provisioned_items": provisioned_items,
            "errors": errors,
            "metadata": {
                "suite_type": "comprehensive",
                "failure_reason": str(e),
            },
        }


def provision_error_test_data(
    self: "TestDataProvisioner",
) -> "ProvisioningResult":
    """
    Provision test data specifically for error condition testing. Args:
    self: TestDataProvisioner instance Returns: ProvisioningResult with
    operation details
    """
    start_time = time.time()
    provisioned_items = []
    errors = []

    try:
        # Generate invalid configurations
        invalid_configs: list[dict[str, Any]] = [
            {
                "name": "missing_model_config",
                "params": {"invalid": True, "missing_keys": ["model"]},
            },
            {
                "name": "missing_training_config",
                "params": {"invalid": True, "missing_keys": ["training"]},
            },
            {
                "name": "invalid_values_config",
                "params": {"invalid": True},
            },
        ]

        for config_spec in invalid_configs:
            name = str(config_spec["name"])
            params = dict(config_spec["params"])
            config_data = self.data_factory.generate_config(**params)
            self.provisioned_data[name] = config_data
            provisioned_items.append(name)

        # Generate corrupted model checkpoints
        corrupted_models: list[dict[str, Any]] = [
            {
                "name": "corrupted_simple_model",
                "params": {"corrupt_data": True},
            },
            {
                "name": "corrupted_complex_model",
                "params": {"model_type": "complex", "corrupt_data": True},
            },
        ]

        for model_spec in corrupted_models:
            name = str(model_spec["name"])
            params = dict(model_spec["params"])
            model_data = self.data_factory.generate_model(**params)
            self.provisioned_data[name] = model_data
            provisioned_items.append(name)

        # Generate edge case images
        edge_case_images: list[dict[str, Any]] = [
            {
                "name": "tiny_image",
                "params": {"width": 32, "height": 32, "image_type": "crack"},
            },
            {
                "name": "large_image",
                "params": {
                    "width": 2048,
                    "height": 2048,
                    "image_type": "crack",
                },
            },
            {
                "name": "extreme_crack_density",
                "params": {"crack_density": 0.5, "image_type": "crack"},
            },
        ]

        for image_spec in edge_case_images:
            name = str(image_spec["name"])
            params = dict(image_spec["params"])
            image_data = self.data_factory.generate_image(**params)
            self.provisioned_data[name] = image_data
            provisioned_items.append(name)

        duration = time.time() - start_time

        self.logger.info(
            f"Error test data provisioned in {duration:.2f}s: "
            f"{len(provisioned_items)} items"
        )

        return {
            "success": True,
            "duration": duration,
            "provisioned_items": provisioned_items,
            "errors": errors,
            "metadata": {
                "suite_type": "error_cases",
                "item_count": len(provisioned_items),
            },
        }

    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Error test data provisioning failed: {e}"
        self.logger.error(error_msg)
        errors.append(error_msg)

        return {
            "success": False,
            "duration": duration,
            "provisioned_items": provisioned_items,
            "errors": errors,
            "metadata": {
                "suite_type": "error_cases",
                "failure_reason": str(e),
            },
        }
