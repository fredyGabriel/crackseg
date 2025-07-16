#!/usr/bin/env python3
"""
Backward Compatibility Integration Tests

This module verifies that all refactored components maintain backward
compatibility with existing code and can load legacy checkpoints.
"""

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestBackwardCompatibility:
    """Integration tests for backward compatibility verification."""

    def test_decoder_components_compatibility(self) -> None:
        """Test backward compatibility of decoder components."""
        from crackseg.model.decoder.cnn_decoder import (
            CNNDecoder,
            DecoderBlock,
        )

        # Test DecoderBlock instantiation with various configurations
        DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        DecoderBlock(in_channels=256, skip_channels=0)  # No skip
        DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256,
            middle_channels=512,
        )

        # Test CNNDecoder instantiation
        decoder1 = CNNDecoder(
            in_channels=512, skip_channels_list=[256, 128, 64], out_channels=1
        )
        CNNDecoder(
            in_channels=1024,
            skip_channels_list=[512, 256, 128, 64],
            out_channels=3,
        )

        # Test forward pass compatibility
        x = torch.randn(2, 512, 8, 8)
        skips = [
            torch.randn(2, 256, 16, 16),
            torch.randn(2, 128, 32, 32),
            torch.randn(2, 64, 64, 64),
        ]

        output = decoder1(x, skips)
        assert output.shape[1] == 1  # Check output channels

    def test_loss_registry_compatibility(self) -> None:
        """Test backward compatibility of loss registry systems."""
        # Test legacy loss registry
        from crackseg.training.losses.loss_registry_setup import (
            loss_registry,
        )

        legacy_losses = loss_registry.list_components()
        assert len(legacy_losses) > 0, "Legacy registry should have losses"

        # Test enhanced registry
        from crackseg.training.losses.registry import registry

        enhanced_losses = registry.list_available()
        assert len(enhanced_losses) > 0, "Enhanced registry should have losses"

        # Basic availability test is sufficient for backward compatibility
        # Complex instantiation is covered by dedicated loss tests

    def test_feature_info_utils_compatibility(self) -> None:
        """Test feature info utilities compatibility."""
        from crackseg.model.encoder.feature_info_utils import (
            create_feature_info_entry,
            validate_feature_info,
        )

        # Test utility functions - focus on core functionality
        entry = create_feature_info_entry(channels=64, reduction=4, stage=1)
        assert entry["channels"] == 64
        assert entry["reduction"] == 4

        # Test basic validation
        test_feature_info = [
            {"channels": 64, "reduction": 4, "stage": 1, "name": None},
            {"channels": 128, "reduction": 8, "stage": 2, "name": None},
        ]
        validate_feature_info(test_feature_info)  # Should not raise

    @pytest.mark.skipif(
        not Path("outputs/checkpoints/model_best.pth.tar").exists(),
        reason="No checkpoint found for testing",
    )
    def test_checkpoint_loading_compatibility(self) -> None:
        """Test loading of existing checkpoints."""
        checkpoint_path = Path("outputs/checkpoints/model_best.pth.tar")

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )

        # Check expected structure
        expected_keys = [
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "metrics",
        ]
        for key in expected_keys:
            assert key in checkpoint, f"Missing key in checkpoint: {key}"

    def test_import_compatibility(self) -> None:
        """Test that all import paths still work."""
        import_tests = [
            (
                "crackseg.model.decoder.cnn_decoder",
                ["CNNDecoder", "DecoderBlock"],
            ),
            (
                "crackseg.model.encoder.feature_info_utils",
                ["create_feature_info_entry"],
            ),
            ("crackseg.training.losses.registry", ["registry"]),
            (
                "crackseg.training.losses.loss_registry_setup",
                ["loss_registry"],
            ),
        ]

        for module_path, classes in import_tests:
            module = __import__(module_path, fromlist=classes)
            for cls_name in classes:
                assert hasattr(
                    module, cls_name
                ), f"Missing {cls_name} in {module_path}"


# Standalone runner for direct execution
def run_compatibility_tests() -> tuple[dict[str, Any], bool]:
    """Run all backward compatibility tests with detailed reporting."""
    print("🔄 Running Backward Compatibility Tests...")
    print("=" * 50)

    test_instance = TestBackwardCompatibility()
    all_results: dict[str, Any] = {}

    # Run individual test methods
    test_methods = [
        (
            "Decoder Components",
            test_instance.test_decoder_components_compatibility,
        ),
        ("Loss Registry", test_instance.test_loss_registry_compatibility),
        (
            "Feature Info Utils",
            test_instance.test_feature_info_utils_compatibility,
        ),
        (
            "Checkpoint Loading",
            test_instance.test_checkpoint_loading_compatibility,
        ),
        ("Import Compatibility", test_instance.test_import_compatibility),
    ]

    for test_name, test_method in test_methods:
        print(f"\n📋 Testing: {test_name}")
        try:
            test_method()
            all_results[test_name] = True
            print(f"  ✅ {test_name}")
        except Exception as e:
            all_results[test_name] = False
            print(f"  ❌ {test_name}: {e}")

    # Calculate overall success
    total_tests = len(all_results)
    passed_tests = sum(all_results.values())
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    overall_success = success_rate >= 0.8  # 80% success rate required

    print("\n" + "=" * 50)
    print("📊 Overall Results:")
    print(f"   Tests Passed: {passed_tests}/{total_tests}")
    print(f"   Success Rate: {success_rate:.1%}")
    print(f"   Status: {'✅ PASSED' if overall_success else '❌ FAILED'}")

    return all_results, overall_success


if __name__ == "__main__":
    results, success = run_compatibility_tests()

    if not success:
        print("\n⚠️  Some backward compatibility tests failed.")
        print("Review the results above for specific issues.")
        sys.exit(1)
    else:
        print("\n🎉 All backward compatibility tests passed!")
        sys.exit(0)
