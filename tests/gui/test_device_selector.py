"""
Unit tests for the device selector component. Tests device detection,
information gathering, and UI component rendering for the CrackSeg
device selector.
"""

from typing import Any
from unittest.mock import Mock, patch

import torch

from gui.components.device_detector import DeviceDetector
from gui.components.device_info import DeviceInfo
from gui.components.device_selector import (
    OptimizedDeviceSelector,
    device_selector,
)


class TestDeviceInfo:
    """Test DeviceInfo class functionality."""

    def test_device_info_initialization(self) -> None:
        """Test DeviceInfo initialization with all parameters."""
        device_info = DeviceInfo(
            device_id="cuda:0",
            device_type="cuda",
            device_name="NVIDIA RTX 3070 Ti",
            memory_total=8.0,
            memory_available=6.5,
            compute_capability="8.6",
            is_available=True,
        )

        assert device_info.device_id == "cuda:0"
        assert device_info.device_type == "cuda"
        assert device_info.device_name == "NVIDIA RTX 3070 Ti"
        assert device_info.memory_total == 8.0
        assert device_info.memory_available == 6.5
        assert device_info.compute_capability == "8.6"
        assert device_info.is_available is True

    def test_device_info_minimal_initialization(self) -> None:
        """Test DeviceInfo initialization with minimal parameters."""
        device_info = DeviceInfo(
            device_id="cpu",
            device_type="cpu",
            device_name="CPU",
        )

        assert device_info.device_id == "cpu"
        assert device_info.device_type == "cpu"
        assert device_info.device_name == "CPU"
        assert device_info.memory_total is None
        assert device_info.memory_available is None
        assert device_info.compute_capability is None
        assert device_info.is_available is True

    def test_device_info_to_dict(self) -> None:
        """Test DeviceInfo to_dict method."""
        device_info = DeviceInfo(
            device_id="cuda:0",
            device_type="cuda",
            device_name="Test GPU",
            memory_total=8.0,
            memory_available=6.5,
            compute_capability="8.6",
            is_available=True,
        )

        expected_dict = {
            "device_id": "cuda:0",
            "device_type": "cuda",
            "device_name": "Test GPU",
            "memory_total": 8.0,
            "memory_available": 6.5,
            "compute_capability": "8.6",
            "is_available": True,
        }

        assert device_info.to_dict() == expected_dict


class TestDeviceDetector:
    """Test DeviceDetector class functionality."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_available_devices_cpu_only(
        self, mock_cuda_available: Mock
    ) -> None:
        """Test device detection when only CPU is available."""
        devices = DeviceDetector.get_available_devices()

        assert len(devices) == 1
        assert devices[0].device_id == "cpu"
        assert devices[0].device_type == "cpu"
        assert devices[0].device_name == "CPU"
        assert devices[0].is_available is True

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 3070 Ti")
    @patch("torch.cuda.memory_reserved", return_value=1024**3)  # 1GB
    def test_get_available_devices_with_cuda(
        self,
        mock_memory_reserved: Mock,
        mock_device_name: Mock,
        mock_device_props: Mock,
        mock_device_count: Mock,
        mock_cuda_available: Mock,
    ) -> None:
        """Test device detection with CUDA device available."""
        # Mock device properties
        mock_props = Mock()
        mock_props.total_memory = 8 * (1024**3)  # 8GB
        mock_props.major = 8
        mock_props.minor = 6
        mock_device_props.return_value = mock_props

        devices = DeviceDetector.get_available_devices()

        assert len(devices) == 2  # CPU + CUDA

        # Check CPU device
        cpu_device = next(d for d in devices if d.device_type == "cpu")
        assert cpu_device.device_id == "cpu"
        assert cpu_device.is_available is True

        # Check CUDA device
        cuda_device = next(d for d in devices if d.device_type == "cuda")
        assert cuda_device.device_id == "cuda:0"
        assert cuda_device.device_name == "NVIDIA RTX 3070 Ti"
        assert cuda_device.memory_total == 8.0
        assert cuda_device.memory_available == 7.0  # 8GB - 1GB reserved
        assert cuda_device.compute_capability == "8.6"
        assert cuda_device.is_available is True

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch(
        "torch.cuda.get_device_properties", side_effect=Exception("GPU Error")
    )
    @patch("torch.cuda.get_device_name", side_effect=Exception("GPU Error"))
    def test_get_available_devices_cuda_error(
        self,
        mock_device_name: Mock,
        mock_device_props: Mock,
        mock_device_count: Mock,
        mock_cuda_available: Mock,
    ) -> None:
        """Test device detection with CUDA error."""
        devices = DeviceDetector.get_available_devices()

        assert len(devices) == 2  # CPU + CUDA (with error)

        # Check CUDA device with error
        cuda_device = next(d for d in devices if d.device_type == "cuda")
        assert cuda_device.device_id == "cuda:0"
        assert cuda_device.device_name == "GPU 0"
        assert not cuda_device.is_available

    @patch("torch.cuda.is_available", return_value=True)
    @patch.object(DeviceDetector, "get_available_devices")
    def test_get_recommended_device_cuda_preferred(
        self, mock_get_devices: Mock, mock_cuda_available: Mock
    ) -> None:
        """Test recommended device selection prefers CUDA with most memory."""
        mock_devices = [
            DeviceInfo("cpu", "cpu", "CPU", is_available=True),
            DeviceInfo(
                "cuda:0",
                "cuda",
                "GPU 0",
                memory_total=4.0,
                memory_available=4.0,
                is_available=True,
            ),
            DeviceInfo(
                "cuda:1",
                "cuda",
                "GPU 1",
                memory_total=8.0,
                memory_available=8.0,
                is_available=True,
            ),
        ]
        mock_get_devices.return_value = mock_devices

        recommended = DeviceDetector.get_recommended_device()
        assert recommended == "cuda:1"  # Should prefer GPU with more memory

    @patch("torch.cuda.is_available", return_value=False)
    @patch.object(DeviceDetector, "get_available_devices")
    def test_get_recommended_device_cpu_fallback(
        self, mock_get_devices: Mock, mock_cuda_available: Mock
    ) -> None:
        """Test recommended device falls back to CPU when no CUDA."""
        mock_devices = [
            DeviceInfo("cpu", "cpu", "CPU", is_available=True),
        ]
        mock_get_devices.return_value = mock_devices

        recommended = DeviceDetector.get_recommended_device()
        assert recommended == "cpu"


class TestOptimizedDeviceSelector:
    """Test OptimizedDeviceSelector class functionality."""

    def test_css_content_not_empty(self) -> None:
        """Test that CSS content is defined and not empty."""
        css_content = OptimizedDeviceSelector._CSS_CONTENT
        assert css_content is not None
        assert len(css_content.strip()) > 0
        assert "crackseg-device-selector" in css_content

    def test_brand_colors_defined(self) -> None:
        """Test that brand colors are properly defined."""
        colors = OptimizedDeviceSelector._BRAND_COLORS
        required_colors = [
            "primary",
            "secondary",
            "accent",
            "success",
            "warning",
            "error",
            "info",
            "cpu",
            "cuda",
        ]

        for color in required_colors:
            assert color in colors
            assert colors[color].startswith("#")
            assert len(colors[color]) == 7  # Format: #RRGGBB

    def test_build_device_card_html_basic(self) -> None:
        """Test HTML generation for basic device card."""
        device = DeviceInfo("cpu", "cpu", "CPU", is_available=True)

        html = OptimizedDeviceSelector._build_device_card_html(device)

        assert "cpu" in html
        assert "CPU" in html
        assert "crackseg-device-card" in html
        assert 'data-device-id="cpu"' in html

    def test_build_device_card_html_cuda_with_memory(self) -> None:
        """Test HTML generation for CUDA device with memory info."""
        device = DeviceInfo(
            "cuda:0",
            "cuda",
            "NVIDIA RTX 3070 Ti",
            memory_total=8.0,
            memory_available=6.5,
            compute_capability="8.6",
            is_available=True,
        )

        html = OptimizedDeviceSelector._build_device_card_html(device)

        assert "cuda:0" in html
        assert "NVIDIA RTX 3070 Ti" in html
        assert "8.0 GB" in html
        assert "6.5 GB" in html
        assert "8.6" in html

    def test_build_device_card_html_selected_recommended(self) -> None:
        """Test HTML generation for selected and recommended device."""
        device = DeviceInfo("cuda:0", "cuda", "Test GPU", is_available=True)

        html = OptimizedDeviceSelector._build_device_card_html(
            device, is_selected=True, is_recommended=True
        )

        assert "selected" in html
        assert "recommended" in html
        assert "Recommended" in html
        assert "Selected" in html

    @patch("streamlit.session_state", new_callable=dict)
    @patch("streamlit.markdown")
    @patch("streamlit.selectbox")
    @patch.object(OptimizedDeviceSelector, "_ensure_css_injected")
    @patch.object(DeviceDetector, "get_available_devices")
    @patch.object(
        DeviceDetector, "get_recommended_device", return_value="cuda:0"
    )
    def test_render_device_selector_basic(
        self,
        mock_get_recommended: Mock,
        mock_get_devices: Mock,
        mock_ensure_css: Mock,
        mock_selectbox: Mock,
        mock_markdown: Mock,
        mock_session_state: dict[str, Any],
    ) -> None:
        """Test basic device selector rendering."""
        mock_devices = [
            DeviceInfo("cpu", "cpu", "CPU", is_available=True),
            DeviceInfo("cuda:0", "cuda", "Test GPU", is_available=True),
        ]
        mock_get_devices.return_value = mock_devices

        # Mock the selectbox to return the expected device
        mock_selectbox.return_value = "Test GPU (cuda:0) [Recommended]"

        # Set up session state with a default value
        mock_session_state["selected_device"] = "cuda:0"

        result = OptimizedDeviceSelector.render_device_selector()

        assert result == "cuda:0"
        mock_ensure_css.assert_called_once()
        assert (
            mock_markdown.call_count >= 2
        )  # Called for header and device cards (may be more in modular imp.)
        mock_selectbox.assert_called_once()

    @patch("gui.components.device_selector.get_device")
    def test_get_device_from_selection_valid(
        self, mock_get_device: Mock
    ) -> None:
        """Test converting valid device selection to PyTorch device."""
        mock_device = torch.device("cuda:0")
        mock_get_device.return_value = mock_device

        result = OptimizedDeviceSelector.get_device_from_selection("cuda:0")

        assert result == mock_device
        mock_get_device.assert_called_once_with("cuda:0")

    @patch(
        "gui.components.device_selector.get_device",
        side_effect=Exception("Device Error"),
    )
    def test_get_device_from_selection_error(
        self, mock_get_device: Mock
    ) -> None:
        """Test error handling when converting device selection."""
        result = OptimizedDeviceSelector.get_device_from_selection(
            "invalid:device"
        )

        assert result == torch.device("cpu")  # Should fallback to CPU


class TestDeviceSelectorConvenienceFunction:
    """Test the convenience function for device selector."""

    @patch.object(OptimizedDeviceSelector, "render_device_selector")
    def test_device_selector_convenience_function(
        self, mock_render: Mock
    ) -> None:
        """Test convenience function delegates to main class."""
        mock_render.return_value = "cuda:0"

        result = device_selector(
            selected_device="cpu",
            component_id="test_selector",
            session_key="test_device",
            show_title=False,
        )

        assert result == "cuda:0"
        mock_render.assert_called_once_with(
            selected_device="cpu",
            component_id="test_selector",
            session_key="test_device",
            show_title=False,
        )

    @patch.object(OptimizedDeviceSelector, "render_device_selector")
    def test_device_selector_default_parameters(
        self, mock_render: Mock
    ) -> None:
        """Test convenience function with default parameters."""
        mock_render.return_value = "cpu"

        result = device_selector()

        assert result == "cpu"
        mock_render.assert_called_once_with(
            selected_device=None,
            component_id="device_selector",
            session_key="selected_device",
            show_title=True,
        )
