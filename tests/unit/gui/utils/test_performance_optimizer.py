import time
from unittest.mock import MagicMock, patch

from gui.utils.performance_optimizer import (
    PerformanceOptimizer,
    get_optimizer,
    track_performance,
)


def test_singleton_pattern() -> None:
    """Test singleton implementation."""
    opt1 = PerformanceOptimizer()
    opt2 = get_optimizer()
    assert opt1 is opt2


def test_inject_css_once() -> None:
    """Test CSS is injected only once."""
    opt = PerformanceOptimizer()
    with patch("streamlit.markdown") as mock_markdown:
        opt.inject_css_once("test_key", "<style>.test {}</style>")
        opt.inject_css_once("test_key", "<style>.test {}</style>")
        mock_markdown.assert_called_once()


def test_should_update_debounce() -> None:
    """Test update debouncing."""
    assert PerformanceOptimizer().should_update("test_comp", 0.1)
    assert not PerformanceOptimizer().should_update("test_comp", 0.1)
    time.sleep(0.2)
    assert PerformanceOptimizer().should_update("test_comp", 0.1)


def test_track_performance() -> None:
    """Test performance tracking."""
    opt = PerformanceOptimizer()
    start = time.time()
    track_performance("comp1", "op1", start)
    report = opt.get_performance_report()
    assert "comp1" in report
    assert report["comp1"]["count"] == 1
    assert "average_time" in report["comp1"]


def test_memory_manager_track() -> None:
    """Test memory usage tracking."""
    from gui.utils.performance_optimizer import MemoryManager

    MemoryManager.track_memory_usage("comp2", "alloc", 100.0)
    report = MemoryManager.get_memory_report()
    assert report["comp2"]["total_allocated"] == 100.0
    assert report["comp2"]["peak_usage"] == 100.0


def test_cleanup_component() -> None:
    """Test component cleanup."""
    from gui.utils.performance_optimizer import MemoryManager

    callback = MagicMock()
    MemoryManager.register_cleanup_callback("temp_comp", callback)
    MemoryManager.cleanup_component("temp_comp")
    callback.assert_called_once()
    assert "temp_comp" not in MemoryManager.get_memory_report()
