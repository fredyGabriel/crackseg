"""
Examples demonstrating real-time log streaming functionality. This
module provides practical examples of how to use the log streaming
system for real-time training monitoring and GUI integration.
"""

from .run_manager import (
    LogLevel,
    StreamedLog,
    add_log_callback,
    clear_log_buffer,
    get_recent_logs,
    get_streaming_status,
)


def example_basic_log_callback() -> None:
    """
    Example: Basic log callback for console output. Demonstrates how to
    register a simple callback that prints all log entries to the console
    with timestamps and levels.
    """
    print("=== Basic Log Callback Example ===")

    def simple_log_handler(log: StreamedLog) -> None:
        """Simple callback that prints logs to console."""
        timestamp_str = log.timestamp.strftime("%H:%M:%S")
        level_str = log.level.value.ljust(8)
        source_str = log.source.ljust(12)
        print(f"[{timestamp_str}] [{level_str}] [{source_str}] {log.content}")

    # Register the callback
    add_log_callback(simple_log_handler)
    print("✓ Log callback registered")

    # The callback will now receive all logs from crackseg.training processes
    # Remove when done:
    # remove_log_callback(simple_log_handler)


def example_filtered_log_callback() -> None:
    """
    Example: Filtered log callback for specific log levels. Shows how to
    create a callback that only processes certain types of log entries
    (e.g., only errors and warnings).
    """
    print("\n=== Filtered Log Callback Example ===")

    def error_warning_handler(log: StreamedLog) -> None:
        """Callback that only processes errors and warnings."""
        if log.level in {LogLevel.ERROR, LogLevel.WARNING, LogLevel.CRITICAL}:
            print(f"🚨 [{log.level.value}] {log.content}")

            # Could also trigger GUI notifications, send alerts, etc.
            if log.level == LogLevel.ERROR:
                print("   ↳ Consider checking training parameters")

    add_log_callback(error_warning_handler)
    print("✓ Error/warning filter callback registered")


def example_metrics_extraction_callback() -> None:
    """
    Example: Extract training metrics from log streams. Demonstrates
    parsing log content to extract numerical metrics like loss, accuracy,
    epoch progress, etc.
    """
    print("\n=== Metrics Extraction Example ===")

    import re

    # Pattern to match common training metrics
    metrics_pattern = re.compile(
        r"(loss|accuracy|iou|dice|lr|learning_rate)[:\s=]+([0-9]+\.?[0-9]*)",
        re.IGNORECASE,
    )

    extracted_metrics: dict[str, list[float]] = {}

    def metrics_extractor(log: StreamedLog) -> None:
        """Extract numerical metrics from log content."""
        if log.source == "stdout":  # Focus on stdout for training metrics
            matches = metrics_pattern.findall(log.content)

            for metric_name, metric_value in matches:
                metric_name = metric_name.lower()
                try:
                    value = float(metric_value)

                    if metric_name not in extracted_metrics:
                        extracted_metrics[metric_name] = []

                    extracted_metrics[metric_name].append(value)

                    # Print recent metrics
                    recent_values = extracted_metrics[metric_name][-5:]
                    avg_recent = sum(recent_values) / len(recent_values)
                    msg = f"📊 {metric_name}: {value:.4f} (avg: {avg_recent:.4f}) "  # noqa E501
                    print(msg)

                except ValueError:
                    continue

    add_log_callback(metrics_extractor)
    print("✓ Metrics extraction callback registered")
    print("   Will extract loss, accuracy, IoU, Dice, learning rate from logs")


def example_gui_integration_callback() -> None:
    """
    Example: GUI integration with log buffering. Shows how to implement a
    callback suitable for GUI frameworks that need thread-safe updates and
    buffered display.
    """
    print("\n=== GUI Integration Example ===")

    # Simulate GUI log buffer (in real GUI, this would be a widget/component)
    gui_log_buffer: list[dict[str, str]] = []
    max_gui_logs = 100

    def gui_log_handler(log: StreamedLog) -> None:
        """GUI-friendly callback with buffering and formatting."""
        # Format for GUI display
        gui_entry = {
            "timestamp": log.timestamp.strftime("%H:%M:%S.%f")[:-3],
            "level": log.level.value,
            "source": log.source,
            "content": log.content,
            "color": _get_log_color(log.level),
        }

        # Add to GUI buffer
        gui_log_buffer.append(gui_entry)

        # Maintain buffer size
        if len(gui_log_buffer) > max_gui_logs:
            gui_log_buffer.pop(0)

        # In real GUI, you would:
        # - Use thread-safe update mechanism (Qt signals, tkinter, etc.)
        # - Update log display widget
        # - Auto-scroll to bottom
        # - Apply syntax highlighting

        print(f"GUI: [{gui_entry['timestamp']}] {gui_entry['content']}")

    def _get_log_color(level: LogLevel) -> str:
        """Get color for log level (for GUI highlighting)."""
        color_map = {
            LogLevel.ERROR: "#FF4444",
            LogLevel.WARNING: "#FFA500",
            LogLevel.INFO: "#000000",
            LogLevel.DEBUG: "#888888",
            LogLevel.CRITICAL: "#FF0000",
            LogLevel.UNKNOWN: "#666666",
        }
        return color_map.get(level, "#000000")

    add_log_callback(gui_log_handler)
    print("✓ GUI integration callback registered")
    print(f"   Buffer size: {max_gui_logs} entries with color coding")


def example_log_monitoring_and_status() -> None:
    """
    Example: Monitor streaming status and retrieve logs. Demonstrates how
    to check streaming status and retrieve buffered logs for analysis or
    display.
    """
    print("\n=== Log Monitoring Example ===")

    # Check current streaming status
    status = get_streaming_status()
    print("Current streaming status:")
    print(f"  • Streaming active: {status['is_streaming']}")
    print(
        f"  • Buffer size: {status['buffer_size']}/{status['max_buffer_size']}"
    )
    print(f"  • Total logs processed: {status['total_logs_processed']}")
    print(f"  • Active callbacks: {status['active_callbacks']}")
    print(f"  • Stdout reader: {status.get('stdout_reader_active', False)}")
    print(f"  • Hydra watcher: {status.get('hydra_watcher_active', False)}")

    # Get recent logs
    recent_logs = get_recent_logs(10)  # Last 10 logs
    if recent_logs:
        print(f"\nLast {len(recent_logs)} log entries:")
        for i, log in enumerate(recent_logs[-5:], 1):  # Show last 5
            print(f"  {i}. [{log.level.value}] {log.content[:50]}...")
    else:
        print("\nNo logs in buffer yet")

    # Clear buffer if needed
    if status["buffer_size"] > 0:
        print(f"\nClearing log buffer ({status['buffer_size']} entries)")
        clear_log_buffer()
        print("✓ Log buffer cleared")


def example_training_session_with_streaming() -> None:
    """
    Example: Complete training session with streaming. Shows how to start
    a training session and monitor it with real-time log streaming
    (simulation).
    """
    print("\n=== Complete Training Session Example ===")

    # Register comprehensive monitoring
    def comprehensive_monitor(log: StreamedLog) -> None:
        """Monitor all aspects of training."""
        prefix = f"[{log.timestamp.strftime('%H:%M:%S')}] [{log.level.value}] "

        if "epoch" in log.content.lower():
            print(f"🏃 {prefix}EPOCH: {log.content}")
        elif "loss" in log.content.lower():
            print(f"📉 {prefix}LOSS: {log.content}")
        elif "error" in log.content.lower():
            print(f"❌ {prefix}ERROR: {log.content}")
        elif log.source.startswith("hydra:"):
            print(f"📁 {prefix}HYDRA: {log.content}")
        else:
            print(f"ℹ️  {prefix}{log.content}")

    add_log_callback(comprehensive_monitor)

    # Simulate training session (in real use, you'd have actual config)
    # config_path = Path("configs")  # This would be your actual config path
    # config_name = "train_baseline"  # Your actual config name

    print("📋 Starting training session monitoring...")
    print("   (This is a simulation - no actual training will start)")
    print("   In real usage, you would:")
    print("   1. Call start_training_session() with valid config")
    print("   2. Monitor logs in real-time via callbacks")
    print("   3. Stop training with stop_training_session()")

    # In real usage:
    # success, errors = start_training_session(config_path, config_name)
    # if success:
    #     print("✓ Training started successfully")
    #     # Monitor via callbacks...
    #     # stop_training_session()
    # else:
    #     print(f"❌ Failed to start training: {errors}")


def run_all_examples() -> None:
    """Run all streaming examples in sequence."""
    print("🚀 Running Real-Time Log Streaming Examples")
    print("=" * 60)

    # Register various callbacks
    example_basic_log_callback()
    example_filtered_log_callback()
    example_metrics_extraction_callback()
    example_gui_integration_callback()

    # Monitor status
    example_log_monitoring_and_status()

    # Complete workflow
    example_training_session_with_streaming()

    print("\n" + "=" * 60)
    print("✨ All examples completed!")
    print("\nTo use in your GUI:")
    print("1. Register callbacks with add_log_callback()")
    print("2. Start training with start_training_session()")
    print("3. Logs will stream to your callbacks in real-time")
    print("4. Check status with get_streaming_status()")
    print("5. Stop training with stop_training_session()")


if __name__ == "__main__":
    run_all_examples()
