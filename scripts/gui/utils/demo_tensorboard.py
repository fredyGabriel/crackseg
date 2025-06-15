"""Demo script for testing TensorBoard component functionality.

This script demonstrates the TensorBoard iframe embedding component
outside of the main application for testing purposes.
"""

from pathlib import Path

from scripts.gui.utils.tb_manager import create_tensorboard_manager


def demo_tensorboard_manager() -> None:
    """Demonstrate TensorBoard manager functionality."""
    print("🧪 TensorBoard Manager Demo")
    print("=" * 50)

    # Create manager
    manager = create_tensorboard_manager(preferred_port=6006)
    print(f"✅ Created TensorBoard manager: {id(manager)}")

    # Check initial state
    print(f"📊 Initial state: {manager.info.state.value}")
    print(f"🔌 Is running: {manager.is_running}")

    # Check available ports
    available_ports = manager.get_available_ports_in_range()
    print(f"🌐 Available ports: {available_ports[:5]}...")  # Show first 5

    # Demo log directory path
    demo_log_dir = Path("outputs/demo_logs/tensorboard")
    print(f"📂 Demo log directory: {demo_log_dir}")

    if demo_log_dir.exists():
        print("✅ Log directory exists - would start TensorBoard")

        # Demonstrate startup (commented to avoid actual startup in demo)
        # success = manager.start_tensorboard(demo_log_dir)
        # print(f"🚀 Startup success: {success}")
        #
        # if success:
        #     print(f"🔗 URL: {manager.get_url()}")
        #     print(f"🔌 Port: {manager.get_port()}")
        #
        #     # Cleanup
        #     manager.stop_tensorboard()
    else:
        print("❌ Log directory does not exist - create logs first")

    print("\n🎯 Demo completed!")


if __name__ == "__main__":
    demo_tensorboard_manager()
