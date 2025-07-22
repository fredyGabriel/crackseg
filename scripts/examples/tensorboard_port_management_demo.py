#!/usr/bin/env python3
"""
Demonstration of enhanced TensorBoard port management capabilities.
This script showcases the new dynamic port allocation, reservation
strategy, and conflict resolution features implemented in subtask 6.2.
Features demonstrated: - Port reservation and allocation tracking -
Global port registry management - Port release strategies - Factory
functions for creating managers Usage: python
scripts/examples/tensorboard_port_management_demo.py
"""

import sys
import time
from pathlib import Path

# Add project root to path for import s
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gui.utils.tb_manager import (
    PortAllocation,
    PortRegistry,
    TensorBoardManager,
    create_tensorboard_manager,
)


def demonstrate_port_allocation() -> None:
    """Demonstrate basic port allocation and tracking."""
    print("=" * 60)
    print("🔍 DEMONSTRATING PORT ALLOCATION & TRACKING")
    print("=" * 60)

    # Create multiple managers
    manager1 = TensorBoardManager()
    manager2 = TensorBoardManager()
    manager3 = TensorBoardManager()

    print("Created 3 TensorBoard managers:")
    print(f"  Manager 1: {id(manager1)}")
    print(f"  Manager 2: {id(manager2)}")
    print(f"  Manager 3: {id(manager3)}")

    # Show initial port state
    allocated_ports = PortRegistry.get_allocated_ports()
    print(f"\nInitial allocated ports: {allocated_ports}")

    # Show available ports in range
    available_ports = manager1.get_available_ports_in_range()
    print(f"Available ports in range (6006-6020): {available_ports[:5]}...")

    print("\n" + "-" * 40)


def demonstrate_registry_management() -> None:
    """Demonstrate global port registry management."""
    print("📊 DEMONSTRATING REGISTRY MANAGEMENT")
    print("=" * 60)

    # Create manager IDs for testing
    manager_id_1 = "test_manager_1"
    manager_id_2 = "test_manager_2"

    # Simulate port allocations
    port1 = 6010
    port2 = 6011

    success1 = PortRegistry.allocate_port(
        port1, manager_id_1, process_id=12345
    )
    success2 = PortRegistry.allocate_port(
        port2, manager_id_2, process_id=12346
    )

    # Show allocation info
    allocation1 = PortRegistry.get_allocation_info(port1)

    print("Allocation results:")
    print(f"  Port {port1}: {'✓ Success' if success1 else '✗ Failed'}")
    print(f"  Port {port2}: {'✓ Success' if success2 else '✗ Failed'}")

    if allocation1:
        print(f"\nPort {port1} allocation info:")
        print(f"  Manager ID: {allocation1.manager_id}")
        print(f"  Process ID: {allocation1.process_id}")
        print(f"  Reserved: {allocation1.reserved}")
        print(f"  Allocated at: {time.ctime(allocation1.allocated_at)}")

    # Test allocation queries
    print("\nAllocation queries:")
    print(f"  Is {port1} allocated: {PortRegistry.is_port_allocated(port1)}")
    print(f"  All allocated ports: {PortRegistry.get_allocated_ports()}")

    # Update process ID
    PortRegistry.update_process_id(port1, 54321, manager_id_1)
    updated_allocation = PortRegistry.get_allocation_info(port1)
    if updated_allocation:
        print(
            f"  Updated process ID for {port1}: "
            f"{updated_allocation.process_id}"
        )

    # Clean up
    PortRegistry.release_port(port1, manager_id_1)
    PortRegistry.release_port(port2, manager_id_2)

    print("\n" + "-" * 40)


def demonstrate_port_conflict_resolution() -> None:
    """Demonstrate port conflict resolution."""
    print("⚡ DEMONSTRATING PORT CONFLICT RESOLUTION")
    print("=" * 60)

    # Pre-allocate some ports to simulate conflicts
    PortRegistry.allocate_port(6006, "external_service")
    PortRegistry.allocate_port(6007, "another_service")
    PortRegistry.allocate_port(6008, "third_service")

    print("Pre-allocated ports 6006-6008 to simulate conflicts")
    print(f"Allocated ports: {PortRegistry.get_allocated_ports()}")

    # Create manager and show available ports after conflicts
    manager = TensorBoardManager()
    available = manager.get_available_ports_in_range()
    print(f"Available ports after conflicts: {available[:5]}...")

    print("\n🔧 Port discovery will automatically resolve conflicts")
    print("  • Multiple strategies: preferred → default → sequential → random")
    print("  • Automatic retry with alternative ports")
    print("  • Thread-safe allocation with global registry")

    # Clean up test allocations
    PortRegistry.force_release_port(6006)
    PortRegistry.force_release_port(6007)
    PortRegistry.force_release_port(6008)

    print("\n" + "-" * 40)


def demonstrate_stale_allocation_cleanup() -> None:
    """Demonstrate automatic cleanup of stale allocations."""
    print("🧹 DEMONSTRATING STALE ALLOCATION CLEANUP")
    print("=" * 60)

    # Create a test allocation
    port = 6015
    manager_id = "test_stale_manager"

    # Allocate port
    PortRegistry.allocate_port(port, manager_id)
    print(f"Allocated port {port}")

    # Show initial state
    allocation = PortRegistry.get_allocation_info(port)
    if allocation:
        print(f"Allocation time: {time.ctime(allocation.allocated_at)}")

    # Simulate old allocation by manually modifying time
    # (In real usage, this would happen after 5 minutes)
    old_time = time.time() - 400  # 400 seconds ago (> 5 minute timeout)

    if allocation:
        allocation.allocated_at = old_time
        print(f"Simulated old allocation time: {time.ctime(old_time)}")

    # Check if port is still considered allocated (should trigger cleanup)
    is_allocated = PortRegistry.is_port_allocated(port)
    print(f"Port {port} still allocated after timeout: {is_allocated}")

    # Verify cleanup worked
    final_allocation = PortRegistry.get_allocation_info(port)
    print(f"Allocation after cleanup: {final_allocation}")

    print("\n" + "-" * 40)


def demonstrate_factory_functions() -> None:
    """Demonstrate factory functions for creating managers."""
    print("🏭 DEMONSTRATING FACTORY FUNCTIONS")
    print("=" * 60)

    # Create manager with custom port range
    manager1 = create_tensorboard_manager(
        port_range=(6030, 6040), preferred_port=6035
    )

    print("Created manager with custom port range:")
    print("  Range: 6030-6040")
    print("  Preferred: 6035")

    # Show available ports in custom range
    available = manager1.get_available_ports_in_range()
    print(f"  Available ports: {available}")

    # Create another manager with different host
    create_tensorboard_manager(port_range=(6050, 6055), host="0.0.0.0")

    print("\nCreated manager with different host:")
    print("  Host: 0.0.0.0")
    print("  Range: 6050-6055")

    print("\n" + "-" * 40)


def demonstrate_manager_features() -> None:
    """Demonstrate TensorBoard manager features."""
    print("🔧 DEMONSTRATING MANAGER FEATURES")
    print("=" * 60)

    manager = TensorBoardManager()

    # Show manager capabilities
    print("Manager capabilities:")
    available_count = len(manager.get_available_ports_in_range())
    print(f"  Available ports in range: {available_count}")
    print(f"  Global allocated ports: {manager.get_allocated_ports()}")
    print(f"  Manager running status: {manager.is_running}")
    print(f"  Manager health status: {manager.is_healthy()}")

    # Test port queries
    test_port = 6012
    PortRegistry.allocate_port(test_port, "test_owner")

    port_info = manager.get_port_info(test_port)
    if port_info and isinstance(port_info, PortAllocation):
        print(f"\nPort {test_port} info:")
        print(f"  Owner: {port_info.manager_id}")
        print(f"  Reserved: {port_info.reserved}")

    # Test port ownership check
    is_owned = manager.is_port_allocated_by_this_manager(test_port)
    print(f"  Owned by this manager: {is_owned}")

    # Clean up
    PortRegistry.force_release_port(test_port)

    print("\n" + "-" * 40)


def main() -> None:
    """Run all demonstrations."""
    print("🚀 TENSORBOARD PORT MANAGEMENT DEMO")
    print("Showcasing Subtask 6.2: Dynamic Port Management")
    print(f"Timestamp: {time.ctime()}")
    print()

    try:
        demonstrate_port_allocation()
        demonstrate_registry_management()
        demonstrate_port_conflict_resolution()
        demonstrate_stale_allocation_cleanup()
        demonstrate_factory_functions()
        demonstrate_manager_features()

        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("\nKey Features Demonstrated:")
        print("  • Dynamic port discovery with conflict resolution")
        print("  • Global port registry with allocation tracking")
        print("  • Automatic cleanup of stale allocations")
        print("  • Thread-safe port management operations")
        print("  • Factory functions for customized managers")
        print("  • Port ownership and information queries")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
        raise

    finally:
        # Clean up any remaining allocations
        allocated = PortRegistry.get_allocated_ports()
        for port in allocated:
            PortRegistry.force_release_port(port)
        print(f"\n🧹 Cleaned up {len(allocated)} remaining port allocations")


if __name__ == "__main__":
    main()
