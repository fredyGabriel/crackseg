"""Demonstration of reactive results scanning with events and caching.

This demo showcases Phase 2 features:
- Event-driven reactive updates
- LRU cache for performance optimization
- Real-time progress monitoring
- Gallery-like UI simulation

Run this script to see the reactive scanning system in action.
"""

import asyncio
import logging
import time
from pathlib import Path

from .cache import TripletCache, get_triplet_cache
from .core import ResultTriplet
from .events import EventType, ScanEvent, get_event_manager
from .scanner import create_results_scanner

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ReactiveGallerySimulator:
    """Simulates a reactive gallery UI that responds to scanner events."""

    def __init__(self, cache: TripletCache) -> None:
        """Initialize the gallery simulator.

        Args:
            cache: Triplet cache for optimized access
        """
        self.cache = cache
        self.triplets: list[ResultTriplet] = []
        self.scan_stats = {
            "total_found": 0,
            "errors": 0,
            "scan_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self._scan_start_time = 0.0

    async def on_scan_started(self, event: ScanEvent) -> None:
        """Handle scan started event."""
        self._scan_start_time = time.time()
        base_path = event.data["base_path"]
        pattern = event.data["pattern"]

        print(f"\n🔍 Starting scan of {base_path}")
        print(f"   Pattern: {pattern}")
        print("   Gallery ready for real-time updates...")

    async def on_scan_progress(self, event: ScanEvent) -> None:
        """Handle scan progress event."""
        percent = event.data["percent"]
        found = event.data["found_triplets"]
        errors = event.data["errors"]

        # Update progress bar simulation
        bar_length = 30
        filled = int(bar_length * percent / 100)
        bar = "█" * filled + "░" * (bar_length - filled)

        print(
            f"\r📊 Progress: [{bar}] {percent:5.1f}% | "
            f"Found: {found:3d} | Errors: {errors}",
            end="",
        )

    async def on_triplet_found(self, event: ScanEvent) -> None:
        """Handle triplet found event."""
        triplet = event.data["triplet"]

        # Cache the triplet for fast access
        self.cache.cache_triplet(triplet)
        self.triplets.append(triplet)

        # Simulate gallery update
        if len(self.triplets) % 10 == 0:  # Update every 10 triplets
            print(
                f"\n✨ Gallery updated: {len(self.triplets)} triplets loaded"
            )

    async def on_scan_completed(self, event: ScanEvent) -> None:
        """Handle scan completed event."""
        total_triplets = event.data["total_triplets"]
        total_errors = event.data["total_errors"]
        success = event.data["success"]

        self.scan_stats.update(
            {
                "total_found": total_triplets,
                "errors": total_errors,
                "scan_time": time.time() - self._scan_start_time,
            }
        )

        status = "✅ SUCCESS" if success else "⚠️  WITH ERRORS"
        print(f"\n\n🎉 Scan completed {status}")
        print(f"   Total triplets: {total_triplets}")
        print(f"   Errors: {total_errors}")
        print(f"   Scan time: {self.scan_stats['scan_time']:.2f}s")

    async def on_scan_error(self, event: ScanEvent) -> None:
        """Handle scan error event."""
        error = event.data["error"]
        context = event.data["context"]

        print(f"\n❌ Scan error in {context}: {error}")

    def demonstrate_cache_access(self) -> None:
        """Demonstrate cache performance with repeated access."""
        if not self.triplets:
            print("No triplets to demonstrate cache with")
            return

        print("\n🚀 Cache Performance Demo")
        cache_stats = self.cache.get_stats()
        print(f"   Cache capacity: {cache_stats['max_size']}")

        # Access some triplets multiple times
        test_ids = [t.id for t in self.triplets[: min(5, len(self.triplets))]]

        print("   Testing repeated access...")
        start_time = time.time()

        for _ in range(3):  # 3 rounds of access
            for triplet_id in test_ids:
                cached_triplet = self.cache.get_triplet(triplet_id)
                if cached_triplet:
                    # Simulate some processing
                    _ = cached_triplet.is_complete

        access_time = time.time() - start_time
        stats = self.cache.get_stats()

        print(f"   Access time: {access_time:.3f}s")
        print(f"   Cache hit rate: {stats['hit_rate']:.1f}%")
        print(
            f"   Cache efficiency: {stats['hits']}/"
            f"{stats['hits'] + stats['misses']} hits"
        )

    def print_summary(self) -> None:
        """Print final summary of the demo."""
        cache_stats = self.cache.get_stats()

        print("\n📈 Demo Summary")
        print(f"   Triplets found: {len(self.triplets)}")
        print(f"   Scan time: {self.scan_stats['scan_time']:.2f}s")
        print(f"   Cache hit rate: {cache_stats['hit_rate']:.1f}%")
        print(f"   Cache fill rate: {cache_stats['fill_rate']:.1f}%")
        print(f"   Cached triplets: {cache_stats['cached_triplets']}")


async def run_reactive_demo(results_path: Path) -> None:
    """Run the reactive scanning demo.

    Args:
        results_path: Path to results directory to scan
    """
    print("🎬 Reactive Results Scanner Demo - Phase 2")
    print("=" * 50)

    # Initialize cache and event manager
    cache = get_triplet_cache(capacity=50, ttl_minutes=10)
    event_manager = get_event_manager()

    # Create gallery simulator
    gallery = ReactiveGallerySimulator(cache)

    # Subscribe to events
    event_manager.subscribe(EventType.SCAN_STARTED, gallery.on_scan_started)
    event_manager.subscribe(EventType.SCAN_PROGRESS, gallery.on_scan_progress)
    event_manager.subscribe(EventType.TRIPLET_FOUND, gallery.on_triplet_found)
    event_manager.subscribe(
        EventType.SCAN_COMPLETED, gallery.on_scan_completed
    )
    event_manager.subscribe(EventType.SCAN_ERROR, gallery.on_scan_error)

    # Create scanner with event manager
    scanner = create_results_scanner(
        results_path=results_path,
        max_concurrent=4,
        event_manager=event_manager,
    )

    try:
        # Run the scan
        triplet_count = 0
        async for _triplet in scanner.scan_async(
            pattern="*.png", recursive=True
        ):
            triplet_count += 1
            # Simulate some processing delay
            if triplet_count % 20 == 0:
                await asyncio.sleep(0.1)  # Brief pause for demo effect

        # Demonstrate cache performance
        gallery.demonstrate_cache_access()

        # Print final summary
        gallery.print_summary()

    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        logger.exception("Demo error")

    finally:
        # Cleanup
        event_manager.clear_handlers()
        cache.clear()


def create_sample_triplets(base_path: Path, count: int = 20) -> None:
    """Create sample triplet files for demo purposes.

    Args:
        base_path: Directory to create sample files in
        count: Number of triplets to create
    """
    print(f"📁 Creating {count} sample triplets in {base_path}")

    base_path.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        triplet_id = f"sample_{i:03d}"

        # Create dummy files
        (base_path / f"{triplet_id}_image.png").touch()
        (base_path / f"{triplet_id}_mask.png").touch()
        (base_path / f"{triplet_id}_prediction.png").touch()

    print(f"✅ Created {count * 3} sample files")


async def main() -> None:
    """Main demo function."""
    # Use a temporary directory for demo
    demo_path = Path("temp_demo_results")

    try:
        # Create sample data
        create_sample_triplets(demo_path, count=25)

        # Run the reactive demo
        await run_reactive_demo(demo_path)

    finally:
        # Cleanup sample data
        if demo_path.exists():
            import shutil

            shutil.rmtree(demo_path)
            print("\n🧹 Cleaned up demo files")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
