"""Results Gallery Module.

This module provides components for scanning, validating, and processing
training results for the crack segmentation project with reactive updates
and caching capabilities.

Components:
    - core: Data structures and types for results handling
    - validation: Basic validation logic for result triplets
    - advanced_validation: Enhanced validation with integrity checks
    - scanner: AsyncIO-based results scanning system with event emission
    - events: Pub-sub event system for reactive gallery updates
    - cache: LRU cache for optimizing triplet access
    - demo: Demonstration scripts for scanner and validation functionality

Phase 3: Enhanced validation + comprehensive error handling.
"""

from .advanced_validation import (
    AdvancedTripletValidator,
    CorruptionError,
    IntegrityError,
    ValidationError,
    ValidationLevel,
    ValidationResult,
    ValidationStats,
)
from .cache import (
    LRUCache,
    TripletCache,
    get_triplet_cache,
    reset_triplet_cache,
)
from .core import (
    ResultTriplet,
    ScanProgress,
    TripletHealth,
    TripletType,
)
from .events import (
    EventManager,
    EventType,
    ScanEvent,
    get_event_manager,
    reset_event_manager,
)
from .scanner import AsyncResultsScanner, create_results_scanner
from .validation import TripletValidator

__all__ = [
    # Core data structures
    "ResultTriplet",
    "ScanProgress",
    "TripletType",
    "TripletHealth",
    # Main components
    "AsyncResultsScanner",
    "create_results_scanner",
    "TripletValidator",
    # Advanced validation
    "AdvancedTripletValidator",
    "ValidationError",
    "CorruptionError",
    "IntegrityError",
    "ValidationLevel",
    "ValidationResult",
    "ValidationStats",
    # Event system
    "EventManager",
    "EventType",
    "ScanEvent",
    "get_event_manager",
    "reset_event_manager",
    # Caching system
    "LRUCache",
    "TripletCache",
    "get_triplet_cache",
    "reset_triplet_cache",
]
