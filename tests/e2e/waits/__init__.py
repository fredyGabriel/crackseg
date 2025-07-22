"""
Wait strategy module for reliable E2E testing. This module provides
comprehensive wait strategies including explicit waits, custom wait
conditions, and smart wait strategies for reliable element
interactions across different browsers and test scenarios. Key
Components: - WaitStrategy: Main wait strategy orchestrator -
CustomConditions: Streamlit-specific wait conditions - FluentWait:
Configurable polling and timeout management - SmartWait: Context-aware
wait selection Examples: >>> from tests.e2e.waits import WaitStrategy,
StreamlitConditions >>> wait_strategy = WaitStrategy(driver,
timeout=15.0) >>> wait_strategy.until(StreamlitConditions.app_ready())
>>> element = wait_strategy.for_element_clickable(locator)
"""

from .conditions import (
    CustomConditions,
    StreamlitConditions,
    element_attribute_contains,
    element_count_equals,
    element_text_matches,
    text_to_be_present_in_element_value,
)
from .strategies import FluentWaitConfig, SmartWait, WaitStrategy

__all__ = [
    # Main strategy classes
    "WaitStrategy",
    "SmartWait",
    "FluentWaitConfig",
    # Condition classes
    "CustomConditions",
    "StreamlitConditions",
    # Individual condition functions
    "element_text_matches",
    "element_count_equals",
    "element_attribute_contains",
    "text_to_be_present_in_element_value",
]
