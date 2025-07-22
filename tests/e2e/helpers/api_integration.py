"""
API integration utilities for E2E testing. This module provides
utilities for testing API interactions, health checks, and load
simulation.
"""

import importlib.util
import logging
import time
from typing import Any, TypedDict
from urllib.parse import urljoin

# Runtime availability check using importlib
_requests_available = importlib.util.find_spec("requests") is not None

logger = logging.getLogger(__name__)


class APIResponse(TypedDict):
    """Type definition for API response data."""

    status_code: int
    response_time: float
    content: Any
    headers: dict[str, str]
    success: bool
    error: str | None


class APIEndpoint(TypedDict):
    """Type definition for API endpoint configuration."""

    url: str
    method: str
    headers: dict[str, str] | None
    expected_status: int
    timeout: float


class APITestHelper:
    """Comprehensive API testing helper with retry mechanisms."""

    def __init__(
        self, base_url: str = "http://localhost:8501", timeout: float = 30.0
    ) -> None:
        """Initialize API test helper.

        Args:
            base_url: Base URL for API requests
            timeout: Default timeout for requests
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = self._create_session() if _requests_available else None
        self.logger = logging.getLogger(f"{__name__}.APITestHelper")

    def _create_session(self) -> Any:
        """Create a requests session with retry configuration."""
        if not _requests_available:
            return None

        # Import here to avoid circular imports and unused import warnings
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def make_request(
        self,
        endpoint: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        data: Any = None,
        params: dict[str, Any] | None = None,
    ) -> APIResponse:
        """Make an HTTP request with comprehensive error handling.

        Args:
            endpoint: API endpoint path
            method: HTTP method
            headers: Request headers
            data: Request data
            params: Query parameters

        Returns:
            APIResponse with request results
        """
        if not _requests_available:
            return {
                "status_code": 0,
                "response_time": 0.0,
                "content": None,
                "headers": {},
                "success": False,
                "error": "requests library not available",
            }

        # Import here to avoid unused import warnings
        import requests

        url = urljoin(self.base_url, endpoint)
        start_time = time.time()

        try:
            if self.session is None:
                raise RuntimeError(
                    "Session not available - requests library not installed"
                )

            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                json=data if method in ["POST", "PUT", "PATCH"] else None,
                data=data if method not in ["POST", "PUT", "PATCH"] else None,
                params=params,
                timeout=self.timeout,
            )

            response_time = time.time() - start_time

            return {
                "status_code": response.status_code,
                "response_time": response_time,
                "content": (
                    response.json()
                    if response.headers.get("content-type", "").startswith(
                        "application/json"
                    )
                    else response.text
                ),
                "headers": dict(response.headers),
                "success": 200 <= response.status_code < 300,
                "error": None,
            }

        except requests.exceptions.RequestException as e:
            response_time = time.time() - start_time
            return {
                "status_code": 0,
                "response_time": response_time,
                "content": None,
                "headers": {},
                "success": False,
                "error": str(e),
            }

    def test_endpoint_health(
        self, endpoint: str = "/", expected_status: int = 200
    ) -> bool:
        """Test endpoint health with retry logic.

        Args:
            endpoint: Endpoint to test
            expected_status: Expected HTTP status code

        Returns:
            True if endpoint is healthy, False otherwise
        """
        response = self.make_request(endpoint)
        return (
            response["success"] and response["status_code"] == expected_status
        )

    def simulate_load_test(
        self,
        endpoint: str,
        requests_count: int = 10,
        concurrent_users: int = 2,
    ) -> dict[str, Any]:
        """Simulate load testing on an endpoint.

        Args:
            endpoint: Endpoint to test
            requests_count: Total number of requests
            concurrent_users: Number of concurrent users

        Returns:
            Load test results
        """
        if not _requests_available:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "average_response_time": 0.0,
                "max_response_time": 0.0,
                "min_response_time": 0.0,
                "error": "requests library not available",
            }

        import threading

        responses: list[APIResponse] = []
        response_lock = threading.Lock()

        def make_requests():
            for _ in range(requests_count // concurrent_users):
                response = self.make_request(endpoint)
                with response_lock:
                    responses.append(response)
                time.sleep(0.1)  # Small delay between requests

        threads = []
        for _ in range(concurrent_users):
            thread = threading.Thread(target=make_requests)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Calculate statistics
        successful_requests = sum(1 for r in responses if r["success"])
        failed_requests = len(responses) - successful_requests
        response_times = [
            r["response_time"] for r in responses if r["response_time"] > 0
        ]

        return {
            "total_requests": len(responses),
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "average_response_time": (
                sum(response_times) / len(response_times)
                if response_times
                else 0.0
            ),
            "max_response_time": (
                max(response_times) if response_times else 0.0
            ),
            "min_response_time": (
                min(response_times) if response_times else 0.0
            ),
            "error": None,
        }

    def validate_api_response(
        self,
        response: APIResponse,
        expected_schema: dict[str, Any] | None = None,
    ) -> bool:
        """Validate API response structure and content.

        Args:
            response: API response to validate
            expected_schema: Expected response schema

        Returns:
            True if response is valid, False otherwise
        """
        if not response["success"]:
            return False

        if expected_schema and isinstance(response["content"], dict):
            # Basic schema validation
            for key, expected_type in expected_schema.items():
                if key not in response["content"]:
                    return False
                if not isinstance(response["content"][key], expected_type):
                    return False

        return True


def verify_streamlit_health(base_url: str = "http://localhost:8501") -> bool:
    """Verify Streamlit application health.

    Args:
        base_url: Streamlit application URL

    Returns:
        True if application is healthy, False otherwise
    """
    helper = APITestHelper(base_url)
    return helper.test_endpoint_health("/")


def test_crackseg_endpoints(
    base_url: str = "http://localhost:8501",
) -> dict[str, bool]:
    """Test CrackSeg application endpoints.

    Args:
        base_url: Application base URL

    Returns:
        Dictionary with endpoint test results
    """
    helper = APITestHelper(base_url)
    results: dict[str, bool] = {}

    # Test main application endpoints
    endpoints = [
        ("/", "Main page"),
        ("/_stcore/health", "Health check"),
        ("/_stcore/static", "Static files"),
    ]

    for endpoint, description in endpoints:
        results[description] = helper.test_endpoint_health(endpoint)

    return results


def simulate_api_load(
    base_url: str = "http://localhost:8501",
    duration_seconds: float = 30.0,
    requests_per_second: float = 2.0,
) -> dict[str, Any]:
    """Simulate API load testing.

    Args:
        base_url: Application base URL
        duration_seconds: Test duration in seconds
        requests_per_second: Requests per second

    Returns:
        Load test results
    """
    helper = APITestHelper(base_url)
    total_requests = int(duration_seconds * requests_per_second)

    return helper.simulate_load_test("/", total_requests, 2)


def validate_api_responses(responses: list[APIResponse]) -> dict[str, Any]:
    """Validate a list of API responses.

    Args:
        responses: List of API responses

    Returns:
        Validation results
    """
    total_responses = len(responses)
    successful_responses = sum(1 for r in responses if r["success"])
    failed_responses = total_responses - successful_responses

    response_times = [
        r["response_time"] for r in responses if r["response_time"] > 0
    ]

    return {
        "total_responses": total_responses,
        "successful_responses": successful_responses,
        "failed_responses": failed_responses,
        "success_rate": (
            successful_responses / total_responses
            if total_responses > 0
            else 0.0
        ),
        "average_response_time": (
            sum(response_times) / len(response_times)
            if response_times
            else 0.0
        ),
        "max_response_time": max(response_times) if response_times else 0.0,
        "min_response_time": min(response_times) if response_times else 0.0,
    }


# Predefined API endpoints for CrackSeg application
CRACKSEG_ENDPOINTS: dict[str, APIEndpoint] = {
    "main_page": {
        "url": "/",
        "method": "GET",
        "headers": None,
        "expected_status": 200,
        "timeout": 30.0,
    },
    "health_check": {
        "url": "/_stcore/health",
        "method": "GET",
        "headers": None,
        "expected_status": 200,
        "timeout": 10.0,
    },
    "static_files": {
        "url": "/_stcore/static",
        "method": "GET",
        "headers": None,
        "expected_status": 200,
        "timeout": 15.0,
    },
}
