"""API integration helpers for E2E testing with external services.

This module provides utilities for testing API interactions, health checks,
load simulation, and response validation. These helpers are specifically
designed for testing the CrackSeg application's external integrations.
"""

import json
import time
from typing import TYPE_CHECKING, Any, TypedDict
from urllib.parse import urljoin

if TYPE_CHECKING:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

try:
    import requests  # noqa: F401
    from requests.adapters import HTTPAdapter  # noqa: F401
    from urllib3.util.retry import Retry  # noqa: F401

    _requests_available = True
except ImportError:
    _requests_available = False

import logging

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
            APIResponse with status and timing information
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
                json=(
                    data
                    if data and method in ["POST", "PUT", "PATCH"]
                    else None
                ),
                params=params,
                timeout=self.timeout,
            )

            response_time = time.time() - start_time

            # Try to parse JSON content
            try:
                content = response.json()
            except (ValueError, json.JSONDecodeError):
                content = response.text

            return {
                "status_code": response.status_code,
                "response_time": response_time,
                "content": content,
                "headers": dict(response.headers),
                "success": response.status_code < 400,
                "error": None,
            }

        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"Request to {url} failed: {e}")

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
        """Test if an endpoint is healthy and responding.

        Args:
            endpoint: Endpoint path to test
            expected_status: Expected HTTP status code

        Returns:
            True if endpoint is healthy
        """
        response = self.make_request(endpoint)

        is_healthy = (
            response["success"]
            and response["status_code"] == expected_status
            and response["response_time"] < self.timeout
        )

        if is_healthy:
            self.logger.info(
                f"Endpoint {endpoint} is healthy "
                f"(status: {response['status_code']}, "
                f"time: {response['response_time']:.3f}s)"
            )
        else:
            self.logger.warning(
                f"Endpoint {endpoint} health check failed: {response}"
            )

        return is_healthy

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
            concurrent_users: Number of concurrent users (simplified)

        Returns:
            Load test results with performance metrics
        """
        results: dict[str, Any] = {
            "total_requests": requests_count,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "min_response_time": float("inf"),
            "max_response_time": 0.0,
            "errors": [],
        }

        response_times: list[float] = []

        self.logger.info(
            f"Starting load test: {requests_count} requests to {endpoint}"
        )

        for i in range(requests_count):
            response = self.make_request(endpoint)

            if response["success"]:
                results["successful_requests"] += 1
                response_times.append(response["response_time"])
                results["min_response_time"] = min(
                    results["min_response_time"], response["response_time"]
                )
                results["max_response_time"] = max(
                    results["max_response_time"], response["response_time"]
                )
            else:
                results["failed_requests"] += 1
                results["errors"].append(
                    {
                        "request_number": i + 1,
                        "error": response["error"],
                        "status_code": response["status_code"],
                    }
                )

            # Simple delay to simulate concurrent users
            if i % concurrent_users == 0 and i > 0:
                time.sleep(0.1)

        if response_times:
            results["average_response_time"] = sum(response_times) / len(
                response_times
            )
        else:
            results["min_response_time"] = 0.0

        self.logger.info(
            f"Load test completed: "
            f"{results['successful_requests']}/{requests_count} successful"
        )
        return results

    def validate_api_response(
        self,
        response: APIResponse,
        expected_schema: dict[str, Any] | None = None,
    ) -> bool:
        """Validate API response against expected criteria.

        Args:
            response: API response to validate
            expected_schema: Optional schema to validate against

        Returns:
            True if response is valid
        """
        # Basic validation
        if not response["success"]:
            self.logger.error(
                f"Response validation failed: {response['error']}"
            )
            return False

        # Schema validation (simplified)
        if expected_schema and response["content"]:
            try:
                # Basic type checking
                if "type" in expected_schema:
                    expected_type = expected_schema["type"]
                    if expected_type == "object" and not isinstance(
                        response["content"], dict
                    ):
                        return False
                    elif expected_type == "array" and not isinstance(
                        response["content"], list
                    ):
                        return False

                # Check required fields
                if "required_fields" in expected_schema and isinstance(
                    response["content"], dict
                ):
                    for field in expected_schema["required_fields"]:
                        if field not in response["content"]:
                            self.logger.error(
                                f"Required field '{field}' missing from "
                                f"response"
                            )
                            return False

            except Exception as e:
                self.logger.error(f"Schema validation error: {e}")
                return False

        return True


def verify_streamlit_health(base_url: str = "http://localhost:8501") -> bool:
    """Verify that Streamlit application is healthy and responding.

    Args:
        base_url: Base URL of Streamlit application

    Returns:
        True if Streamlit is healthy
    """
    helper = APITestHelper(base_url)

    # Test main endpoint
    if not helper.test_endpoint_health("/"):
        return False

    # Test Streamlit-specific endpoints
    streamlit_endpoints = [
        "/healthz",  # Common health check endpoint
        "/_stcore/health",  # Streamlit internal health
    ]

    healthy_endpoints = 0
    for endpoint in streamlit_endpoints:
        if helper.test_endpoint_health(endpoint, expected_status=200):
            healthy_endpoints += 1
        # Don't fail if some endpoints don't exist

    logger.info(
        f"Streamlit health check: {healthy_endpoints}/"
        f"{len(streamlit_endpoints) + 1} endpoints healthy"
    )
    return True  # Main endpoint working is sufficient


def test_crackseg_endpoints(
    base_url: str = "http://localhost:8501",
) -> dict[str, bool]:
    """Test CrackSeg-specific endpoints for functionality.

    Args:
        base_url: Base URL of CrackSeg application

    Returns:
        Dictionary of endpoint test results
    """
    helper = APITestHelper(base_url)

    # CrackSeg-specific endpoints to test
    endpoints: dict[str, APIEndpoint] = {
        "main_page": {
            "url": "/",
            "method": "GET",
            "headers": None,
            "expected_status": 200,
            "timeout": 10.0,
        },
        "static_files": {
            "url": "/static/style.css",  # Example static file
            "method": "GET",
            "headers": None,
            "expected_status": 200,
            "timeout": 5.0,
        },
    }

    results: dict[str, bool] = {}

    for name, config in endpoints.items():
        response = helper.make_request(
            config["url"], method=config["method"], headers=config["headers"]
        )

        results[name] = (
            response["success"]
            and response["status_code"] == config["expected_status"]
            and response["response_time"] < config["timeout"]
        )

        logger.info(
            f"CrackSeg endpoint {name}: {'✅' if results[name] else '❌'}"
        )

    return results


def simulate_api_load(
    base_url: str = "http://localhost:8501",
    duration_seconds: float = 30.0,
    requests_per_second: float = 2.0,
) -> dict[str, Any]:
    """Simulate API load over a specified duration.

    Args:
        base_url: Base URL to test
        duration_seconds: How long to run the load test
        requests_per_second: Target requests per second

    Returns:
        Load test results and performance metrics
    """
    helper = APITestHelper(base_url)

    total_requests = int(duration_seconds * requests_per_second)

    logger.info(
        f"Starting API load simulation: {total_requests} requests "
        f"over {duration_seconds}s"
    )

    start_time = time.time()
    results = helper.simulate_load_test("/", requests_count=total_requests)
    actual_duration = time.time() - start_time

    # Add duration metrics
    results["planned_duration"] = duration_seconds
    results["actual_duration"] = actual_duration
    results["planned_rps"] = requests_per_second
    results["actual_rps"] = (
        total_requests / actual_duration if actual_duration > 0 else 0
    )

    logger.info(
        f"Load simulation completed in {actual_duration:.1f}s "
        f"(planned: {duration_seconds:.1f}s)"
    )
    return results


def validate_api_responses(responses: list[APIResponse]) -> dict[str, Any]:
    """Validate a collection of API responses for consistency and correctness.

    Args:
        responses: List of API responses to validate

    Returns:
        Validation report with statistics and issues
    """
    if not responses:
        return {"total": 0, "valid": 0, "invalid": 0, "issues": []}

    validation_report: dict[str, Any] = {
        "total": len(responses),
        "valid": 0,
        "invalid": 0,
        "issues": [],
        "response_time_stats": {
            "min": float("inf"),
            "max": 0.0,
            "average": 0.0,
        },
    }

    response_times: list[float] = []
    helper = APITestHelper()  # For validation methods

    for i, response in enumerate(responses):
        is_valid = helper.validate_api_response(response)

        if is_valid:
            validation_report["valid"] += 1
            response_times.append(response["response_time"])
        else:
            validation_report["invalid"] += 1
            validation_report["issues"].append(
                {
                    "response_index": i,
                    "status_code": response["status_code"],
                    "error": response["error"],
                }
            )

    # Calculate response time statistics
    if response_times:
        validation_report["response_time_stats"] = {
            "min": min(response_times),
            "max": max(response_times),
            "average": sum(response_times) / len(response_times),
        }

    logger.info(
        f"Response validation: {validation_report['valid']}/"
        f"{validation_report['total']} valid"
    )
    return validation_report
