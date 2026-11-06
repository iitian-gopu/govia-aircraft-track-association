"""HTTP client for the association API. Every failure surfaces as one typed ApiError, never a raw exception."""
import os
from urllib.parse import quote

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
TIMEOUT_S = 30.0  # every request except the health check, which runs on every page rerun
HEALTH_TIMEOUT_S = 5.0

# Codes this client adds to the API's own: the API can't report these about itself.
BACKEND_UNAVAILABLE = "backend_unavailable"
TIMEOUT = "timeout"
BAD_RESPONSE = "bad_response"


class ApiError(Exception):
    def __init__(self, code: str, message: str, issues: list | None = None, status: int | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.issues = issues or []
        self.status = status


def _error_from_body(status: int, body) -> ApiError:
    """Build the error from the API's envelope, or say plainly that the response had none."""
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and error.get("code"):
        issues = error.get("issues")
        return ApiError(
            str(error["code"]), str(error.get("message") or "The API gave no message."),
            [i for i in issues if isinstance(i, dict)] if isinstance(issues, list) else [], status,
        )
    code = "internal_error" if status >= 500 else BAD_RESPONSE
    return ApiError(code, f"The API returned HTTP {status} without a recognised error body.", status=status)


def _require(body: dict, *fields_and_types) -> None:
    for field, kind in fields_and_types:
        if not isinstance(body.get(field), kind):
            raise ApiError(BAD_RESPONSE, f"The API response has no usable '{field}' field.")


class ApiClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = TIMEOUT_S, transport: httpx.BaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._transport = transport

    @classmethod
    def from_env(cls) -> "ApiClient":
        return cls(os.environ.get("GOVIA_API_URL") or DEFAULT_BASE_URL)

    def _request(self, method: str, path: str, *, json=None, timeout: float | None = None, body_on: tuple[int, ...] = ()) -> dict:
        try:
            with httpx.Client(base_url=self.base_url, timeout=timeout or self.timeout, transport=self._transport) as http:
                response = http.request(method, path, json=json)
        except httpx.TimeoutException as error:
            raise ApiError(TIMEOUT, f"The API at {self.base_url} did not answer within {timeout or self.timeout:.0f} s.") from error
        except (httpx.TransportError, httpx.InvalidURL) as error:
            raise ApiError(BACKEND_UNAVAILABLE, f"Could not reach the API at {self.base_url}.") from error

        try:
            body = response.json()
        except ValueError as error:
            if response.is_success:
                raise ApiError(BAD_RESPONSE, f"The API returned HTTP {response.status_code} with a body that is not JSON.", status=response.status_code) from error
            code = "internal_error" if response.status_code >= 500 else BAD_RESPONSE
            raise ApiError(code, f"The API returned HTTP {response.status_code} with a body that is not JSON.", status=response.status_code) from error

        if not response.is_success and response.status_code not in body_on:
            raise _error_from_body(response.status_code, body)
        if not isinstance(body, dict):
            raise ApiError(BAD_RESPONSE, "The API returned JSON that is not an object.", status=response.status_code)
        return body

    def health(self) -> dict:
        """200 or 503 with a health body. A 503 is an answer ("not ready"), so it's returned, not raised."""
        body = self._request("GET", "/health", timeout=HEALTH_TIMEOUT_S, body_on=(503,))
        if "ready" not in body:
            raise _error_from_body(503, body) if "error" in body else ApiError(BAD_RESPONSE, "The health response has no 'ready' field.")
        return body

    def scenarios(self) -> list[dict]:
        body = self._request("GET", "/scenarios")
        _require(body, ("scenarios", list))
        return body["scenarios"]

    def records(self, partition: str, name: str) -> dict:
        body = self._request("GET", f"/scenarios/{quote(partition, safe='')}/{quote(name, safe='')}/records")
        _require(body, ("observations", list), ("supporting_records", list))
        return body

    def associate(self, records: dict) -> dict:
        """POST one scenario's own records. Never combines snapshots."""
        payload = {
            "observations": records["observations"],
            "supporting_records": records["supporting_records"],
            "scenario_id": records.get("scenario_id"),
            "snapshot_id": records.get("snapshot_id"),
            "receipt_cutoff": records.get("receipt_cutoff"),
        }
        body = self._request("POST", "/tracklets/associate", json=payload)
        _require(body, ("analysis", dict), ("baseline", dict), ("learned", dict))
        for method in ("baseline", "learned"):
            _require(body[method], ("tracklets", list), ("links", dict))
        return body
