"""HTTP API over the frozen association methods. Inference only: nothing here trains or reads labels.

Models and scenario metadata are loaded once at startup. A load failure doesn't stop the server: it is logged in
full, and /health and inference report it in plain words. Responses never carry file paths, exception text from
a load, or tracebacks.

  uvicorn govia_tracks.api:app --port 8000
"""
import json
import logging
import math
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import sklearn
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import baseline, learned
from .association import AcceptancePolicy
from .errors import InputValidationError, Issue
from .loader import _reject_constant, _reject_duplicate_keys, read_jsonl
from .preprocess import PreparedScenario, prepare_scenario

logger = logging.getLogger(__name__)

# Which folders can be served. The test partition is deliberately absent, so no request can reach it.
PARTITION_FOLDERS = {"train": "scenario_*", "validation": "scenario_*", "staged_demo": "snapshot_[12]"}
# The only manifest fields kept. The manifest also holds labeled_link_count, which must not reach a response.
SCENARIO_FIELDS = (
    "scenario_id", "snapshot_id", "receipt_cutoff", "tracklet_count", "unique_observation_count", "observation_row_count", "supporting_row_count",
)
_ISSUE_CODES = {"missing": "missing_field", "extra_forbidden": "unknown_field"}
_STATUS_CODES = {404: "not_found", 405: "method_not_allowed"}
_STATUS_MESSAGES = {404: "Not found.", 405: "Method not allowed."}


class ApiError(Exception):
    """A failure with a known status and code. Its message and issues go to the client, so they must not hold paths."""

    def __init__(self, status: int, code: str, message: str, issues=()):
        super().__init__(message)
        self.status, self.code, self.message, self.issues = status, code, message, tuple(issues)


class _IncompatibleModel(ValueError):
    """The saved model doesn't fit this environment. Its message is written for clients, and holds no path."""


class AssociateRequest(BaseModel):
    """Rows stay raw parsed JSON, so the validators in `preprocess` see them untouched, including missing versus null."""

    model_config = ConfigDict(extra="forbid", strict=True)

    observations: list[Any]
    supporting_records: list[Any] = []
    scenario_id: str | None = None
    snapshot_id: str | None = None
    receipt_cutoff: str | None = None

    @field_validator("scenario_id", "snapshot_id")
    @classmethod
    def _identifier_not_blank(cls, value: str | None) -> str | None:
        """A blank identifier would go on to label the result, where it identifies nothing."""
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value


@dataclass(frozen=True)
class ScenarioEntry:
    folder: Path
    summary: dict  # exactly what /scenarios returns for this scenario


@dataclass
class Runtime:
    """Everything loaded at startup. Read-only afterwards, so request threads can share it."""

    problems: list[str] = field(default_factory=list)
    baseline_config: baseline.BaselineConfig | None = None
    model: learned.LearnedModel | None = None
    policy: AcceptancePolicy | None = None
    scenarios: dict[tuple[str, str], ScenarioEntry] = field(default_factory=dict)

    @property
    def learned_available(self) -> bool:
        return self.model is not None and self.policy is not None

    @property
    def ready(self) -> bool:
        return self.baseline_config is not None and self.learned_available


def _bounded_int(text: str) -> int:
    """An integer beyond the double range isn't a usable JSON number, and `math.isfinite` raises on one."""
    value = int(text)
    if abs(value) > sys.float_info.max:
        raise ValueError("integer is out of range")
    return value


def _strict_loads(text: str | bytes) -> Any:
    """Parse JSON that can be safely answered: no NaN or Infinity (also 1e999), repeated keys, or unencodable strings.

    Python's parser turns 1e999 into infinity without calling `parse_constant`, and accepts a lone surrogate that
    would then fail when written to a response, so the parsed value is written once here as a check.
    """
    parsed = json.loads(text, parse_constant=_reject_constant, parse_int=_bounded_int, object_pairs_hook=_reject_duplicate_keys)
    json.dumps(parsed, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return parsed


def _attempt(load, problems: list[str], missing: str, invalid: str):
    """Run one startup step. The details go to the log; the client-facing problem is fixed text without paths."""
    try:
        return load()
    except FileNotFoundError:
        logger.exception("startup: %s", missing)
        problems.append(missing)
    except _IncompatibleModel as exc:
        logger.error("startup: %s", exc)
        problems.append(str(exc))
    except Exception:
        logger.exception("startup: %s", invalid)
        problems.append(invalid)
    return None


def _load_model(folder: Path) -> learned.LearnedModel:
    """Refuse a model from another scikit-learn before unpickling it, then check that the loaded object can score."""
    metadata = json.loads((folder / learned.METADATA_FILE).read_text(encoding="utf-8"))
    saved = metadata["library_versions"]["scikit-learn"]
    if saved != sklearn.__version__:
        raise _IncompatibleModel(f"learned model was saved with scikit-learn {saved}, but this server runs {sklearn.__version__}")
    if not isinstance(metadata.get("model_version"), str):
        raise ValueError("model metadata has no version")
    model = learned.load_model(folder)
    model.estimator.predict_proba([[math.nan] * len(model.feature_names)])
    return model


def _discover_scenarios(data_root: Path) -> dict[tuple[str, str], ScenarioEntry]:
    """List the servable folders. A folder with an unusable manifest is logged and left out."""
    found = {}
    for partition, pattern in PARTITION_FOLDERS.items():
        for folder in sorted((data_root / partition).glob(pattern)):
            if not folder.is_dir():
                continue
            try:
                manifest = _strict_loads((folder / "snapshot.json").read_text(encoding="utf-8"))
                metadata = {name: manifest[name] for name in SCENARIO_FIELDS}
            except (OSError, ValueError, KeyError, TypeError):
                logger.exception("skipping scenario folder %s", folder)
                continue
            key = f"{partition}/{folder.name}"
            found[(partition, folder.name)] = ScenarioEntry(folder, {"key": key, "partition": partition, "name": folder.name, **metadata})
    if not found:
        logger.warning("no scenarios found under %s", data_root)
    return found


def _load_runtime(root: Path) -> Runtime:
    """Load once. Never raises: whatever fails becomes a reported problem."""
    runtime = Runtime()
    problems = runtime.problems
    runtime.baseline_config = _attempt(
        lambda: baseline.load_config(root / "configs" / "baseline.json"), problems,
        "baseline configuration is missing", "baseline configuration could not be loaded",
    )
    runtime.model = _attempt(
        lambda: _load_model(root / "models"), problems,
        "learned model files are missing", "learned model could not be loaded or is damaged",
    )
    if runtime.model is not None:
        runtime.policy = _attempt(
            lambda: learned.load_policy(root / "configs" / "learned.json", runtime.model.version), problems,
            "learned thresholds are missing", "learned thresholds could not be loaded or were selected for a different model",
        )
    runtime.scenarios = _attempt(
        lambda: _discover_scenarios(root / "candidate-packet" / "data"), problems,
        "scenario data is missing", "scenario data could not be read",
    ) or {}
    return runtime


def _error_response(status: int, code: str, message: str, issues=(), headers=None) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "issues": [issue.to_dict() for issue in issues]}}
    return JSONResponse(body, status_code=status, headers=headers)


def _request_issues(errors) -> list[Issue]:
    """Pydantic errors as our issues. The offending input isn't echoed: it could be large, or hold labels."""
    issues = []
    for error in errors:
        name = ".".join(str(part) for part in error["loc"] if part != "body") or None
        issues.append(Issue(_ISSUE_CODES.get(error["type"], "invalid_field"), "request", f"{name or 'request body'}: {error['msg']}", field=name))
    return issues


def _invalid_request(errors) -> ApiError:
    return ApiError(422, "invalid_request", "The request does not have the expected shape.", _request_issues(errors))


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    def on_api_error(request: Request, exc: ApiError):
        return _error_response(exc.status, exc.code, exc.message, exc.issues)

    @app.exception_handler(RequestValidationError)
    def on_request_validation(request: Request, exc: RequestValidationError):
        error = _invalid_request(exc.errors())
        return _error_response(error.status, error.code, error.message, error.issues)

    @app.exception_handler(StarletteHTTPException)
    def on_http_exception(request: Request, exc: StarletteHTTPException):
        code = _STATUS_CODES.get(exc.status_code, "internal_error" if exc.status_code >= 500 else "invalid_request")
        message = _STATUS_MESSAGES.get(exc.status_code, "The request could not be processed.")
        return _error_response(exc.status_code, code, message, headers=exc.headers)

    @app.exception_handler(Exception)
    def on_unhandled(request: Request, exc: Exception):
        logger.error("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
        return _error_response(500, "internal_error", "An internal error occurred.")


def _parse_associate_request(body: bytes) -> AssociateRequest:
    try:
        payload = _strict_loads(body)
    except (ValueError, RecursionError) as exc:
        # Escaped because a repeated-key message can quote a key that isn't encodable.
        reason = str(exc).encode("ascii", "backslashreplace").decode("ascii")
        raise ApiError(400, "invalid_json", f"The request body is not valid JSON: {reason}") from exc
    try:
        return AssociateRequest.model_validate(payload)
    except ValidationError as exc:
        raise _invalid_request(exc.errors(include_input=False, include_url=False, include_context=False)) from exc


def _analysis_summary(prepared: PreparedScenario) -> dict:
    return {
        "scenario_id": prepared.scenario_id,
        "snapshot_id": prepared.snapshot_id,
        "receipt_cutoff": prepared.receipt_cutoff,
        "input_fingerprint": prepared.input_fingerprint,
        "tracklet_count": len(prepared.tracklets),
        "observation_count": prepared.quality.unique_observations,
        "supporting_count": prepared.quality.unique_supporting,
        "quality": asdict(prepared.quality),
    }


def _health_body(runtime: Runtime) -> dict:
    return {
        "status": "ready" if runtime.ready else "unavailable",
        "ready": runtime.ready,
        "baseline": {
            "available": runtime.baseline_config is not None,
            "version": None if runtime.baseline_config is None else baseline.config_version(runtime.baseline_config),
        },
        "learned": {
            "available": runtime.learned_available,
            "version": runtime.model.version if runtime.learned_available else None,
            "feature_names": list(runtime.model.feature_names) if runtime.learned_available else None,
        },
        "problems": list(runtime.problems),
    }


async def _request_body(request: Request) -> bytes:
    """Only the read happens on the event loop. Parsing and inference run in the sync endpoint's thread."""
    return await request.body()


def create_app(root: Path | None = None) -> FastAPI:
    """Build the API. Nothing is loaded until startup. `root` wins over the GOVIA_ROOT variable, then the repo root."""
    resolved = Path(root or os.environ.get("GOVIA_ROOT") or Path(__file__).parents[2])

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = _load_runtime(resolved)
        yield

    # Without redirect_slashes=False, /health/ would answer with a bare redirect instead of the error shape.
    app = FastAPI(title="Govia tracklet association", lifespan=lifespan, redirect_slashes=False)
    app.state.runtime = Runtime(problems=["the service has not finished starting"])
    _install_error_handlers(app)

    @app.get("/health")
    def health():
        runtime = app.state.runtime
        body = _health_body(runtime)
        if runtime.ready:
            return JSONResponse(body)
        # The health fields and the error shape are both required of a non-2xx here, and they don't collide.
        message = "; ".join(runtime.problems) or "The service is not ready."
        return JSONResponse({**body, "error": {"code": "model_unavailable", "message": message, "issues": []}}, status_code=503)

    @app.post(
        "/tracklets/associate",
        openapi_extra={"requestBody": {"required": True, "content": {"application/json": {"schema": AssociateRequest.model_json_schema()}}}},
    )
    def associate(body: bytes = Depends(_request_body)):
        request = _parse_associate_request(body)
        runtime = app.state.runtime
        if not runtime.ready:
            raise ApiError(503, "model_unavailable", "; ".join(runtime.problems) or "The service is not ready.")
        if not request.observations:
            raise ApiError(422, "empty_input", "No observations were supplied.")
        try:
            prepared = prepare_scenario(
                request.observations, request.supporting_records,
                scenario_id=request.scenario_id, snapshot_id=request.snapshot_id, receipt_cutoff=request.receipt_cutoff,
            )
        except InputValidationError as exc:
            raise ApiError(422, "validation_failed", str(exc), exc.issues) from exc
        except RecursionError as exc:
            raise ApiError(422, "invalid_request", "A record is nested too deeply.") from exc
        # Rendered here, in the worker thread. JSONResponse refuses a non-finite value, which then surfaces as internal_error.
        return JSONResponse({
            "analysis": _analysis_summary(prepared),
            "baseline": baseline.analyze(prepared, runtime.baseline_config),
            "learned": learned.analyze(prepared, runtime.model, runtime.policy),
        })

    @app.get("/scenarios")
    def scenarios():
        return JSONResponse({"scenarios": [entry.summary for entry in app.state.runtime.scenarios.values()]})

    @app.get("/scenarios/{partition}/{name}/records")
    def records(partition: str, name: str):
        # Only an exact match against what startup discovered: no path is ever built from these values.
        entry = app.state.runtime.scenarios.get((partition, name))
        if entry is None:
            raise ApiError(404, "not_found", "No such scenario.")
        # A bad data file isn't the caller's fault, so InputValidationError here falls through to internal_error.
        return JSONResponse({
            "key": entry.summary["key"],
            "scenario_id": entry.summary["scenario_id"],
            "snapshot_id": entry.summary["snapshot_id"],
            "receipt_cutoff": entry.summary["receipt_cutoff"],
            "observations": read_jsonl(entry.folder / "observations.jsonl", "observation"),
            "supporting_records": read_jsonl(entry.folder / "supporting.jsonl", "supporting"),
        })

    return app


app = create_app()
