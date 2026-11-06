"""Validation failures. Every problem in an input is collected and reported together."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Issue:
    code: str  # invalid_field, missing_field, conflicting_id, after_cutoff, malformed_json, count_mismatch, invalid_structure
    record_kind: str  # observation, supporting, label, snapshot
    message: str
    index: int | None = None  # zero-based row position in the supplied input
    record_id: str | None = None
    field: str | None = None

    def to_dict(self):
        return asdict(self)


class InputValidationError(ValueError):
    """The input was rejected. No partial result is produced."""

    def __init__(self, issues):
        self.issues = tuple(issues)
        shown = "; ".join(i.message for i in self.issues[:5])
        extra = len(self.issues) - 5
        suffix = f"; and {extra} more" if extra > 0 else ""
        super().__init__(f"{len(self.issues)} validation issue(s): {shown}{suffix}")
