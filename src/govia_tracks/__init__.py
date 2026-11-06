"""Aircraft tracklet association. Phase 1 covers validation and preprocessing only."""
from .errors import InputValidationError, Issue
from .loader import iter_partition, load_snapshot, read_jsonl
from .movement import TrackletMotion, summarize_tracklet
from .preprocess import PreparedScenario, Tracklet, prepare_scenario

__all__ = [
    "InputValidationError",
    "Issue",
    "PreparedScenario",
    "Tracklet",
    "TrackletMotion",
    "iter_partition",
    "load_snapshot",
    "prepare_scenario",
    "read_jsonl",
    "summarize_tracklet",
]
