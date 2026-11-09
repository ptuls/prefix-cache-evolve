"""Compatibility exports for metadata-only trace replay."""

from prefix_cache_evolve.evaluators.trace_replay import (
    TRACE_REPLAY_CONTRACT as TRACE_REPLAY_CONTRACT,
)
from prefix_cache_evolve.evaluators.trace_replay import TraceRecord as TraceRecord
from prefix_cache_evolve.evaluators.trace_replay import (
    calibrate_anonymized_trace as calibrate_anonymized_trace,
)
from prefix_cache_evolve.evaluators.trace_replay import iter_trace_records as iter_trace_records
from prefix_cache_evolve.evaluators.trace_replay import (
    load_anonymized_trace as load_anonymized_trace,
)
