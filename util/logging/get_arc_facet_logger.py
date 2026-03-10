"""
Logging wrapper for getArcFacet function calls.

This module provides monkey patching functionality to log all calls to getArcFacet
for debugging and test case extraction.

Usage:
    from util.logging.get_arc_facet_logger import (
        enable_logging,
        disable_logging,
        arc_facet_log_context,
    )

    enable_logging(log_file="get_arc_facet_calls.log")
    with arc_facet_log_context(experiment="zalesak", algo="safe_circle"):
        ...
    disable_logging()
"""

from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from main.geoms.circular_facet import LinearFacetShortcut


ALLOWED_METADATA_KEYS = {
    "experiment",
    "algo",
    "resolution",
    "wiggle",
    "seed",
    "save_name",
    "case_index",
    "call_source",
    "grid_coords",
    "merge_id",
    "merge_coords",
    "metric_stage",
    "facet_index",
}

_LOG_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar("arc_facet_log_context", default={})


def _normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {}
    for key, value in metadata.items():
        if key in ALLOWED_METADATA_KEYS and value is not None:
            normalized[key] = value
    return normalized


@contextmanager
def arc_facet_log_context(**metadata):
    current = dict(_LOG_CONTEXT.get())
    current.update(_normalize_metadata(metadata))
    token = _LOG_CONTEXT.set(current)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


def get_current_log_context() -> Dict[str, Any]:
    return dict(_LOG_CONTEXT.get())


class GetArcFacetLogger:
    """Logger for getArcFacet function calls."""

    def __init__(self, log_file: str = "get_arc_facet_calls.log"):
        self.log_file = log_file
        self.call_count = 0
        self.enabled = False
        self.original_function = None
        self.log_handle = None

    def _build_inputs(
        self,
        poly1: List[List[float]],
        poly2: List[List[float]],
        poly3: List[List[float]],
        a1: float,
        a2: float,
        a3: float,
        epsilon: float,
        gcenterx: Optional[float],
        gcentery: Optional[float],
        gradius: Optional[float],
    ) -> Dict[str, Any]:
        return {
            "poly1": poly1,
            "poly2": poly2,
            "poly3": poly3,
            "a1": a1,
            "a2": a2,
            "a3": a3,
            "epsilon": epsilon,
            "gcenterx": gcenterx,
            "gcentery": gcentery,
            "gradius": gradius,
        }

    def _write_entry(self, entry: Dict[str, Any]):
        if not self.enabled or self.log_handle is None:
            return
        self.log_handle.write(json.dumps(self._to_jsonable(entry)) + "\n")
        self.log_handle.flush()

    def _to_jsonable(self, value: Any, stack=None, depth: int = 0) -> Any:
        if stack is None:
            stack = set()
        if depth > 32:
            return repr(value)
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            obj_id = id(value)
            if obj_id in stack:
                return "<circular-dict>"
            stack.add(obj_id)
            try:
                return {
                    str(key): self._to_jsonable(item, stack=stack, depth=depth + 1)
                    for key, item in value.items()
                }
            finally:
                stack.remove(obj_id)
        if isinstance(value, (list, tuple)):
            obj_id = id(value)
            if obj_id in stack:
                return "<circular-sequence>"
            stack.add(obj_id)
            try:
                return [
                    self._to_jsonable(item, stack=stack, depth=depth + 1)
                    for item in value
                ]
            finally:
                stack.remove(obj_id)
        if hasattr(value, "item"):
            try:
                return self._to_jsonable(value.item(), stack=stack, depth=depth + 1)
            except Exception:
                return repr(value)
        return repr(value)

    def wrapper(
        self,
        poly1,
        poly2,
        poly3,
        afrac1,
        afrac2,
        afrac3,
        epsilon,
        gcenterx=None,
        gcentery=None,
        gradius=None,
        _orientation_retry=False,
    ):
        """Wrapper function that logs calls to getArcFacet."""
        self.call_count += 1
        call_id = self.call_count
        metadata = get_current_log_context()
        inputs = self._build_inputs(
            poly1,
            poly2,
            poly3,
            afrac1,
            afrac2,
            afrac3,
            epsilon,
            gcenterx,
            gcentery,
            gradius,
        )

        self._write_entry(
            {
                "call_id": call_id,
                "timestamp": time.time(),
                "status": "started",
                "inputs": inputs,
                "metadata": metadata,
            }
        )

        start_time = time.time()

        try:
            result = self.original_function(
                poly1,
                poly2,
                poly3,
                afrac1,
                afrac2,
                afrac3,
                epsilon,
                gcenterx,
                gcentery,
                gradius,
                _orientation_retry,
            )
            end_time = time.time()
            center, radius, intersects = result
            status = "success"
            if center is None or radius is None or intersects is None:
                status = "failed"
            self._write_entry(
                {
                    "call_id": call_id,
                    "timestamp": time.time(),
                    "execution_time_seconds": end_time - start_time,
                    "status": status,
                    "inputs": inputs,
                    "metadata": metadata,
                    "outputs": {
                        "center": center,
                        "radius": radius,
                        "num_intersects": len(intersects) if intersects else 0,
                    },
                }
            )
            return result
        except LinearFacetShortcut as shortcut:
            end_time = time.time()
            self._write_entry(
                {
                    "call_id": call_id,
                    "timestamp": time.time(),
                    "execution_time_seconds": end_time - start_time,
                    "status": "linear_shortcut",
                    "inputs": inputs,
                    "metadata": metadata,
                    "outputs": {
                        "pLeft": shortcut.pLeft,
                        "pRight": shortcut.pRight,
                        "linear_fraction": shortcut.linear_fraction,
                        "target_fraction": shortcut.target_fraction,
                    },
                }
            )
            raise
        except Exception as error:
            end_time = time.time()
            self._write_entry(
                {
                    "call_id": call_id,
                    "timestamp": time.time(),
                    "execution_time_seconds": end_time - start_time,
                    "status": "exception",
                    "inputs": inputs,
                    "metadata": metadata,
                    "error": str(error),
                }
            )
            raise

    def enable(self, log_file: Optional[str] = None):
        """Enable logging by monkey patching getArcFacet."""
        if self.enabled:
            print(
                "Warning: Logging is already enabled. Disable first before re-enabling."
            )
            return

        if log_file:
            self.log_file = log_file

        from main.geoms import circular_facet

        self.original_function = circular_facet.getArcFacet

        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = open(log_path, "w", encoding="utf-8")

        self.log_handle.write("# getArcFacet call log\n")
        self.log_handle.write(f"# Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_handle.write("# Format: One JSON object per line\n")
        self.log_handle.write("#\n")

        circular_facet.getArcFacet = self.wrapper

        self.enabled = True
        print(f"Logging enabled: calls will be logged to {self.log_file}")
        print("Original function saved, ready to log calls.")

    def disable(self):
        """Disable logging and restore original function."""
        if not self.enabled:
            print("Warning: Logging is not currently enabled.")
            return

        from main.geoms import circular_facet

        circular_facet.getArcFacet = self.original_function

        if self.log_handle:
            self.log_handle.write(
                f"# Logging stopped at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            self.log_handle.write(f"# Total calls logged: {self.call_count}\n")
            self.log_handle.close()
            self.log_handle = None

        self.enabled = False
        print(f"Logging disabled: {self.call_count} calls logged to {self.log_file}")
        print("Original function restored.")

    def get_stats(self) -> dict:
        """Get statistics about logged calls."""
        stats = {
            "total_calls": self.call_count,
            "log_file": self.log_file,
        }
        if self.call_count == 0:
            return stats

        try:
            from util.logging.analyze_log import consolidate_log_entries, load_log_entries

            log_path = Path(self.log_file)
            if not log_path.exists():
                return stats

            entries = consolidate_log_entries(load_log_entries(str(log_path)))
            execution_times = [
                entry.get("execution_time_seconds", 0.0)
                for entry in entries
                if entry.get("execution_time_seconds") is not None
            ]
            stats.update(
                {
                    "success_count": sum(
                        1 for entry in entries if entry.get("status") == "success"
                    ),
                    "failed_count": sum(
                        1 for entry in entries if entry.get("status") == "failed"
                    ),
                    "exception_count": sum(
                        1 for entry in entries if entry.get("status") == "exception"
                    ),
                    "started_only_count": sum(
                        1 for entry in entries if entry.get("status") == "started_only"
                    ),
                    "avg_execution_time": (
                        sum(execution_times) / len(execution_times)
                        if execution_times
                        else 0.0
                    ),
                    "max_execution_time": max(execution_times) if execution_times else 0.0,
                    "total_execution_time": sum(execution_times),
                }
            )
        except Exception as error:
            stats["error"] = f"Could not read log file: {error}"

        return stats


_logger = GetArcFacetLogger()


def enable_logging(log_file: str = "get_arc_facet_calls.log"):
    """Enable logging of getArcFacet calls."""
    _logger.enable(log_file)


def disable_logging():
    """Disable logging and restore original function."""
    _logger.disable()


def get_stats() -> dict:
    """Get statistics about logged calls."""
    return _logger.get_stats()


def is_enabled() -> bool:
    """Check if logging is currently enabled."""
    return _logger.enabled
