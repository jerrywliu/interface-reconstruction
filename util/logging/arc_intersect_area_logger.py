"""
Logging wrapper for ArcFacet.getPolyIntersectArea calls.

This is intended for diagnosing rare stalls in arc-polygon area evaluation,
especially in the Zalesak area-error path.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from util.logging.get_arc_facet_logger import get_current_log_context


class ArcIntersectAreaLogger:
    """Logger for ArcFacet.getPolyIntersectArea calls."""

    def __init__(self, log_file: str = "arc_intersect_area_calls.log"):
        self.log_file = log_file
        self.call_count = 0
        self.enabled = False
        self.original_function = None
        self.patched_function = None
        self.log_handle = None

    def _build_inputs(self, facet, poly: List[List[float]]) -> Dict[str, Any]:
        return {
            "poly": poly,
            "center": facet.center,
            "radius": facet.radius,
            "pLeft": facet.pLeft,
            "pRight": facet.pRight,
            "is_major_arc": facet.is_major_arc,
        }

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

    def _write_entry(self, entry: Dict[str, Any]):
        if not self.enabled or self.log_handle is None:
            return
        self.log_handle.write(json.dumps(self._to_jsonable(entry)) + "\n")
        self.log_handle.flush()

    def wrapper(self, facet, poly):
        self.call_count += 1
        call_id = self.call_count
        metadata = get_current_log_context()
        inputs = self._build_inputs(facet, poly)

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
            area = self.original_function(facet, poly)
            end_time = time.time()
            self._write_entry(
                {
                    "call_id": call_id,
                    "timestamp": time.time(),
                    "execution_time_seconds": end_time - start_time,
                    "status": "success",
                    "inputs": inputs,
                    "metadata": metadata,
                    "outputs": {
                        "area": area,
                    },
                }
            )
            return area
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
        if self.enabled:
            print(
                "Warning: Arc intersect area logging is already enabled. Disable first before re-enabling."
            )
            return

        if log_file:
            self.log_file = log_file

        from main.structs.facets import circular_facet as facet_module

        self.original_function = facet_module.ArcFacet.getPolyIntersectArea

        def _patched(facet, poly):
            return self.wrapper(facet, poly)

        self.patched_function = _patched

        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = open(log_path, "w", encoding="utf-8")
        self.log_handle.write("# ArcFacet.getPolyIntersectArea call log\n")
        self.log_handle.write(f"# Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_handle.write("# Format: One JSON object per line\n")
        self.log_handle.write("#\n")

        facet_module.ArcFacet.getPolyIntersectArea = self.patched_function

        self.enabled = True
        print(f"Arc intersect area logging enabled: {self.log_file}")

    def disable(self):
        if not self.enabled:
            print("Warning: Arc intersect area logging is not currently enabled.")
            return

        from main.structs.facets import circular_facet as facet_module

        facet_module.ArcFacet.getPolyIntersectArea = self.original_function

        if self.log_handle:
            self.log_handle.write(
                f"# Logging stopped at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            self.log_handle.write(f"# Total calls logged: {self.call_count}\n")
            self.log_handle.close()
            self.log_handle = None

        self.enabled = False
        print(
            f"Arc intersect area logging disabled: {self.call_count} calls logged to {self.log_file}"
        )

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "total_calls": self.call_count,
            "log_file": self.log_file,
        }
        if not Path(self.log_file).exists():
            return stats

        success_count = 0
        exception_count = 0
        started_only_count = 0
        total_execution_time = 0.0
        max_execution_time = 0.0
        call_statuses: Dict[int, Dict[str, Any]] = {}

        for line in Path(self.log_file).open(encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            call_id = entry.get("call_id")
            if call_id is None:
                continue
            call_statuses.setdefault(call_id, {})
            call_statuses[call_id][entry.get("status")] = entry

        for statuses in call_statuses.values():
            if list(statuses.keys()) == ["started"]:
                started_only_count += 1
                continue
            if "success" in statuses:
                success_count += 1
                duration = statuses["success"].get("execution_time_seconds", 0.0) or 0.0
            elif "exception" in statuses:
                exception_count += 1
                duration = (
                    statuses["exception"].get("execution_time_seconds", 0.0) or 0.0
                )
            else:
                duration = 0.0
            total_execution_time += duration
            max_execution_time = max(max_execution_time, duration)

        stats.update(
            {
                "success_count": success_count,
                "exception_count": exception_count,
                "started_only_count": started_only_count,
                "total_execution_time": total_execution_time,
                "max_execution_time": max_execution_time,
            }
        )
        return stats


_logger = ArcIntersectAreaLogger()


def enable_logging(log_file: str = "arc_intersect_area_calls.log"):
    _logger.enable(log_file)


def disable_logging():
    _logger.disable()


def get_stats() -> Dict[str, Any]:
    return _logger.get_stats()


def is_enabled() -> bool:
    return _logger.enabled
