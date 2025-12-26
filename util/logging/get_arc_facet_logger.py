"""
Logging wrapper for getArcFacet function calls.

This module provides monkey patching functionality to log all calls to getArcFacet
for debugging and test case extraction.

Usage:
    from util.logging.get_arc_facet_logger import enable_logging, disable_logging

    enable_logging(log_file='get_arc_facet_calls.log')
    # Run your experiments...
    disable_logging()
"""

import time
import json
import sys
from typing import Optional, List, Tuple, Any
from pathlib import Path


class GetArcFacetLogger:
    """Logger for getArcFacet function calls."""

    def __init__(self, log_file: str = "get_arc_facet_calls.log"):
        self.log_file = log_file
        self.call_count = 0
        self.enabled = False
        self.original_function = None
        self.log_handle = None

    def _log_call(
        self,
        poly1: List[List[float]],
        poly2: List[List[float]],
        poly3: List[List[float]],
        a1: float,
        a2: float,
        a3: float,
        epsilon: float,
        start_time: float,
        end_time: float,
        result: Tuple[Any, Any, Any],
        success: bool,
    ):
        """Log a single function call."""
        if not self.enabled or self.log_handle is None:
            return

        center, radius, intersects = result

        # Calculate execution time
        execution_time = end_time - start_time

        # Determine convergence status
        if center is None or radius is None or intersects is None:
            status = "failed"
        else:
            status = "success"

        # Create log entry
        log_entry = {
            "call_id": self.call_count,
            "timestamp": time.time(),
            "execution_time_seconds": execution_time,
            "status": status,
            "inputs": {
                "poly1": poly1,
                "poly2": poly2,
                "poly3": poly3,
                "a1": a1,
                "a2": a2,
                "a3": a3,
                "epsilon": epsilon,
            },
            "outputs": {
                "center": center,
                "radius": radius,
                "num_intersects": len(intersects) if intersects else 0,
            },
        }

        # Write as JSON line (one JSON object per line for easy parsing)
        self.log_handle.write(json.dumps(log_entry) + "\n")
        self.log_handle.flush()  # Ensure it's written immediately

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
    ):
        """Wrapper function that logs calls to getArcFacet."""
        self.call_count += 1
        start_time = time.time()

        try:
            # Call the original function
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
            )
            end_time = time.time()

            # Determine success
            center, radius, intersects = result
            success = (
                center is not None and radius is not None and intersects is not None
            )

            # Log the call
            self._log_call(
                poly1,
                poly2,
                poly3,
                afrac1,
                afrac2,
                afrac3,
                epsilon,
                start_time,
                end_time,
                result,
                success,
            )

            return result

        except Exception as e:
            end_time = time.time()
            # Log the exception
            log_entry = {
                "call_id": self.call_count,
                "timestamp": time.time(),
                "execution_time_seconds": end_time - start_time,
                "status": "exception",
                "inputs": {
                    "poly1": poly1,
                    "poly2": poly2,
                    "poly3": poly3,
                    "a1": afrac1,
                    "a2": afrac2,
                    "a3": afrac3,
                    "epsilon": epsilon,
                },
                "error": str(e),
            }
            if self.log_handle:
                self.log_handle.write(json.dumps(log_entry) + "\n")
                self.log_handle.flush()
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

        # Import the module and get the original function
        from main.geoms import circular_facet

        self.original_function = circular_facet.getArcFacet

        # Open log file for writing
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = open(log_path, "w")

        # Write header comment
        self.log_handle.write(f"# getArcFacet call log\n")
        self.log_handle.write(f"# Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_handle.write(f"# Format: One JSON object per line\n")
        self.log_handle.write(f"#\n")

        # Monkey patch the function
        circular_facet.getArcFacet = self.wrapper

        self.enabled = True
        print(f"✓ Logging enabled: calls will be logged to {self.log_file}")
        print(f"  Original function saved, ready to log calls.")

    def disable(self):
        """Disable logging and restore original function."""
        if not self.enabled:
            print("Warning: Logging is not currently enabled.")
            return

        # Restore original function
        from main.geoms import circular_facet

        circular_facet.getArcFacet = self.original_function

        # Close log file
        if self.log_handle:
            self.log_handle.write(
                f"# Logging stopped at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            self.log_handle.write(f"# Total calls logged: {self.call_count}\n")
            self.log_handle.close()
            self.log_handle = None

        self.enabled = False
        print(f"✓ Logging disabled: {self.call_count} calls logged to {self.log_file}")
        print(f"  Original function restored.")

    def get_stats(self) -> dict:
        """Get statistics about logged calls."""
        if not self.enabled or self.call_count == 0:
            return {"total_calls": 0}

        # Read the log file to get stats
        stats = {
            "total_calls": self.call_count,
            "log_file": self.log_file,
        }

        # Try to read the log file for more detailed stats
        try:
            log_path = Path(self.log_file)
            if log_path.exists():
                success_count = 0
                failed_count = 0
                exception_count = 0
                total_time = 0.0
                max_time = 0.0

                with open(log_path, "r") as f:
                    for line in f:
                        if line.strip().startswith("#"):
                            continue
                        try:
                            entry = json.loads(line)
                            status = entry.get("status", "unknown")
                            if status == "success":
                                success_count += 1
                            elif status == "failed":
                                failed_count += 1
                            elif status == "exception":
                                exception_count += 1

                            exec_time = entry.get("execution_time_seconds", 0)
                            total_time += exec_time
                            max_time = max(max_time, exec_time)
                        except json.JSONDecodeError:
                            continue

                stats.update(
                    {
                        "success_count": success_count,
                        "failed_count": failed_count,
                        "exception_count": exception_count,
                        "avg_execution_time": (
                            total_time / self.call_count if self.call_count > 0 else 0
                        ),
                        "max_execution_time": max_time,
                        "total_execution_time": total_time,
                    }
                )
        except Exception as e:
            stats["error"] = f"Could not read log file: {e}"

        return stats


# Global logger instance
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


