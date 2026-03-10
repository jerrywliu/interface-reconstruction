"""
Shared harness for getArcFacet regression cases.
"""

from dataclasses import dataclass, field
import threading
from typing import Any, Dict, List, Optional

from main.geoms.circular_facet import LinearFacetShortcut, getArcFacet


@dataclass
class TestCase:
    """Represents a single test case for getArcFacet."""

    name: str
    poly1: List[List[float]]
    poly2: List[List[float]]
    poly3: List[List[float]]
    a1: float
    a2: float
    a3: float
    epsilon: float = 1e-10
    expected_result: Optional[str] = None  # "converge", "fail", or None
    description: str = ""
    source_suite: str = "seed_cases"
    metadata: Dict[str, Any] = field(default_factory=dict)


class TimeoutError(Exception):
    """Raised when a test case times out."""

    pass


def run_with_timeout(func, timeout_seconds):
    """
    Run a function with a timeout using threading.
    Returns the result or raises TimeoutError.
    """
    result_container = {"result": None, "exception": None, "completed": False}

    def target():
        try:
            result_container["result"] = func()
            result_container["completed"] = True
        except Exception as error:
            result_container["exception"] = error
            result_container["completed"] = True

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_seconds)

    if not result_container["completed"]:
        raise TimeoutError(
            f"Function execution timed out after {timeout_seconds} seconds"
        )

    if result_container["exception"]:
        raise result_container["exception"]

    return result_container["result"]


def run_get_arc_facet_with_timeout(
    test_case: TestCase, timeout_seconds: int = 10, verbose: bool = False
) -> Dict[str, Any]:
    """
    Run getArcFacet with a timeout and return detailed results.
    """
    result = {
        "converged": False,
        "result": (None, None, None),
        "error": None,
        "timeout": False,
        "test_case": test_case.name,
    }

    def execute_get_arc_facet():
        return getArcFacet(
            test_case.poly1,
            test_case.poly2,
            test_case.poly3,
            test_case.a1,
            test_case.a2,
            test_case.a3,
            test_case.epsilon,
        )

    try:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Running test case: {test_case.name}")
            print(f"Description: {test_case.description}")
            print(f"Source suite: {test_case.source_suite}")
            if test_case.metadata:
                print(f"Metadata: {test_case.metadata}")
            print(f"Poly1: {test_case.poly1}")
            print(f"Poly2: {test_case.poly2}")
            print(f"Poly3: {test_case.poly3}")
            print(
                "Area fractions: "
                f"a1={repr(test_case.a1)}, "
                f"a2={repr(test_case.a2)}, "
                f"a3={repr(test_case.a3)}"
            )
            print(f"Epsilon: {repr(test_case.epsilon)}")
            print(f"{'=' * 60}")

        center, radius, intersects = run_with_timeout(
            execute_get_arc_facet, timeout_seconds
        )

        result["result"] = (center, radius, intersects)

        if center is not None and radius is not None and intersects is not None:
            result["converged"] = True
            if verbose:
                print(f"Converged: center={center}, radius={radius}")
                print(f"Intersects: {intersects}")
        else:
            result["error"] = "Function returned None (convergence failed)"
            if verbose:
                print("Failed to converge: function returned None")

    except TimeoutError as error:
        result["timeout"] = True
        result["error"] = str(error)
        if verbose:
            print(f"TIMEOUT: {error}")
    except LinearFacetShortcut as shortcut:
        result["converged"] = True
        result["result"] = ("linear", shortcut.pLeft, shortcut.pRight)
        if verbose:
            print(
                "Linear shortcut: "
                f"pLeft={shortcut.pLeft}, pRight={shortcut.pRight}, "
                f"linear_fraction={shortcut.linear_fraction:.6e}, "
                f"target_fraction={shortcut.target_fraction:.6e}"
            )
    except Exception as error:
        result["error"] = f"Exception: {str(error)}"
        if verbose:
            print(f"EXCEPTION: {error}")

    return result
