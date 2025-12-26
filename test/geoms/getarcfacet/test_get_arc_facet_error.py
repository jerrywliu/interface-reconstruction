"""
Test suite for getArcFacet function to debug infinite loops and convergence issues.

Usage:
    # Run all tests with pytest:
    python -m pytest test/geoms/getarcfacet/test_get_arc_facet_error.py -v

    # Run all tests and see detailed output:
    python -m test.geoms.getarcfacet.test_get_arc_facet_error

    # Run a specific test case by index:
    python -m test.geoms.getarcfacet.test_get_arc_facet_error 0

    # Run with verbose output for debugging:
    python -m pytest test/geoms/getarcfacet/test_get_arc_facet_error.py::test_get_arc_facet_convergence_pytest -v -s

Test Cases:
    - All test cases are defined in the TEST_CASES list
    - Each test case includes polygons, area fractions, and epsilon
    - Tests verify that getArcFacet either converges or fails gracefully (no infinite loops)

Features:
    - Timeout detection to catch infinite loops
    - Detailed logging for debugging
    - Summary report of all test results
    - Works with or without pytest installed
"""

import sys
import threading
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

try:
    import pytest
except ImportError:
    pytest = None

from main.geoms.circular_facet import getArcFacet


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


# Test cases extracted from error reports
# All floating point values are preserved to exact precision as they appeared in error reports
TEST_CASES = [
    # Original test case from error report
    TestCase(
        name="case_0",
        poly1=[[62.0, 50.0], [64.0, 50.0], [64.0, 52.0], [62.0, 52.0]],
        poly2=[[62.0, 52.0], [64.0, 52.0], [64.0, 54.0], [62.0, 54.0]],
        poly3=[[60.0, 52.0], [62.0, 52.0], [62.0, 54.0], [60.0, 54.0]],
        a1=0.9744726478258485,
        a2=0.34263693403664774,
        a3=0.9926595259164515,
        epsilon=1e-10,
        description="Original test case from error report",
    ),
    # Error case from error reports
    TestCase(
        name="case_1",
        poly1=[[46.0, 60.0], [48.0, 60.0], [48.0, 62.0], [46.0, 62.0]],
        poly2=[[46.0, 62.0], [48.0, 62.0], [48.0, 64.0], [46.0, 64.0]],
        poly3=[[48.0, 62.0], [50.0, 62.0], [50.0, 64.0], [48.0, 64.0]],
        a1=0.3964222994691795,
        a2=0.9982853955276596,
        a3=0.49414908878799224,
        epsilon=1e-10,
        description="Error case from error reports",
    ),
    # Error case from error reports
    TestCase(
        name="case_2",
        poly1=[[58.0, 42.0], [60.0, 42.0], [60.0, 44.0], [58.0, 44.0]],
        poly2=[[58.0, 44.0], [60.0, 44.0], [60.0, 46.0], [58.0, 46.0]],
        poly3=[[60.0, 44.0], [62.0, 44.0], [62.0, 46.0], [60.0, 46.0]],
        a1=0.4108604136960139,
        a2=0.9990973220492947,
        a3=0.5093991295365186,
        epsilon=1e-10,
        description="Error case from error reports",
    ),
    # Error case from error reports
    TestCase(
        name="case_3",
        poly1=[[60.0, 62.0], [62.0, 62.0], [62.0, 64.0], [60.0, 64.0]],
        poly2=[[58.0, 62.0], [60.0, 62.0], [60.0, 64.0], [58.0, 64.0]],
        poly3=[[56.0, 62.0], [58.0, 62.0], [58.0, 64.0], [56.0, 64.0]],
        a1=0.5350609117473368,
        a2=0.8048182938294985,
        a3=0.9932597699354346,
        epsilon=1e-10,
        description="Error case from error reports",
    ),
    # Error case from error reports
    TestCase(
        name="case_4",
        poly1=[[56.0, 64.0], [58.0, 64.0], [58.0, 66.0], [56.0, 66.0]],
        poly2=[[56.0, 62.0], [58.0, 62.0], [58.0, 64.0], [56.0, 64.0]],
        poly3=[[58.0, 62.0], [60.0, 62.0], [60.0, 64.0], [58.0, 64.0]],
        a1=0.9186840940238312,
        a2=0.006740230064565367,
        a3=0.19518170617050146,
        epsilon=1e-10,
        description="Error case from error reports",
    ),
]


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
        except Exception as e:
            result_container["exception"] = e
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

    Args:
        test_case: The test case to run
        timeout_seconds: Maximum time to wait for convergence
        verbose: Whether to print detailed information

    Returns:
        Dictionary with results including:
        - converged: bool
        - result: tuple (center, radius, intersects) or (None, None, None)
        - error: str if error occurred
        - timeout: bool if timeout occurred
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
            print(f"\n{'='*60}")
            print(f"Running test case: {test_case.name}")
            print(f"Description: {test_case.description}")
            print(f"Poly1: {test_case.poly1}")
            print(f"Poly2: {test_case.poly2}")
            print(f"Poly3: {test_case.poly3}")
            # Use repr() to preserve exact floating point precision
            print(
                f"Area fractions: a1={repr(test_case.a1)}, a2={repr(test_case.a2)}, a3={repr(test_case.a3)}"
            )
            print(f"Epsilon: {repr(test_case.epsilon)}")
            print(f"{'='*60}")

        center, radius, intersects = run_with_timeout(
            execute_get_arc_facet, timeout_seconds
        )

        result["result"] = (center, radius, intersects)

        if center is not None and radius is not None and intersects is not None:
            result["converged"] = True
            if verbose:
                print(f"✓ Converged: center={center}, radius={radius}")
                print(f"  Intersects: {intersects}")
        else:
            result["error"] = "Function returned None (convergence failed)"
            if verbose:
                print(f"✗ Failed to converge: function returned None")

    except TimeoutError as e:
        result["timeout"] = True
        result["error"] = str(e)
        if verbose:
            print(f"✗ TIMEOUT: {e}")
    except Exception as e:
        result["error"] = f"Exception: {str(e)}"
        if verbose:
            print(f"✗ EXCEPTION: {e}")

    return result


# Test functions - work with or without pytest
def test_get_arc_facet_convergence(test_case: TestCase, capfd=None):
    """
    Test that getArcFacet converges (or fails gracefully) for each test case.

    This test verifies that:
    1. The function doesn't hang indefinitely
    2. Either converges successfully or returns None (indicating failure)
    3. Doesn't throw unexpected exceptions
    """
    result = run_get_arc_facet_with_timeout(
        test_case, timeout_seconds=10, verbose=False
    )

    # Check that we didn't timeout (infinite loop)
    assert not result["timeout"], (
        f"Test case {test_case.name} timed out - possible infinite loop. "
        f"Description: {test_case.description}"
    )

    # Check that we got a result (either success or graceful failure)
    assert result["error"] is None or result["error"].startswith(
        "Function returned None"
    ), f"Test case {test_case.name} raised an exception: {result['error']}"

    # If expected_result is set, check it matches
    if test_case.expected_result == "converge":
        assert result[
            "converged"
        ], f"Test case {test_case.name} was expected to converge but failed"
    elif test_case.expected_result == "fail":
        assert not result[
            "converged"
        ], f"Test case {test_case.name} was expected to fail but converged"

    # Capture and check output for error messages if capfd is available
    if capfd is not None:
        out, err = capfd.readouterr()
        combined = out + err

        # If it didn't converge, we should see an error message
        if not result["converged"]:
            # Check for expected error messages
            has_error_msg = (
                "Error in getArcFacet(" in combined
                or "Max timesteps reached in getArcFacet(" in combined
            )
            # Note: We don't fail if there's no error message, as the function
            # might fail silently by returning None


def test_all_cases_with_summary(capfd=None):
    """
    Run all test cases and print a summary of results.
    Useful for debugging and understanding which cases fail.
    """
    print("\n" + "=" * 80)
    print("RUNNING ALL TEST CASES WITH DETAILED SUMMARY")
    print("=" * 80)

    results = []
    for test_case in TEST_CASES:
        result = run_get_arc_facet_with_timeout(
            test_case, timeout_seconds=10, verbose=True
        )
        results.append(result)

        status = "✓ CONVERGED" if result["converged"] else "✗ FAILED"
        if result["timeout"]:
            status = "⏱ TIMEOUT"

        print(f"\n{status}: {test_case.name}")
        if result["error"]:
            print(f"  Error: {result['error']}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    converged_count = sum(1 for r in results if r["converged"])
    failed_count = sum(1 for r in results if not r["converged"] and not r["timeout"])
    timeout_count = sum(1 for r in results if r["timeout"])

    print(f"Total test cases: {len(TEST_CASES)}")
    print(f"  ✓ Converged: {converged_count}")
    print(f"  ✗ Failed: {failed_count}")
    print(f"  ⏱ Timed out: {timeout_count}")

    if timeout_count > 0:
        print("\n⚠ WARNING: Some test cases timed out - possible infinite loops!")
        for result in results:
            if result["timeout"]:
                print(f"  - {result['test_case']}")

    if failed_count > 0:
        print("\nFailed test cases:")
        for result in results:
            if not result["converged"] and not result["timeout"]:
                print(
                    f"  - {result['test_case']}: {result.get('error', 'Unknown error')}"
                )


# Pytest-specific parametrization
# This allows pytest to discover and run tests with: pytest test_get_arc_facet_error.py -v
if pytest:
    # Use pytest.mark.parametrize to create tests for each case
    @pytest.mark.parametrize(
        "test_case", TEST_CASES, ids=[tc.name for tc in TEST_CASES]
    )
    def test_get_arc_facet_convergence_pytest(test_case: TestCase, capfd):
        """Pytest wrapper for test_get_arc_facet_convergence."""
        return test_get_arc_facet_convergence(test_case, capfd)


def test_single_case_debug():
    """
    Helper function to debug a single test case.
    Modify the index to test different cases.
    """
    test_case = TEST_CASES[0]  # Change index to test different cases
    result = run_get_arc_facet_with_timeout(test_case, timeout_seconds=30, verbose=True)

    print(f"\nResult: {result}")
    assert (
        result["converged"] or not result["timeout"]
    ), "Test case should converge or fail gracefully"


if __name__ == "__main__":
    # Run a specific test case for debugging
    if len(sys.argv) > 1:
        case_index = int(sys.argv[1])
        if 0 <= case_index < len(TEST_CASES):
            test_case = TEST_CASES[case_index]
            result = run_get_arc_facet_with_timeout(
                test_case, timeout_seconds=30, verbose=True
            )
            print(f"\nFinal result: {result}")
        else:
            print(
                f"Invalid test case index: {case_index}. Valid range: 0-{len(TEST_CASES)-1}"
            )
    else:
        # Run all test cases and show summary
        test_all_cases_with_summary(None)
