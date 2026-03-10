"""
Test suite for getArcFacet function to debug infinite loops and convergence issues.

Usage:
    python -m pytest test/geoms/getarcfacet/test_get_arc_facet_error.py -v
    python -m test.geoms.getarcfacet.test_get_arc_facet_error
    python -m test.geoms.getarcfacet.test_get_arc_facet_error 0
"""

import sys

try:
    import pytest
except ImportError:
    pytest = None

from test.geoms.getarcfacet.case_harness import run_get_arc_facet_with_timeout
from test.geoms.getarcfacet.seed_cases import TEST_CASES as SEED_CASES

try:
    from test.geoms.getarcfacet.zalesak_harvest_cases import TEST_CASES as HARVEST_CASES
except ImportError:
    HARVEST_CASES = []

try:
    from test.geoms.getarcfacet.zalesak_priority_cases import TEST_CASES as PRIORITY_CASES
except ImportError:
    PRIORITY_CASES = []

CASE_SUITES = {
    "seed": list(SEED_CASES),
    "harvest": list(HARVEST_CASES),
    "priority": list(PRIORITY_CASES),
    "all": list(SEED_CASES) + list(HARVEST_CASES),
}

TEST_CASES = CASE_SUITES["all"]


def test_get_arc_facet_convergence(test_case, capfd=None):
    """
    Test that getArcFacet converges (or fails gracefully) for each test case.
    """
    result = run_get_arc_facet_with_timeout(
        test_case, timeout_seconds=10, verbose=False
    )

    assert not result["timeout"], (
        f"Test case {test_case.name} timed out - possible infinite loop. "
        f"Description: {test_case.description}"
    )

    assert result["error"] is None or result["error"].startswith(
        "Function returned None"
    ), f"Test case {test_case.name} raised an exception: {result['error']}"

    if test_case.expected_result == "converge":
        assert result[
            "converged"
        ], f"Test case {test_case.name} was expected to converge but failed"
    elif test_case.expected_result == "fail":
        assert not result[
            "converged"
        ], f"Test case {test_case.name} was expected to fail but converged"

    if capfd is not None:
        out, err = capfd.readouterr()
        combined = out + err
        if not result["converged"]:
            _ = (
                "Error in getArcFacet(" in combined
                or "Max timesteps reached in getArcFacet(" in combined
            )


def test_all_cases_with_summary(capfd=None, cases=None):
    """
    Run all test cases and print a summary of results.
    """
    cases = TEST_CASES if cases is None else cases
    print("\n" + "=" * 80)
    print("RUNNING ALL TEST CASES WITH DETAILED SUMMARY")
    print("=" * 80)

    results = []
    for test_case in cases:
        result = run_get_arc_facet_with_timeout(
            test_case, timeout_seconds=10, verbose=True
        )
        results.append(result)

        status = "CONVERGED" if result["converged"] else "FAILED"
        if result["timeout"]:
            status = "TIMEOUT"

        print(f"\n{status}: {test_case.name}")
        if result["error"]:
            print(f"  Error: {result['error']}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    converged_count = sum(1 for result in results if result["converged"])
    failed_count = sum(
        1 for result in results if not result["converged"] and not result["timeout"]
    )
    timeout_count = sum(1 for result in results if result["timeout"])

    print(f"Total test cases: {len(cases)}")
    print(f"  Converged: {converged_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Timed out: {timeout_count}")

    if timeout_count > 0:
        print("\nWarning: Some test cases timed out - possible infinite loops")
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


if pytest:

    @pytest.mark.parametrize(
        "test_case", TEST_CASES, ids=[test_case.name for test_case in TEST_CASES]
    )
    def test_get_arc_facet_convergence_pytest(test_case, capfd):
        return test_get_arc_facet_convergence(test_case, capfd)


def test_single_case_debug():
    """
    Helper function to debug a single test case.
    """
    test_case = TEST_CASES[0]
    result = run_get_arc_facet_with_timeout(test_case, timeout_seconds=30, verbose=True)

    print(f"\nResult: {result}")
    assert (
        result["converged"] or not result["timeout"]
    ), "Test case should converge or fail gracefully"


def _parse_cli_args(argv):
    suite_name = "all"
    case_index = None
    remaining = list(argv[1:])

    if remaining and not remaining[0].lstrip("-").isdigit():
        suite_name = remaining.pop(0)

    if remaining:
        case_index = int(remaining.pop(0))

    return suite_name, case_index


if __name__ == "__main__":
    suite_name, case_index = _parse_cli_args(sys.argv)
    if suite_name not in CASE_SUITES:
        print(
            f"Invalid suite: {suite_name}. "
            f"Valid suites: {', '.join(sorted(CASE_SUITES))}"
        )
        sys.exit(1)

    selected_cases = CASE_SUITES[suite_name]

    if case_index is not None:
        if 0 <= case_index < len(selected_cases):
            test_case = selected_cases[case_index]
            result = run_get_arc_facet_with_timeout(
                test_case, timeout_seconds=30, verbose=True
            )
            print(f"\nFinal result: {result}")
        else:
            print(
                f"Invalid test case index: {case_index}. "
                f"Valid range: 0-{len(selected_cases)-1}"
            )
    else:
        test_all_cases_with_summary(None, selected_cases)
