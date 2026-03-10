"""
Synthetic tests for getArcFacet harvest logging and analysis helpers.
"""

from util.logging.analyze_log import (
    consolidate_log_entries,
    dedupe_extracted_cases,
    filter_entries,
    limit_slow_success_cases,
    render_test_case_module,
)
from util.logging.get_arc_facet_logger import (
    arc_facet_log_context,
    get_current_log_context,
)


def test_nested_log_context_merges():
    assert get_current_log_context() == {}
    with arc_facet_log_context(experiment="zalesak", algo="safe_circle"):
        assert get_current_log_context() == {
            "experiment": "zalesak",
            "algo": "safe_circle",
        }
        with arc_facet_log_context(case_index=3, merge_id=12):
            assert get_current_log_context() == {
                "experiment": "zalesak",
                "algo": "safe_circle",
                "case_index": 3,
                "merge_id": 12,
            }
        assert get_current_log_context() == {
            "experiment": "zalesak",
            "algo": "safe_circle",
        }
    assert get_current_log_context() == {}


def test_consolidate_and_limit_problematic_cases():
    raw_entries = [
        {
            "call_id": 1,
            "timestamp": 1.0,
            "status": "started",
            "inputs": {"poly1": [[0, 0]], "poly2": [[1, 0]], "poly3": [[2, 0]], "a1": 0.1, "a2": 0.2, "a3": 0.3, "epsilon": 1e-10},
            "metadata": {"experiment": "zalesak", "algo": "safe_circle", "resolution": 0.5, "call_source": "safe_circle"},
        },
        {
            "call_id": 2,
            "timestamp": 2.0,
            "status": "started",
            "inputs": {"poly1": [[0, 1]], "poly2": [[1, 1]], "poly3": [[2, 1]], "a1": 0.4, "a2": 0.5, "a3": 0.6, "epsilon": 1e-10},
            "metadata": {"experiment": "zalesak", "algo": "safe_circle", "resolution": 0.5, "call_source": "safe_circle"},
        },
        {
            "call_id": 2,
            "timestamp": 2.2,
            "status": "failed",
            "execution_time_seconds": 0.2,
            "inputs": {"poly1": [[0, 1]], "poly2": [[1, 1]], "poly3": [[2, 1]], "a1": 0.4, "a2": 0.5, "a3": 0.6, "epsilon": 1e-10},
            "metadata": {"experiment": "zalesak", "algo": "safe_circle", "resolution": 0.5, "call_source": "safe_circle"},
        },
    ]

    for call_id in range(3, 15):
        raw_entries.extend(
            [
                {
                    "call_id": call_id,
                    "timestamp": float(call_id),
                    "status": "started",
                    "inputs": {
                        "poly1": [[call_id, 0]],
                        "poly2": [[call_id, 1]],
                        "poly3": [[call_id, 2]],
                        "a1": 0.1,
                        "a2": 0.2,
                        "a3": 0.3 + 0.001 * call_id,
                        "epsilon": 1e-10,
                    },
                    "metadata": {
                        "experiment": "zalesak",
                        "algo": "circular",
                        "resolution": 0.64,
                        "call_source": "circular",
                    },
                },
                {
                    "call_id": call_id,
                    "timestamp": float(call_id) + 0.1,
                    "status": "success",
                    "execution_time_seconds": float(call_id),
                    "inputs": {
                        "poly1": [[call_id, 0]],
                        "poly2": [[call_id, 1]],
                        "poly3": [[call_id, 2]],
                        "a1": 0.1,
                        "a2": 0.2,
                        "a3": 0.3 + 0.001 * call_id,
                        "epsilon": 1e-10,
                    },
                    "metadata": {
                        "experiment": "zalesak",
                        "algo": "circular",
                        "resolution": 0.64,
                        "call_source": "circular",
                    },
                    "outputs": {"center": [0.0, 0.0], "radius": 1.0, "num_intersects": 2},
                },
            ]
        )

    consolidated = consolidate_log_entries(raw_entries)
    assert len(consolidated) == 14
    assert consolidated[0]["status"] == "started_only"
    assert consolidated[1]["status"] == "failed"

    filtered = filter_entries(
        consolidated,
        experiment=["zalesak"],
        algo=["circular"],
        resolution=[0.64],
        source=["circular"],
    )
    assert len(filtered) == 12

    deduped = dedupe_extracted_cases(filtered + filtered[:2])
    assert len(deduped) == 12

    limited = limit_slow_success_cases(consolidated, per_bucket_limit=10)
    success_kept = [entry for entry in limited if entry.get("status") == "success"]
    assert len(success_kept) == 10
    assert any(entry.get("status") == "started_only" for entry in limited)
    assert any(entry.get("status") == "failed" for entry in limited)

    module_text = render_test_case_module(limited[:2], infinite_loop_threshold=0.01)
    assert "from test.geoms.getarcfacet.case_harness import TestCase" in module_text
    assert "TEST_CASES = [" in module_text
    compile(module_text, "<harvest_cases>", "exec")
