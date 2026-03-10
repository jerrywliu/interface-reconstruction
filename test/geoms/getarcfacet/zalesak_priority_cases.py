"""Curated high-signal subset of harvested Zalesak getArcFacet cases."""

from test.geoms.getarcfacet.zalesak_harvest_cases import TEST_CASES as HARVEST_CASES

PRIORITY_CASE_NAMES = [
    "zalesak_circular_r0p64_call545",
    "zalesak_circular_r0p64_call1035",
    "zalesak_circular_r0p64_call1532",
    "zalesak_circular_r0p64_call41",
    "zalesak_circular_r0p64_call544",
    "zalesak_circular_r0p64_call2041",
    "zalesak_circular_r0p64_call2030",
    "zalesak_circular_r0p64_call2027",
    "zalesak_circular_r0p50_call530",
]

_CASES_BY_NAME = {case.name: case for case in HARVEST_CASES}
TEST_CASES = [_CASES_BY_NAME[name] for name in PRIORITY_CASE_NAMES]
