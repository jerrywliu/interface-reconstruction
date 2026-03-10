"""
Seed getArcFacet regression cases extracted from earlier error reports.
"""

from test.geoms.getarcfacet.case_harness import TestCase


TEST_CASES = [
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
        source_suite="seed_cases",
    ),
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
        source_suite="seed_cases",
    ),
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
        source_suite="seed_cases",
    ),
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
        source_suite="seed_cases",
    ),
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
        source_suite="seed_cases",
    ),
]
