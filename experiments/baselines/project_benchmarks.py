"""Canonical Cartesian fixtures for the five static project benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from experiments.static.zalesak import (
    create_true_facets_zalesak,
    initialize_zalesak,
    rotate_point_around_center,
)
from main.algos.baselines.external_geometry import (
    ExternalArcPrimitive,
    ExternalLinePrimitive,
    ExternalPrimitive,
)
from main.geoms.geoms import getPolyLineIntersects
from main.structs.facets.circular_facet import ArcFacet
from main.structs.meshes.base_mesh import BaseMesh
from util.initialize.areas import (
    initializeCircle,
    initializeEllipse,
    initializeLine,
    initializePoly,
)
from util.initialize.points import makeFineCartesianGrid


DOMAIN_SIZE = 100.0
CANONICAL_CASE_COUNT = 25
DEFAULT_BENCHMARKS = ("lines", "squares", "circles", "ellipses", "zalesak")
DEFAULT_CASE_INDICES = (0, 1, 2, 3, 4)
DEFAULT_RESOLUTIONS = (32, 64, 128)
MIXED_TOLERANCE = 1.0e-10

_SEEDS = {
    "lines": 42,
    "squares": 42,
    "circles": 41,
    "ellipses": 42,
    "zalesak": 43,
}


def _point(value: Sequence[float]) -> Tuple[float, float]:
    return float(value[0]), float(value[1])


def _arc_primitive(facet: ArcFacet) -> ExternalArcPrimitive:
    points = facet.sample(3)
    start = math.atan2(
        points[0][1] - facet.center[1], points[0][0] - facet.center[0]
    )
    middle = math.atan2(
        points[1][1] - facet.center[1], points[1][0] - facet.center[0]
    )
    end = math.atan2(
        points[2][1] - facet.center[1], points[2][0] - facet.center[0]
    )
    ccw = (end - start) % (2.0 * math.pi)
    middle_ccw = (middle - start) % (2.0 * math.pi)
    sweep = ccw if middle_ccw <= ccw + 1.0e-12 else ccw - 2.0 * math.pi
    return ExternalArcPrimitive(
        _point(facet.center), abs(float(facet.radius)), start, sweep
    )


@dataclass(frozen=True)
class ProjectBenchmarkCase:
    """One exact member of a canonical 25-case static benchmark sequence."""

    benchmark: str
    case_index: int
    random_seed: int
    parameters: Mapping[str, Any]

    @property
    def cell_count_reference(self) -> int:
        return CANONICAL_CASE_COUNT

    def build_mesh(self, cells_per_side: int) -> BaseMesh:
        if cells_per_side <= 0:
            raise ValueError("cells_per_side must be positive")
        resolution = cells_per_side / DOMAIN_SIZE
        points = makeFineCartesianGrid(DOMAIN_SIZE, resolution)
        return BaseMesh(points, MIXED_TOLERANCE)

    def initialize_fractions(self, mesh: BaseMesh) -> list:
        p = self.parameters
        if self.benchmark == "lines":
            fractions = initializeLine(mesh, p["p_left"], p["p_right"])
        elif self.benchmark == "squares":
            fractions = initializePoly(mesh, p["vertices"])
        elif self.benchmark == "circles":
            fractions = initializeCircle(mesh, p["center"], p["radius"])
        elif self.benchmark == "ellipses":
            fractions = initializeEllipse(
                mesh,
                p["major_axis"],
                p["minor_axis"],
                p["theta"],
                p["center"],
            )
        elif self.benchmark == "zalesak":
            fractions = initialize_zalesak(
                mesh,
                p["center"],
                p["radius"],
                p["slot_width"],
                y_top_rel=p["slot_top_rel"],
                theta=p["theta"],
            )
        else:
            raise ValueError(f"unknown benchmark {self.benchmark!r}")
        mesh.initializeFractions(fractions)
        return fractions

    def truth_primitives(self, ellipse_segments: int = 720) -> Tuple[ExternalPrimitive, ...]:
        p = self.parameters
        if self.benchmark == "lines":
            domain = ((0.0, 0.0), (DOMAIN_SIZE, 0.0), (DOMAIN_SIZE, DOMAIN_SIZE), (0.0, DOMAIN_SIZE))
            intersections = getPolyLineIntersects(domain, p["p_left"], p["p_right"])
            if len(intersections) != 2:
                raise RuntimeError("canonical line did not cross the domain twice")
            return (ExternalLinePrimitive(intersections[0], intersections[1]),)
        if self.benchmark == "squares":
            vertices = p["vertices"]
            return tuple(
                ExternalLinePrimitive(vertices[i], vertices[(i + 1) % 4])
                for i in range(4)
            )
        if self.benchmark == "circles":
            center = p["center"]
            radius = p["radius"]
            return (
                ExternalArcPrimitive(center, radius, 0.0, math.pi),
                ExternalArcPrimitive(center, radius, math.pi, math.pi),
            )
        if self.benchmark == "ellipses":
            if ellipse_segments < 32:
                raise ValueError("ellipse_segments must be at least 32")
            center = np.asarray(p["center"], dtype=float)
            cosine, sine = math.cos(p["theta"]), math.sin(p["theta"])
            points = []
            for index in range(ellipse_segments + 1):
                angle = 2.0 * math.pi * index / ellipse_segments
                local_x = p["major_axis"] * math.cos(angle)
                local_y = p["minor_axis"] * math.sin(angle)
                points.append(
                    (
                        center[0] + cosine * local_x - sine * local_y,
                        center[1] + sine * local_x + cosine * local_y,
                    )
                )
            return tuple(
                ExternalLinePrimitive(points[i], points[i + 1], {"truth": "sampled ellipse"})
                for i in range(ellipse_segments)
            )
        if self.benchmark == "zalesak":
            facets = create_true_facets_zalesak(
                p["center"], p["radius"], p["slot_vertices"], p["theta"]
            )
            return tuple(
                _arc_primitive(facet)
                if isinstance(facet, ArcFacet)
                else ExternalLinePrimitive(facet.pLeft, facet.pRight)
                for facet in facets
            )
        raise ValueError(f"unknown benchmark {self.benchmark!r}")

    def true_curvature_at(self, point: Sequence[float]) -> float:
        """Return unsigned analytic curvature at the nearest regular branch."""

        p = self.parameters
        if self.benchmark in ("lines", "squares"):
            return 0.0
        if self.benchmark == "circles":
            return 1.0 / p["radius"]
        if self.benchmark == "ellipses":
            dx = float(point[0]) - p["center"][0]
            dy = float(point[1]) - p["center"][1]
            cosine, sine = math.cos(p["theta"]), math.sin(p["theta"])
            local_x = cosine * dx + sine * dy
            local_y = -sine * dx + cosine * dy
            angle = math.atan2(local_y / p["minor_axis"], local_x / p["major_axis"])
            denominator = (
                p["major_axis"] ** 2 * math.sin(angle) ** 2
                + p["minor_axis"] ** 2 * math.cos(angle) ** 2
            ) ** 1.5
            return p["major_axis"] * p["minor_axis"] / denominator
        if self.benchmark == "zalesak":
            truth = self.truth_primitives()
            nearest = min(truth, key=lambda primitive: primitive.distance_to_point(point))
            return 1.0 / p["radius"] if isinstance(nearest, ExternalArcPrimitive) else 0.0
        raise ValueError(f"unknown benchmark {self.benchmark!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "case_index": self.case_index,
            "canonical_case_count": CANONICAL_CASE_COUNT,
            "random_seed": self.random_seed,
            "parameters": dict(self.parameters),
        }


def canonical_benchmark_cases(
    benchmark: str, case_indices: Sequence[int] = DEFAULT_CASE_INDICES
) -> Tuple[ProjectBenchmarkCase, ...]:
    """Return selected cases while advancing the exact full-sequence RNG."""

    if benchmark not in DEFAULT_BENCHMARKS:
        raise ValueError(f"unknown benchmark {benchmark!r}")
    requested = set(int(index) for index in case_indices)
    if any(index < 0 or index >= CANONICAL_CASE_COUNT for index in requested):
        raise ValueError("case indices must lie in [0, 24]")
    rng = np.random.default_rng(_SEEDS[benchmark])
    cases = []
    for index in range(CANONICAL_CASE_COUNT):
        if benchmark == "lines":
            angle = float(np.linspace(0.0, 2.0 * math.pi, CANONICAL_CASE_COUNT + 1)[:-1][index])
            x1, y1 = float(rng.uniform(50, 51)), float(rng.uniform(50, 51))
            parameters = {
                "angle": angle,
                "p_left": [x1, y1],
                "p_right": [x1 + 0.2, y1 + math.tan(angle) * 0.2],
            }
        elif benchmark == "squares":
            side_length = float(np.linspace(10.0, 30.0, CANONICAL_CASE_COUNT)[index])
            center = [float(rng.uniform(50, 51)), float(rng.uniform(50, 51))]
            theta = float(rng.uniform(0.0, math.pi / 2.0))
            half = 0.5 * side_length
            cosine, sine = math.cos(theta), math.sin(theta)
            vertices = []
            for x, y in ((-half, -half), (half, -half), (half, half), (-half, half)):
                vertices.append(
                    [center[0] + cosine * x - sine * y, center[1] + sine * x + cosine * y]
                )
            parameters = {
                "center": center,
                "side_length": side_length,
                "theta": theta,
                "vertices": vertices,
            }
        elif benchmark == "circles":
            parameters = {
                "center": [float(rng.uniform(50, 51)), float(rng.uniform(50, 51))],
                "radius": 10.0,
            }
        elif benchmark == "ellipses":
            aspect_ratio = float(np.linspace(1.5, 3.0, CANONICAL_CASE_COUNT)[index])
            parameters = {
                "center": [float(rng.uniform(50, 51)), float(rng.uniform(50, 51))],
                "major_axis": 30.0,
                "minor_axis": 30.0 / aspect_ratio,
                "aspect_ratio": aspect_ratio,
                "theta": float(rng.uniform(0.0, math.pi / 2.0)),
            }
        else:
            center = [float(rng.uniform(50, 51)), float(rng.uniform(50, 51))]
            theta = float(rng.uniform(0.0, math.pi / 2.0))
            radius, slot_width, slot_top_rel = 15.0, 5.0, 10.0
            slot = [
                [center[0] - slot_width / 2.0, center[1] - radius - 1.0e-6],
                [center[0] + slot_width / 2.0, center[1] - radius - 1.0e-6],
                [center[0] + slot_width / 2.0, center[1] + slot_top_rel],
                [center[0] - slot_width / 2.0, center[1] + slot_top_rel],
            ]
            parameters = {
                "center": center,
                "radius": radius,
                "slot_width": slot_width,
                "slot_top_rel": slot_top_rel,
                "theta": theta,
                "slot_vertices": [rotate_point_around_center(point, center, theta) for point in slot],
            }
        if index in requested:
            cases.append(
                ProjectBenchmarkCase(benchmark, index, _SEEDS[benchmark], parameters)
            )
    return tuple(cases)


__all__ = [
    "CANONICAL_CASE_COUNT",
    "DEFAULT_BENCHMARKS",
    "DEFAULT_CASE_INDICES",
    "DEFAULT_RESOLUTIONS",
    "DOMAIN_SIZE",
    "MIXED_TOLERANCE",
    "ProjectBenchmarkCase",
    "canonical_benchmark_cases",
]
