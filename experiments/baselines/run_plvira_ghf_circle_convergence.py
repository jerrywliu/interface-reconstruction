"""Deterministic Cartesian subset of Popinet's GHF circle study.

Popinet (2009), Section 6.1 and Figure 5, evaluates generalized-height-
function curvature on randomly translated circles.  This focused reproducer
uses fixed cell-relative translations, one radius, and uniform Cartesian
meshes.  It reports the same relative curvature norms and the frequency of
each source-algorithm fallback.  It does not exercise perturbed grids.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from main.algos.baselines.plvira_ghf import cartesian_ghf_curvature
from main.geoms.circular_facet import getCircleIntersectArea


DEFAULT_RESOLUTIONS = (32, 64, 128, 256)
DEFAULT_OFFSETS = ((-0.31, 0.17), (0.13, -0.27), (0.37, 0.41), (-0.43, -0.11))


def _cell_fraction(
    x0: float,
    y0: float,
    cell_size: float,
    center: Tuple[float, float],
    radius: float,
) -> float:
    x1, y1 = x0 + cell_size, y0 + cell_size
    radius_squared = radius * radius
    corner_distances = [
        (x - center[0]) ** 2 + (y - center[1]) ** 2 for x in (x0, x1) for y in (y0, y1)
    ]
    if max(corner_distances) <= radius_squared:
        return 1.0
    dx = 0.0 if x0 <= center[0] <= x1 else min(abs(center[0] - x0), abs(center[0] - x1))
    dy = 0.0 if y0 <= center[1] <= y1 else min(abs(center[1] - y0), abs(center[1] - y1))
    if dx * dx + dy * dy >= radius_squared:
        return 0.0
    polygon = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    area, _ = getCircleIntersectArea(center, radius, polygon)
    fraction = area / (cell_size * cell_size)
    if fraction <= 0.0:
        return 0.0
    if fraction >= 1.0:
        return 1.0
    return float(fraction)


def _circle_fractions(
    resolution: int,
    radius: float,
    offset: Tuple[float, float],
) -> Tuple[List[List[float]], Tuple[float, float]]:
    cell_size = 1.0 / resolution
    center = (offset[0] * cell_size, offset[1] * cell_size)
    fractions = []
    for row in range(resolution):
        y0 = -0.5 + row * cell_size
        fraction_row = []
        for column in range(resolution):
            x0 = -0.5 + column * cell_size
            fraction_row.append(_cell_fraction(x0, y0, cell_size, center, radius))
        fractions.append(fraction_row)
    return fractions, center


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _orders(rows: List[Dict[str, object]], field: str) -> None:
    previous = None
    for row in rows:
        if previous is None:
            row[field + "_order"] = None
        else:
            ratio = row["resolution"] / previous["resolution"]
            row[field + "_order"] = math.log(previous[field] / row[field]) / math.log(
                ratio
            )
        previous = row


def run(
    resolutions: Sequence[int],
    radius: float,
    offsets: Iterable[Tuple[float, float]],
) -> Dict[str, object]:
    offsets = tuple(offsets)
    exact_curvature = 1.0 / radius
    rows = []
    for resolution in resolutions:
        errors = []
        method_counts = {
            "height_function": 0,
            "mixed_height_parabola": 0,
            "plic_centroid_parabola": 0,
            "degenerate_zero": 0,
        }
        per_offset_cells = []
        for offset in offsets:
            fractions, _ = _circle_fractions(resolution, radius, offset)
            offset_cells = 0
            for row in range(2, resolution - 2):
                for column in range(2, resolution - 2):
                    fraction = fractions[row][column]
                    if not 0.0 < fraction < 1.0:
                        continue
                    result = cartesian_ghf_curvature(
                        fractions, (row, column), 1.0 / resolution
                    )
                    method_counts[result.method] += 1
                    errors.append(
                        abs(result.curvature - exact_curvature) / exact_curvature
                    )
                    offset_cells += 1
            per_offset_cells.append(offset_cells)
        if not errors:
            raise RuntimeError("circle study produced no mixed cells")
        rows.append(
            {
                "resolution": resolution,
                "radius_over_h": radius * resolution,
                "samples": len(errors),
                "mixed_cells_per_offset": per_offset_cells,
                "relative_l1": sum(errors) / len(errors),
                "relative_l2": math.sqrt(
                    sum(error * error for error in errors) / len(errors)
                ),
                "relative_linf": max(errors),
                "method_counts": method_counts,
            }
        )
    for field in ("relative_l1", "relative_l2", "relative_linf"):
        _orders(rows, field)
    source = Path(__file__).resolve().parents[2] / "main/algos/baselines/plvira_ghf.py"
    return {
        "study": "Popinet 2009 Section 6.1/Figure 5 deterministic 2D subset",
        "scope": "uniform Cartesian circles; no perturbed grids",
        "radius": radius,
        "exact_curvature": exact_curvature,
        "domain": [-0.5, 0.5, -0.5, 0.5],
        "offsets_in_cell_widths": [list(offset) for offset in offsets],
        "fraction_initialization": "analytic circle-square intersection",
        "curvature_source": "cartesian-ghf",
        "implementation_sha256": _sha256(source),
        "rows": rows,
    }


def _write_outputs(payload: Dict[str, object], output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")

    fieldnames = [
        "resolution",
        "radius_over_h",
        "samples",
        "relative_l1",
        "relative_l1_order",
        "relative_l2",
        "relative_l2_order",
        "relative_linf",
        "relative_linf_order",
        "height_function",
        "mixed_height_parabola",
        "plic_centroid_parabola",
        "degenerate_zero",
    ]
    with (output_directory / "results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["rows"]:
            writer.writerow(
                {
                    **{key: row.get(key) for key in fieldnames},
                    **row["method_counts"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolutions", nargs="+", type=int, default=DEFAULT_RESOLUTIONS
    )
    parser.add_argument("--radius", type=float, default=0.2)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("experiments/baselines/results/plvira_ghf_circle"),
    )
    arguments = parser.parse_args()
    if any(resolution < 8 for resolution in arguments.resolutions):
        parser.error("all resolutions must be at least 8")
    if not 0.0 < arguments.radius < 0.5:
        parser.error("radius must lie in (0, 0.5)")
    payload = run(arguments.resolutions, arguments.radius, DEFAULT_OFFSETS)
    _write_outputs(payload, arguments.output_directory)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
