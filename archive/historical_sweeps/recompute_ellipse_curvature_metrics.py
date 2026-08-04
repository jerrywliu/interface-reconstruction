"""
Recompute ellipse curvature-error metrics from saved reconstructed facet VTP files.

This utility recalculates per-ellipse mean absolute curvature error for completed
ellipse runs under `plots/` without rerunning reconstruction.
"""

import argparse
import glob
import math
from pathlib import Path

import numpy as np
import vtk


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute ellipse curvature-error metrics from saved VTP facets."
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="plots/perturb_sweep_ellipses_*",
        help="Glob for run directories.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed used by experiments/static/ellipses.py.",
    )
    parser.add_argument(
        "--major_axis",
        type=float,
        default=30.0,
        help="Ellipse major axis used by experiments/static/ellipses.py.",
    )
    parser.add_argument(
        "--aspect_min",
        type=float,
        default=1.5,
        help="Minimum aspect ratio in sweep.",
    )
    parser.add_argument(
        "--aspect_max",
        type=float,
        default=3.0,
        help="Maximum aspect ratio in sweep.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["corrected", "legacy"],
        default="corrected",
        help="Curvature reference mode: corrected uses ellipse parameter angle.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="curvature_error_corrected.txt",
        help="Filename to write under each run's metrics directory.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace metrics/curvature_error.txt in place.",
    )
    return parser.parse_args()


def get_ellipse_to_circle_matrix(major_axis, minor_axis, theta):
    return np.linalg.inv(
        np.array(
            [
                [
                    major_axis * math.cos(theta) ** 2
                    + minor_axis * math.sin(theta) ** 2,
                    (major_axis - minor_axis) * math.cos(theta) * math.sin(theta),
                ],
                [
                    (major_axis - minor_axis) * math.cos(theta) * math.sin(theta),
                    major_axis * math.sin(theta) ** 2
                    + minor_axis * math.cos(theta) ** 2,
                ],
            ]
        )
    )


def ellipse_true_curvature(major_axis, minor_axis, t):
    denom = (
        major_axis**2 * math.sin(t) ** 2 + minor_axis**2 * math.cos(t) ** 2
    ) ** (3 / 2)
    return (major_axis * minor_axis) / denom


def facet_curvature_from_points(points):
    if points.shape[0] < 3:
        return 0.0
    p1 = points[0]
    p2 = points[points.shape[0] // 2]
    p3 = points[-1]
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)
    denom = a * b * c
    if denom <= 0.0:
        return 0.0
    cross = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
    if cross <= 1e-15:
        return 0.0
    return 2.0 * cross / denom


def read_facets(vtp_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    poly = reader.GetOutput()
    for cell_id in range(poly.GetNumberOfCells()):
        cell = poly.GetCell(cell_id)
        ids = cell.GetPointIds()
        points = np.array(
            [[poly.GetPoint(ids.GetId(j))[0], poly.GetPoint(ids.GetId(j))[1]] for j in range(ids.GetNumberOfIds())]
        )
        yield points


def ellipse_centers_thetas(num_ellipses, seed):
    rng = np.random.default_rng(seed)
    centers = []
    thetas = []
    for _ in range(num_ellipses):
        centers.append([rng.uniform(50, 51), rng.uniform(50, 51)])
        thetas.append(rng.uniform(0, math.pi / 2))
    return centers, thetas


def compute_run_curvature_errors(
    run_dir,
    major_axis,
    aspect_min,
    aspect_max,
    seed,
    mode,
):
    facet_dir = Path(run_dir) / "vtk" / "reconstructed" / "facets"
    if not facet_dir.exists():
        return None

    facet_files = sorted(
        facet_dir.glob("*.vtp"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )
    if not facet_files:
        return None

    num_ellipses = len(facet_files)
    aspect_ratios = np.linspace(aspect_min, aspect_max, num_ellipses)
    centers, thetas = ellipse_centers_thetas(num_ellipses, seed)
    per_ellipse_errors = []

    for idx, facet_file in enumerate(facet_files):
        center = np.array(centers[idx])
        theta = thetas[idx]
        minor_axis = major_axis / aspect_ratios[idx]
        ellipse_to_circle = get_ellipse_to_circle_matrix(major_axis, minor_axis, theta)
        total_error = 0.0
        total_facets = 0

        for points in read_facets(facet_file):
            facet_center = 0.5 * (points[0] + points[-1])
            if mode == "legacy":
                dx = facet_center[0] - center[0]
                dy = facet_center[1] - center[1]
                t = math.atan2(dy, dx) - theta
            else:
                mapped = ellipse_to_circle @ (facet_center - center)
                t = math.atan2(mapped[1], mapped[0])

            true_kappa = ellipse_true_curvature(major_axis, minor_axis, t)
            recon_kappa = facet_curvature_from_points(points)
            total_error += abs(recon_kappa - true_kappa)
            total_facets += 1

        if total_facets == 0:
            per_ellipse_errors.append(0.0)
        else:
            per_ellipse_errors.append(total_error / total_facets)

    return per_ellipse_errors


def write_metrics(run_dir, values, output_name, replace):
    metrics_dir = Path(run_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = metrics_dir / output_name
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.writelines(f"{value}\n" for value in values)

    if replace:
        replace_path = metrics_dir / "curvature_error.txt"
        with open(replace_path, "w", encoding="utf-8") as handle:
            handle.writelines(f"{value}\n" for value in values)


def main():
    args = parse_args()
    run_dirs = sorted(glob.glob(args.glob))
    if not run_dirs:
        print(f"No run directories matched glob: {args.glob}")
        return

    processed = 0
    skipped = 0
    for run_dir in run_dirs:
        values = compute_run_curvature_errors(
            run_dir=run_dir,
            major_axis=args.major_axis,
            aspect_min=args.aspect_min,
            aspect_max=args.aspect_max,
            seed=args.seed,
            mode=args.mode,
        )
        if values is None:
            skipped += 1
            print(f"Skipped (no facets): {run_dir}")
            continue
        write_metrics(
            run_dir=run_dir,
            values=values,
            output_name=args.output_name,
            replace=args.replace,
        )
        processed += 1
        print(f"Updated {run_dir} ({len(values)} ellipse metrics)")

    print(f"Done. processed={processed}, skipped={skipped}, mode={args.mode}")


if __name__ == "__main__":
    main()
