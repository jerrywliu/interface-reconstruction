#!/usr/bin/env python3
"""
Regression coverage for merged-cell orientation edge cases.
"""

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.static.zalesak import RANDOM_SEED, initialize_zalesak
from main.structs.meshes.merge_mesh import MergeMesh
from util.config import read_yaml
from util.initialize.mesh_factory import apply_mesh_overrides, make_points_from_config


def test_corner_behavior_profiles_configure_mesh_and_polygons():
    points = [
        [[0.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [1.0, 1.0]],
    ]
    mesh = MergeMesh(points, 1e-12)
    poly = mesh.polys[0][0]

    assert mesh.corner_behavior_profile == "pre_f8_corner"
    assert not mesh.use_three_neighbor_orientation_hint
    assert not mesh.use_late_three_neighbor_orientation_hint
    assert not mesh.retry_greedy_orientations
    assert not mesh.continue_greedy_orientation_conflicts
    assert not poly.use_linear_corner_locality_guard
    assert not poly.require_both_linear_corner_branches
    assert not poly.use_corner_branch_propagation

    mesh.configure_corner_behavior("current")
    assert mesh.use_three_neighbor_orientation_hint
    assert not mesh.use_late_three_neighbor_orientation_hint
    assert not mesh.retry_greedy_orientations
    assert not mesh.continue_greedy_orientation_conflicts
    assert poly.use_linear_corner_locality_guard
    assert poly.require_both_linear_corner_branches
    assert poly.use_corner_branch_propagation

    mesh.configure_corner_behavior("pre_f8_corner_late_hint")
    assert not mesh.use_three_neighbor_orientation_hint
    assert mesh.use_late_three_neighbor_orientation_hint
    assert not poly.use_linear_corner_locality_guard
    assert not poly.require_both_linear_corner_branches
    assert not poly.use_corner_branch_propagation

    mesh.configure_corner_behavior("pre_f8_corner_greedy_retry")
    assert not mesh.use_three_neighbor_orientation_hint
    assert not mesh.use_late_three_neighbor_orientation_hint
    assert mesh.retry_greedy_orientations
    assert not mesh.continue_greedy_orientation_conflicts
    assert not poly.use_linear_corner_locality_guard
    assert not poly.require_both_linear_corner_branches
    assert not poly.use_corner_branch_propagation

    mesh.configure_corner_behavior("pre_f8_corner_greedy_continue")
    assert not mesh.use_three_neighbor_orientation_hint
    assert not mesh.use_late_three_neighbor_orientation_hint
    assert not mesh.retry_greedy_orientations
    assert mesh.continue_greedy_orientation_conflicts
    assert not poly.use_linear_corner_locality_guard
    assert not poly.require_both_linear_corner_branches
    assert not poly.use_corner_branch_propagation

    mesh.configure_corner_behavior("pre_f8_corner_late_hint_retry")
    assert not mesh.use_three_neighbor_orientation_hint
    assert mesh.use_late_three_neighbor_orientation_hint
    assert mesh.retry_greedy_orientations
    assert not mesh.continue_greedy_orientation_conflicts


def test_find_orientations_handles_extreme_three_neighbor_shoulder_cell():
    config = read_yaml("config/static/zalesak.yaml")
    mesh_cfg = apply_mesh_overrides(
        config["MESH"],
        resolution=1.5,
        mesh_type="perturbed_quads",
        perturb_wiggle=0.3,
        perturb_seed=0,
        perturb_fix_boundary=True,
    )
    opoints = make_points_from_config(mesh_cfg)
    threshold = config["GEOMS"]["THRESHOLD"]

    rng = np.random.default_rng(RANDOM_SEED)
    for _ in range(6):
        center = [rng.uniform(50, 51), rng.uniform(50, 51)]
        theta = rng.uniform(0, math.pi / 2)

    m = MergeMesh(opoints, threshold)
    m.configure_corner_behavior("current")
    fractions = initialize_zalesak(
        m, center, 15.0, 5.0, y_top_rel=10.0, theta=theta
    )
    m.setFractions(fractions)
    m.merge1Neighbors()
    m.findOrientations()

    target_id = m._get_merge_id(96, 78)
    assert target_id is not None

    obj_to_id = {id(obj): merge_id for merge_id, obj in m.merged_polys.items()}
    target_poly = m.merged_polys[target_id]
    left_id = obj_to_id[id(target_poly.getLeftNeighbor())]
    right_id = obj_to_id[id(target_poly.getRightNeighbor())]

    assert left_id == m._get_merge_id(97, 78)
    assert right_id == m._get_merge_id(96, 79)


if __name__ == "__main__":
    test_find_orientations_handles_extreme_three_neighbor_shoulder_cell()
    print("Find orientation tests completed.")
