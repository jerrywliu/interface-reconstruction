import argparse
import os

from main.structs.meshes.merge_mesh import MergeMesh

from util.config import read_yaml
from util.io.setup import setupOutputDirs
from util.reconstruction import runReconstruction
from util.advection import runAdvection
from util.initialize.mesh_factory import make_points_from_config
from util.initialize.areas import initializeAreas, trueFinalAreas
from util.initialize.velocity import initializeVelocity
from util.plotting.plt_utils import plotAreas, plotPartialAreas
from util.plotting.vtk_utils import writeMesh
from util.metrics import computeFinalMetrics
from util.checkpointing import Checkpoint


def initialize_simulation(
    m,
    mesh_cfg,
    test_setting,
    threshold,
    output_dirs,
    t_total=None,
):
    """Initialize mesh and run initial reconstruction."""
    print("Generating mesh...")
    opoints = make_points_from_config(mesh_cfg)
    m = MergeMesh(opoints, threshold)
    writeMesh(m, os.path.join(output_dirs["vtk"], "mesh.vtk"))

    # Initialize fractions
    fractions = initializeAreas(m, test_setting=test_setting)
    m.initializeFractions(fractions)

    # Calculate final areas if doing advection
    true_final_areas = None
    if t_total is not None:
        true_final_areas = trueFinalAreas(m, test_setting=test_setting, t=t_total)

    # Plot initial areas
    plotAreas(m, os.path.join(output_dirs["plt_areas"], "initial.png"))
    plotPartialAreas(m, os.path.join(output_dirs["plt_partial"], "initial.png"))

    return m, true_final_areas


def main(
    config_setting,
    setting=None,
    resolution=None,
    facet_algo=None,
    save_name=None,
    resume_iter=None,
    **kwargs,
):

    # Read config
    config = read_yaml(f"config/{config_setting}.yaml")

    # Test settings
    save_name = save_name if save_name is not None else config["TEST"]["SAVE_NAME"]
    test_setting = setting if setting is not None else config["TEST"]["SETTING"]

    # Mesh settings
    grid_size = config["MESH"]["GRID_SIZE"]
    resolution = resolution if resolution is not None else config["MESH"]["RESOLUTION"]

    # Area and facet settings
    facet_algo = facet_algo if facet_algo is not None else config["GEOMS"]["FACET_ALGO"]
    threshold = config["GEOMS"]["THRESHOLD"]
    do_c0 = config["GEOMS"]["DO_C0"]

    # Advection settings
    do_advect = config["ADVECTION"]["DO_ADVECT"]
    if do_advect:
        dt = config["ADVECTION"]["DT"]
        t_total = config["ADVECTION"]["T_TOTAL"]

    # Setup
    output_dirs = setupOutputDirs(save_name)
    checkpointer = Checkpoint(output_dirs["base"], config)

    # -----

    # Initialize or load from checkpoint
    if resume_iter is None:
        mesh_cfg = dict(config["MESH"])
        mesh_cfg["RESOLUTION"] = resolution

        m, true_final_areas = initialize_simulation(
            m=None,
            mesh_cfg=mesh_cfg,
            test_setting=test_setting,
            threshold=threshold,
            output_dirs=output_dirs,
            t_total=(t_total if do_advect else None),
        )

        # Run initial reconstruction
        print("Initial interface reconstruction")
        reconstructed_facets = runReconstruction(
            m=m,
            facet_algo=facet_algo,
            do_c0=do_c0,
            iter=0,
            output_dirs=output_dirs,
            algo_kwargs={},  # TODO JL 3/13/25 add algo_kwargs for safe_linear (maybe for static tests?)
        )

    else:
        print("Loading from checkpoint...")
        checkpoint_data = checkpointer.load_iteration(resume_iter)
        if checkpoint_data is None:
            raise ValueError(f"No checkpoint found for iteration {resume_iter}")
        m = checkpoint_data["state"]
        config = checkpointer.load_config()

    # Run advection if enabled
    if do_advect:
        velocity = initializeVelocity(m, t_total, test_setting=test_setting)

        m = runAdvection(
            m=m,
            velocity=velocity,
            dt=dt,
            t_total=t_total,
            facet_algo=facet_algo,
            do_c0=do_c0,
            output_dirs=output_dirs,
            checkpointer=checkpointer,
            checkpoint_freq=300,  # TODO: make this a config setting
            resume_iter=resume_iter,
        )

        # Compute final metrics
        if true_final_areas is not None:
            computeFinalMetrics(m, true_final_areas, output_dirs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advection tests.")
    parser.add_argument("--config", type=str, help="config")
    parser.add_argument("--setting", type=str, help="setting", required=False)
    parser.add_argument("--resolution", type=float, help="resolution", required=False)
    parser.add_argument("--facet_algo", type=str, help="facet_algo", required=False)
    parser.add_argument("--save_name", type=str, help="save_name", required=False)
    parser.add_argument(
        "--resume_iter", type=int, help="iteration to resume from", required=False
    )

    args = parser.parse_args()

    main(
        config_setting=args.config,
        setting=args.setting,
        resolution=args.resolution,
        facet_algo=args.facet_algo,
        save_name=args.save_name,
        resume_iter=args.resume_iter,
    )
