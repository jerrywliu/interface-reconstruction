import os
from tqdm import tqdm

from util.plotting.plt_utils import plotAreas, plotPartialAreas
from util.plotting.vtk_utils import writeFacets
from util.reconstruction import runReconstruction


def runAdvection(
    m,
    velocity,
    dt,
    t_total,
    facet_algo,
    do_c0,
    output_dirs,
    checkpointer,
    checkpoint_freq=50,
    resume_iter=None,
):
    """
    Run advection simulation with interface reconstruction.

    Args:
        m: MergeMesh object
        velocity: Velocity field function
        dt: Time step size
        t_total: Total simulation time
        facet_algo: Interface reconstruction algorithm
        do_c0: Boolean for C0 continuity enforcement
        output_dirs: Dictionary of output directories
        checkpointer: Checkpoint object for saving/loading states
        checkpoint_freq: How often to save checkpoints (iterations)
        resume_iter: Optional specific iteration to resume from

    Returns:
        m: Updated MergeMesh object
    """

    # Handle checkpoint loading
    if resume_iter is not None:
        checkpoint_data = checkpointer.load_iteration(resume_iter)
        if checkpoint_data is None:
            raise ValueError(f"No checkpoint found for iteration {resume_iter}")
    else:
        checkpoint_data = checkpointer.load_latest()

    # Set starting point
    if checkpoint_data:
        m = checkpoint_data["state"]
        start_iter = checkpoint_data["iteration"] + 1
        t = start_iter * dt
        print(f"Resuming from iteration {start_iter}")
    else:
        start_iter = 1
        t = 0

    # Run simulation
    num_iters = int(t_total / dt)
    for iter in tqdm(range(start_iter, num_iters + 1), desc="Running advection"):
        print(f"t = {t:.3f}")

        # Advect facets
        advected_facets = m.advectMergedFacets(velocity, t, dt, checkSize=2)
        # Save advected facets
        writeFacets(
            advected_facets, os.path.join(output_dirs["vtk_advected"], f"{iter}.vtp")
        )

        # Run reconstruction (and save facets and partial cells)
        reconstructed_facets = runReconstruction(
            m, facet_algo, do_c0, iter, output_dirs
        )

        # Save checkpoint if needed
        if iter % checkpoint_freq == 0:
            checkpointer.save(m, iter)

        t += dt

    return m
