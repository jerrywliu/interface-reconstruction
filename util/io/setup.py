import os


def setupOutputDirs(save_name):
    """
    Create all necessary output directories if they don't exist.
    Returns a dictionary of directory paths.
    """
    base_dir = os.path.join("plots", save_name)

    # Define directory structure
    dirs = {
        "base": base_dir,
        "vtk": os.path.join(base_dir, "vtk"),
        "vtk_reconstructed": os.path.join(base_dir, "vtk", "reconstructed"),
        "vtk_reconstructed_mixed": os.path.join(
            base_dir, "vtk", "reconstructed", "mixed_cells"
        ),
        "vtk_reconstructed_c0": os.path.join(
            base_dir, "vtk", "reconstructed", "C0_facets"
        ),
        "vtk_reconstructed_facets": os.path.join(
            base_dir, "vtk", "reconstructed", "facets"
        ),
        "vtk_advected": os.path.join(base_dir, "vtk", "advected", "facets"),
        "plt": os.path.join(base_dir, "plt"),
        "plt_areas": os.path.join(base_dir, "plt", "areas"),
        "plt_partial": os.path.join(base_dir, "plt", "partial_areas"),
    }

    # Create all directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs
