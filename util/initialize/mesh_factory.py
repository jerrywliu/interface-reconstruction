from util.initialize.points import makeFineCartesianGrid, makePerturbedCartesianGrid


def apply_mesh_overrides(
    mesh_cfg,
    resolution=None,
    mesh_type=None,
    perturb_wiggle=None,
    perturb_seed=None,
    perturb_fix_boundary=None,
    perturb_max_tries=None,
    perturb_type=None,
):
    mesh_cfg = dict(mesh_cfg or {})
    if resolution is not None:
        mesh_cfg["RESOLUTION"] = resolution
    if mesh_type is not None:
        mesh_cfg["TYPE"] = mesh_type

    perturb_updates = {
        "WIGGLE": perturb_wiggle,
        "SEED": perturb_seed,
        "FIX_BOUNDARY": perturb_fix_boundary,
        "MAX_TRIES": perturb_max_tries,
        "TYPE": perturb_type,
    }
    if any(value is not None for value in perturb_updates.values()):
        perturb_cfg = dict(mesh_cfg.get("PERTURB") or {})
        for key, value in perturb_updates.items():
            if value is not None:
                perturb_cfg[key] = value
        mesh_cfg["PERTURB"] = perturb_cfg

    return mesh_cfg


def make_points_from_config(mesh_cfg):
    if mesh_cfg is None:
        raise ValueError("mesh_cfg must not be None")

    grid_size = mesh_cfg.get("GRID_SIZE")
    resolution = mesh_cfg.get("RESOLUTION")
    if grid_size is None or resolution is None:
        raise ValueError("MESH.GRID_SIZE and MESH.RESOLUTION are required")

    mesh_type = mesh_cfg.get("TYPE", "cartesian")
    mesh_type = str(mesh_type).lower()

    if mesh_type in ["cartesian", "fine_cartesian", "regular"]:
        return makeFineCartesianGrid(grid_size, resolution)

    if mesh_type in ["perturbed_cartesian", "perturbed_quad", "perturbed_quads"]:
        perturb_cfg = mesh_cfg.get("PERTURB", {}) or {}
        wiggle = perturb_cfg.get("WIGGLE", 0.0)
        seed = perturb_cfg.get("SEED", 0)
        fix_boundary = perturb_cfg.get("FIX_BOUNDARY", True)
        max_tries = perturb_cfg.get("MAX_TRIES", 20)
        perturb_type = perturb_cfg.get("TYPE", "random")
        return makePerturbedCartesianGrid(
            grid_size,
            resolution,
            wiggle,
            seed=seed,
            fix_boundary=fix_boundary,
            max_tries=max_tries,
            perturb_type=perturb_type,
        )

    raise ValueError(f"Unsupported mesh type: {mesh_type}")
