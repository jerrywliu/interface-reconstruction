import argparse
import pickle
import numpy as np

from main.structs.meshes.merge_mesh import MergeMesh
from util.config import read_yaml, override_yaml
from util.initialize_points import makeFineCartesianGrid
from util.initialize_areas import initializeAreas
from util.initialize_velocity import initializeVelocity
# from util.write_facets import writeFacets
from util.plot_plt import plotAreas, plotPartialAreas, plotInitialAreaCompare
from util.plot_vtk import writeMesh, writePartialCells, writeFacets
from util.util import writeToPickle
from util.metrics import trueFinalAreas, L2ErrorFractions, LinfErrorFractions

def main(
    config_setting,
    setting=None,
    resolution=None,
    facet_algo=None,
    save_name=None,
    **kwargs
):

    # Read config
    config = read_yaml(f"config/{config_setting}.yaml")

    # Make dict of kwargs if not none
    # kwargs = {k: v for k, v in kwargs.items() if v is not None}
    # Override config
    # config = override_yaml(f"config/{config_setting}.yaml", kwargs)

    # Test settings
    from_temp = config["TEST"]["FROM_TEMP"]
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

    #-----

    # Initialize mesh
    if not(from_temp):
        print("Generating mesh...")
        opoints = makeFineCartesianGrid(grid_size, resolution)
        m = MergeMesh(opoints, threshold)
        writeMesh(m, f"plots/{save_name}/vtk/mesh.vtk")

        # Initialize fractions
        fractions = initializeAreas(m, test_setting=test_setting)
        m.initializeFractions(fractions)

        if do_advect:
            # Calculate final areas
            true_final_areas = trueFinalAreas(m, test_setting=test_setting, t=t_total)
    
        # Plot areas
        plotAreas(m, f"plots/{save_name}/plt/areas/initial.png")
        plotPartialAreas(m, f"plots/{save_name}/plt/partial_areas/initial.png")

        # Merge and run interface reconstruction
        print("Initial interface reconstruction")

        if facet_algo in ["Youngs", "LVIRA", "safe_linear", "safe_circle", "safe_linear_corner"]:
            m.createMergedPolys()
            plotAreas(m, f"plots/{save_name}/plt/areas/0.png")
            plotPartialAreas(m, f"plots/{save_name}/plt/partial_areas/0.png")
            writePartialCells(m, f"plots/{save_name}/vtk/reconstructed/mixed_cells/0.vtp")
            if facet_algo == "Youngs":
                m.runYoungs()
            elif facet_algo == "LVIRA":
                m.runLVIRA()
            elif facet_algo == "safe_linear":
                m.runSafeLinear()
            elif facet_algo == "safe_circle":
                m.runSafeCircle()
            elif facet_algo == "safe_linear_corner":
                _ = m.findSafeOrientations() # basic orientation finding
                m.runSafeLinearCorner()
            reconstructed_facets = [p.getFacet() for p in m.merged_polys.values()]
        else:
            m.merge1Neighbors()
            merge_ids = m.findOrientations()
            # Update plot variables with merged cells
            m.updatePlots()
            plotAreas(m, f"plots/{save_name}/plt/areas/0.png")
            plotPartialAreas(m, f"plots/{save_name}/plt/partial_areas/0.png")
            writePartialCells(m, f"plots/{save_name}/vtk/reconstructed/mixed_cells/0.vtp")
            merged_polys = m.fitFacets(merge_ids, setting=facet_algo)
            reconstructed_facets = [p.getFacet() for p in merged_polys]

            # Run C0
            if do_c0:
                merged_polys = m.makeC0(merged_polys)
                C0_facets = [p.getFacet() for p in merged_polys]
                writeFacets(C0_facets, f"plots/{save_name}/vtk/reconstructed/C0_facets/0.vtp")
        
        writeFacets(reconstructed_facets, f"plots/{save_name}/vtk/reconstructed/facets/0.vtp")

        # Stop after initial reconstruction
        # print(1/0)

    else:
        print("Loading mesh...")
        temp_f = open(f"plots/{save_name}/temp.pickle", "rb")
        [m, t_init] = pickle.load(temp_f)
        print(f"Skipping to timestep {t_init}...")

    if do_advect:

        # Initialize velocities
        velocity = initializeVelocity(m, t_total, test_setting=test_setting)

        # Advection
        print("Advection test")
        t = 0
        num_iters = int(t_total/dt)
        for iter in range(1, num_iters+1):

            # If loading from temp, skip
            if from_temp and iter < t_init:
                t += dt
                continue

            print("t = {}".format(t))

            # Advect facets and compute new areas
            advected_facets = m.advectMergedFacets(velocity, t, dt, checkSize=2)
            writeFacets(advected_facets, f"plots/{save_name}/vtk/advected/facets/{iter}.vtp")

            # Plot areas
            plotAreas(m, f"plots/{save_name}/plt/areas/{iter}.png")
            plotPartialAreas(m, f"plots/{save_name}/plt/partial_areas/{iter}.png")

            # Merge and run interface reconstruction
            if facet_algo in ["Youngs", "LVIRA", "safe_linear", "safe_circle", "safe_linear_corner"]:
                m.createMergedPolys()
                writePartialCells(m, f"plots/{save_name}/vtk/reconstructed/mixed_cells/{iter}.vtp")
                if facet_algo == "Youngs":
                    m.runYoungs()
                elif facet_algo == "LVIRA":
                    m.runLVIRA()
                elif facet_algo == "safe_linear":
                    m.runSafeLinear()
                elif facet_algo == "safe_circle":
                    m.runSafeCircle()
                elif facet_algo == "safe_linear_corner":
                    m.runSafeLinearCorner()
                reconstructed_facets = [p.getFacet() for p in m.merged_polys.values()]
            else:
                m.merge1Neighbors()
                merge_ids = m.findOrientations()
                # Update plot variables with merged cells
                m.updatePlots()
                writePartialCells(m, f"plots/{save_name}/vtk/reconstructed/mixed_cells/{iter}.vtp")
                merged_polys = m.fitFacets(merge_ids, setting=facet_algo)
                reconstructed_facets = [p.getFacet() for p in merged_polys]

                # Run C0
                if do_c0:
                    merged_polys = m.makeC0(merged_polys)
                    C0_facets = [p.getFacet() for p in merged_polys]
                    writeFacets(C0_facets, f"plots/{save_name}/vtk/reconstructed/C0_facets/{iter}.vtp")
                
            writeFacets(reconstructed_facets, f"plots/{save_name}/vtk/reconstructed/facets/{iter}.vtp")
            
            if from_temp:
                print(1/0)
                
            # writeToPickle(m, f"plots/{save_name}/temp.pickle", iter)

            t += dt
            # if iter == 1:
            #     print(1/0)

        # Compute final metrics
        # Volume errors
        volume_l2_error, l2_mixed_count = L2ErrorFractions(m.getFractions(), true_final_areas)
        with open(f"plots/{save_name}/volume_l2_error.txt", "w") as f:
            f.write(f"{volume_l2_error}\n{l2_mixed_count}\n")
        volume_linf_error, _ = LinfErrorFractions(m.getFractions(), true_final_areas)
        with open(f"plots/{save_name}/volume_linf_error.txt", "w") as f:
            f.write(f"{volume_linf_error}\n{l2_mixed_count}\n")
        plotInitialAreaCompare(m, f"plots/{save_name}/plt/{iter}_initial_compare.png")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Advection tests.")
    parser.add_argument("--config", type=str, help="config")
    # Optional
    parser.add_argument("--setting", type=str, help="setting", required=False)
    parser.add_argument("--resolution", type=float, help="resolution", required=False)
    parser.add_argument("--facet_algo", type=str, help="facet_algo", required=False)
    parser.add_argument("--save_name", type=str, help="save_name", required=False)

    # Parse
    args = parser.parse_args()

    main(
        config_setting=args.config,
        setting=args.setting,
        resolution=args.resolution,
        facet_algo=args.facet_algo,
        save_name=args.save_name
    )
