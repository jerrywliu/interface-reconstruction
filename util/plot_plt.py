import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as plt_polygon
from matplotlib.collections import PatchCollection

from main.structs.meshes.base_mesh import BaseMesh
from main.structs.meshes.merge_mesh import MergeMesh

# Plot mesh and areas as plt images
def plotPolyValues(m: BaseMesh, values, path):
    base_path = '/'.join(path.split('/')[:-1])
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    patchcollection = PatchCollection(m._plt_patches, cmap="plasma", edgecolor='black') #jet
    patchcollection.set_array(values)
    _, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(m._min_x, m._max_x)
    ax.set_ylim(m._min_y, m._max_y)
    ax.add_collection(patchcollection)
    plt.savefig(path, dpi=199)
    plt.close()
    plt.clf()

def plotAreas(m: BaseMesh, path):
    plotPolyValues(m, m._plt_patchareas, path)

def plotPartialAreas(m: BaseMesh, path):
    plotPolyValues(m, m._plt_patchpartialareas, path)

def plotInitialAreaCompare(m: BaseMesh, path):
    plotPolyValues(m, np.abs(m._plt_patchareas - m._plt_patchinitialareas), path)
