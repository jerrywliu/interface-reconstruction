#Used in VisIt to plot facets
#1. Open mesh file, set plot bounds
#2. Call plot_facets method

from visit_utils import *

def plot_facets(facetformat='timestep_{}_all.visit', saveformat='facets_%04d.png', tfinal=100):
    # set basic save options
    swatts = SaveWindowAttributes()
    swatts.family = 0
    swatts.format = swatts.PNG
    swatts.width = 1024
    swatts.height = 1024
    file_idx = 0
    for ts in range(tfinal+1):
        #Open file
        ReOpenDatabase(facetformat.format(ts))
        AddPlot("Mesh", "mesh", 1, 1)
        MeshAtts = MeshAttributes()
        MeshAtts.legendFlag = 1
        MeshAtts.lineWidth = 0
        MeshAtts.meshColor = (0, 0, 0, 255)
        MeshAtts.meshColorSource = MeshAtts.MeshCustom  # Foreground, MeshCustom, MeshRandom
        MeshAtts.opaqueColorSource = MeshAtts.Background  # Background, OpaqueCustom, OpaqueRandom
        MeshAtts.opaqueMode = MeshAtts.Auto  # Auto, On, Off
        MeshAtts.pointSize = 0.05
        MeshAtts.opaqueColor = (255, 255, 255, 255)
        MeshAtts.smoothingLevel = MeshAtts.Fast  # None, Fast, High
        MeshAtts.pointSizeVarEnabled = 0
        MeshAtts.pointSizeVar = "default"
        MeshAtts.pointType = MeshAtts.Point  # Box, Axis, Icosahedron, Octahedron, Tetrahedron, SphereGeometry, Point, Sphere
        MeshAtts.showInternal = 0
        MeshAtts.pointSizePixels = 2
        MeshAtts.opacity = 1
        SetPlotOptions(MeshAtts)
        DrawPlots()
        swatts.fileName = saveformat % file_idx
        SetSaveWindowAttributes(swatts)
        SaveWindow()
        SetActivePlots(1)
        DeleteActivePlots()
        file_idx += 1



#ffmpeg -f image2 -r 30 -i 0604_facets_%04d.png -vcodec libx264 -profile:v high444 -refs 16 -crf 0 -preset ultrafast output.mp4