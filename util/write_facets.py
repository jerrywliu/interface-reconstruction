import vtk
import os

ARC_RESOLUTION = 8

"""
#Write facet objects as vtk files
def writeFacets(facets, savename):
    try:
        os.mkdir("advection_vtk/{}".format(savename))
    except:
        pass

    g = open("advection_vtk/{}_all.visit".format(savename), "w")

    facetnum = 0
    for facet in facets:
        if facet.name == 'linear':
            line = vtk.vtkLineSource()
            line.SetPoint1(facet.pLeft[0], facet.pLeft[1], 0)
            line.SetPoint2(facet.pRight[0], facet.pRight[1], 0)
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName("advection_vtk/{}/facet_{}.vtp".format(savename, facetnum))
            writer.SetInputConnection(line.GetOutputPort())
            writer.Write()
            facetnum += 1
        elif facet.name == 'arc':
            arc = vtk.vtkArcSource()
            arc.SetPoint1(facet.pLeft[0], facet.pLeft[1], 0)
            arc.SetPoint2(facet.pRight[0], facet.pRight[1], 0)
            arc.SetCenter(facet.center[0], facet.center[1], 0)
            arc.SetResolution(8)
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName("advection_vtk/{}/facet_{}.vtp".format(savename, facetnum))
            writer.SetInputConnection(arc.GetOutputPort())
            writer.Write()
            facetnum += 1
        elif facet.name == 'corner':
            if facet.radiusLeft is not None:
                arc = vtk.vtkArcSource()
                arc.SetPoint1(facet.pLeft[0], facet.pLeft[1], 0)
                arc.SetPoint2(facet.corner[0], facet.corner[1], 0)
                arc.SetCenter(facet.centerLeft[0], facet.centerLeft[1], 0)
                arc.SetResolution(8)
                writer = vtk.vtkXMLPolyDataWriter()
                writer.SetFileName("advection_vtk/{}/facet_{}.vtp".format(savename, facetnum))
                writer.SetInputConnection(arc.GetOutputPort())
            else:
                line = vtk.vtkLineSource()
                line.SetPoint1(facet.pLeft[0], facet.pLeft[1], 0)
                line.SetPoint2(facet.corner[0], facet.corner[1], 0)
                writer = vtk.vtkXMLPolyDataWriter()
                writer.SetFileName("advection_vtk/{}/facet_{}.vtp".format(savename, facetnum))
                writer.SetInputConnection(line.GetOutputPort())
            writer.Write()
            facetnum += 1
            if facet.radiusRight is not None:
                arc = vtk.vtkArcSource()
                arc.SetPoint1(facet.corner[0], facet.corner[1], 0)
                arc.SetPoint2(facet.pRight[0], facet.pRight[1], 0)
                arc.SetCenter(facet.centerRight[0], facet.centerRight[1], 0)
                arc.SetResolution(8)
                writer = vtk.vtkXMLPolyDataWriter()
                writer.SetFileName("advection_vtk/{}/facet_{}.vtp".format(savename, facetnum))
                writer.SetInputConnection(arc.GetOutputPort())
            else:
                line = vtk.vtkLineSource()
                line.SetPoint1(facet.corner[0], facet.corner[1], 0)
                line.SetPoint2(facet.pRight[0], facet.pRight[1], 0)
                writer = vtk.vtkXMLPolyDataWriter()
                writer.SetFileName("advection_vtk/{}/facet_{}.vtp".format(savename, facetnum))
                writer.SetInputConnection(line.GetOutputPort())
            writer.Write()
            facetnum += 1

    g.write("!NBLOCKS {}\n".format(facetnum))
    for i in range(facetnum):
        g.write("{}/facet_{}.vtp\n".format(savename, i))
    g.close()
"""

def writeFacets(facets, path):
    base_path = '/'.join(path.split('/')[:-1])
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    vtkappend = vtk.vtkAppendPolyData()

    for facet in facets:
        if facet.name == 'linear':
            line = vtk.vtkLineSource()
            line.SetPoint1(facet.pLeft[0], facet.pLeft[1], 0)
            line.SetPoint2(facet.pRight[0], facet.pRight[1], 0)
            line.Update()
            vtkappend.AddInputData(line.GetOutput())
        elif facet.name == 'arc':
            arc = vtk.vtkArcSource()
            arc.SetPoint1(facet.pLeft[0], facet.pLeft[1], 0)
            arc.SetPoint2(facet.pRight[0], facet.pRight[1], 0)
            arc.SetCenter(facet.center[0], facet.center[1], 0)
            arc.SetResolution(ARC_RESOLUTION)
            arc.Update()
            vtkappend.AddInputData(arc.GetOutput())
        elif facet.name == 'corner':
            # Left
            if facet.centerLeft is None and facet.radiusLeft is None:
                line = vtk.vtkLineSource()
                line.SetPoint1(facet.pLeft[0], facet.pLeft[1], 0)
                line.SetPoint2(facet.corner[0], facet.corner[1], 0)
                line.Update()
                vtkappend.AddInputData(line.GetOutput())
            else:
                arc = vtk.vtkArcSource()
                arc.SetPoint1(facet.pLeft[0], facet.pLeft[1], 0)
                arc.SetPoint2(facet.corner[0], facet.corner[1], 0)
                arc.SetCenter(facet.centerLeft[0], facet.centerLeft[1], 0)
                arc.SetResolution(ARC_RESOLUTION)
                arc.Update()
                vtkappend.AddInputData(arc.GetOutput())
            # Right
            if facet.centerRight is None and facet.radiusRight is None:
                line = vtk.vtkLineSource()
                line.SetPoint1(facet.corner[0], facet.corner[1], 0)
                line.SetPoint2(facet.pRight[0], facet.pRight[1], 0)
                line.Update()
                vtkappend.AddInputData(line.GetOutput())
            else:
                arc = vtk.vtkArcSource()
                arc.SetPoint1(facet.corner[0], facet.corner[1], 0)
                arc.SetPoint2(facet.pRight[0], facet.pRight[1], 0)
                arc.SetCenter(facet.centerRight[0], facet.centerRight[1], 0)
                arc.SetResolution(ARC_RESOLUTION)
                arc.Update()
                vtkappend.AddInputData(arc.GetOutput())
        else:
            print(f"Unknown facet type: {facet.name}")
    
    vtkappend.Update()
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputConnection(vtkappend.GetOutputPort())
    writer.Update()
    writer.Write()



def writeFacets2():
    # Create the polydata where we will store all the geometric data
    linesPolyData = vtkPolyData()

    # Create three points
    origin = [0.0, 0.0, 0.0]
    p0 = [1.0, 0.0, 0.0]
    p1 = [0.0, 1.0, 0.0]

    # Create a vtkPoints container and store the points in it
    pts = vtkPoints()
    pts.InsertNextPoint(origin)
    pts.InsertNextPoint(p0)
    pts.InsertNextPoint(p1)

    # Add the points to the polydata container
    linesPolyData.SetPoints(pts)

    # Create the first line (between Origin and P0)
    line0 = vtkLine()
    line0.GetPointIds().SetId(0, 0)  # the second 0 is the index of the Origin in linesPolyData's points
    line0.GetPointIds().SetId(1, 1)  # the second 1 is the index of P0 in linesPolyData's points

    # Create the second line (between Origin and P1)
    line1 = vtkLine()
    line1.GetPointIds().SetId(0, 0)  # the second 0 is the index of the Origin in linesPolyData's points
    line1.GetPointIds().SetId(1, 2)  # 2 is the index of P1 in linesPolyData's points

    # Create a vtkCellArray container and store the lines in it
    lines = vtkCellArray()
    lines.InsertNextCell(line0)
    lines.InsertNextCell(line1)

    # Add the lines to the polydata container
    linesPolyData.SetLines(lines)

    namedColors = vtkNamedColors()

    # Create a vtkUnsignedCharArray container and store the colors in it
    colors = vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    try:
        colors.InsertNextTupleValue(namedColors.GetColor3ub("Tomato"))
        colors.InsertNextTupleValue(namedColors.GetColor3ub("Mint"))
    except AttributeError:
        # For compatibility with new VTK generic data arrays.
        colors.InsertNextTypedTuple(namedColors.GetColor3ub("Tomato"))
        colors.InsertNextTypedTuple(namedColors.GetColor3ub("Mint"))

    # Color the lines.
    # SetScalars() automatically associates the values in the data array passed as parameter
    # to the elements in the same indices of the cell data array on which it is called.
    # This means the first component (red) of the colors array
    # is matched with the first component of the cell array (line 0)
    # and the second component (green) of the colors array
    # is matched with the second component of the cell array (line 1)
    linesPolyData.GetCellData().SetScalars(colors)

    # Setup the visualization pipeline
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(linesPolyData)

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(4)

    renderer = vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(namedColors.GetColor3d("SlateGray"))

    window = vtkRenderWindow()
    window.SetWindowName("ColoredLines")
    window.AddRenderer(renderer)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)

    # Visualize
    window.Render()
    interactor.Start()