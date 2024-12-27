import os
import vtk

from main.structs.meshes.base_mesh import BaseMesh
from main.structs.meshes.merge_mesh import MergeMesh

# Plot mesh as vtk file
def writeMesh(m: BaseMesh, path):
    base_path = '/'.join(path.split('/')[:-1])
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    sgrid = vtk.vtkStructuredGrid()
    sgrid.SetDimensions([len(m._points), len(m._points[0]), 1])
    vtkpoints = vtk.vtkPoints()
    vtkpoints.Allocate(len(m._points)*len(m._points[0]))
    counter = 0
    for x in range(len(m._points)):
        for y in range(len(m._points[0])):
            vtkpoints.InsertPoint(counter, [m._points[x][y][0], m._points[x][y][1], 0])
            counter += 1
    sgrid.SetPoints(vtkpoints)
    writer = vtk.vtkStructuredGridWriter()
    writer.SetFileName(path)
    writer.SetInputData(sgrid)
    writer.Write()
    
# Plot partial cells
def writePartialCells(m: BaseMesh, path):
    base_path = '/'.join(path.split('/')[:-1])
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    # Plot individual cells
    points = vtk.vtkPoints()
    mixed_polygons = vtk.vtkCellArray()
    areas = vtk.vtkDoubleArray()
    assert len(m._vtk_mixed_polys) == len(m._vtk_mixed_polyareas)
    for i, mixed_poly in enumerate(m._vtk_mixed_polys):
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(mixed_poly))
        counter = 0
        for mixed_poly_point in mixed_poly:
            point_id = points.InsertNextPoint([mixed_poly_point[0], mixed_poly_point[1], 0])
            polygon.GetPointIds().SetId(counter, point_id)
            counter += 1
        mixed_polygons.InsertNextCell(polygon)
        areas.InsertNextTypedTuple([m._vtk_mixed_polyareas[i]])

    mixedPolyData = vtk.vtkPolyData()
    mixedPolyData.SetPoints(points)
    mixedPolyData.SetPolys(mixed_polygons)
    mixedPolyData.GetCellData().SetScalars(areas)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(mixedPolyData)
    writer.Update()
    writer.Write()

ARC_RESOLUTION = 8
# Plot facets as vtk file
def writeFacets(facets, path):
    base_path = '/'.join(path.split('/')[:-1])
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    vtkappend = vtk.vtkAppendPolyData()
    facet_types = vtk.vtkIntArray()

    for facet in facets:
        if facet is None:
            continue # TODO: Fix this
        if facet.name == "linear" or facet.name == "default_linear" or facet.name == "linear_deadend" or facet.name == "Youngs" or facet.name == "LVIRA":
            line = vtk.vtkLineSource()
            line.SetPoint1(facet.pLeft[0], facet.pLeft[1], 0)
            line.SetPoint2(facet.pRight[0], facet.pRight[1], 0)
            line.Update()
            vtkappend.AddInputData(line.GetOutput())
            if facet.name == "linear":
                # Facet type: linear = 0
                facet_types.InsertNextTypedTuple([0])
            elif facet.name == "default_linear":
                # Facet type: default_linear = 4
                facet_types.InsertNextTypedTuple([4])
            elif facet.name == "linear_deadend":
                # Facet type: linear_deadend = 5
                facet_types.InsertNextTypedTuple([5])
            elif facet.name == "Youngs":
                # Facet type: Youngs = 6
                facet_types.InsertNextTypedTuple([6])
            else:
                # Facet type: LVIRA = 7
                facet_types.InsertNextTypedTuple([7])
        elif facet.name == 'arc':
            arc = vtk.vtkArcSource()
            arc.SetPoint1(facet.pLeft[0], facet.pLeft[1], 0)
            arc.SetPoint2(facet.pRight[0], facet.pRight[1], 0)
            arc.SetCenter(facet.center[0], facet.center[1], 0)
            arc.SetResolution(ARC_RESOLUTION)
            arc.Update()
            vtkappend.AddInputData(arc.GetOutput())
            # Facet type: arc = 1
            facet_types.InsertNextTypedTuple([1])
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

            # Facet type:
            if facet.centerLeft is None and facet.radiusLeft is None and facet.centerRight is None and facet.radiusRight is None:
                # Linear corner = 2
                facet_types.InsertNextTypedTuple([2])
                facet_types.InsertNextTypedTuple([2])
            else:
                # Arc corner = 3
                facet_types.InsertNextTypedTuple([3])
                facet_types.InsertNextTypedTuple([3])
        else:
            print(f"Unknown facet type: {facet.name}")
    
    vtkappend.Update()
    vtkappend.GetOutput().GetCellData().SetScalars(facet_types)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputConnection(vtkappend.GetOutputPort())
    writer.Update()
    writer.Write()
