import os
import vtk

from main.structs.meshes.base_mesh import BaseMesh
from main.structs.meshes.merge_mesh import MergeMesh
from main.structs.interface_geometry import ArcPrimitive, iter_primitives_from_facets, primitive_type_code

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

    for facet in iter_primitives_from_facets(facets):
        if facet is None:
            continue
        if isinstance(facet, ArcPrimitive):
            arc = vtk.vtkArcSource()
            arc.SetPoint1(facet.pLeft[0], facet.pLeft[1], 0)
            arc.SetPoint2(facet.pRight[0], facet.pRight[1], 0)
            arc.SetCenter(facet.center[0], facet.center[1], 0)
            arc.SetResolution(ARC_RESOLUTION)
            arc.Update()
            vtkappend.AddInputData(arc.GetOutput())
            facet_types.InsertNextTypedTuple([primitive_type_code(facet)])
        else:
            line = vtk.vtkLineSource()
            line.SetPoint1(facet.pLeft[0], facet.pLeft[1], 0)
            line.SetPoint2(facet.pRight[0], facet.pRight[1], 0)
            line.Update()
            vtkappend.AddInputData(line.GetOutput())
            facet_types.InsertNextTypedTuple([primitive_type_code(facet)])
    
    vtkappend.Update()
    vtkappend.GetOutput().GetCellData().SetScalars(facet_types)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputConnection(vtkappend.GetOutputPort())
    writer.Update()
    writer.Write()
