import json
import os
import vtk

from main.structs.meshes.base_mesh import BaseMesh
from main.structs.meshes.merge_mesh import MergeMesh
from main.structs.interface_geometry import (
    ArcPrimitive,
    composite_from_facet,
    iter_primitives_from_facets,
    primitive_type_code,
)

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


def _corner_tip_metadata_path(path: str) -> str:
    root, _ext = os.path.splitext(path)
    return f"{root}.corner_tips.json"


def _facet_metadata_path(path: str) -> str:
    root, _ext = os.path.splitext(path)
    return f"{root}.facet_metadata.json"


def _point_metadata(point):
    return [float(point[0]), float(point[1])]


def _serialize_primitive(primitive, *, facet_index, primitive_index, global_index):
    record = {
        "index": int(global_index),
        "facet_index": int(facet_index),
        "primitive_index": int(primitive_index),
        "kind": "arc" if isinstance(primitive, ArcPrimitive) else "line",
        "source_name": primitive.source_name,
        "p_left": _point_metadata(primitive.pLeft),
        "p_right": _point_metadata(primitive.pRight),
    }
    if isinstance(primitive, ArcPrimitive):
        signed_delta = float(primitive._signed_delta())
        record.update(
            {
                "center": _point_metadata(primitive.center),
                "radius": float(primitive.radius),
                "signed_delta": signed_delta,
                "orientation": "ccw" if signed_delta > 0.0 else "cw",
                "is_major_arc": bool(primitive.is_major_arc),
            }
        )
    return record


def _write_facet_metadata(facets, path: str):
    """Write exact primitive and corner records alongside the sampled VTP.

    The VTP remains a convenient rendering artifact, while this sidecar is the
    authoritative geometry record used by later figure generation.  In
    particular, arcs retain their center, signed radius, and orientation rather
    than only the eight-point polyline emitted by vtkArcSource.
    """
    primitives = []
    corners = []
    global_index = 0
    for facet_index, facet in enumerate(facets):
        if facet is None:
            continue
        composite = composite_from_facet(facet)
        local_records = []
        for primitive_index, primitive in enumerate(composite.primitives):
            record = _serialize_primitive(
                primitive,
                facet_index=facet_index,
                primitive_index=primitive_index,
                global_index=global_index,
            )
            primitives.append(record)
            local_records.append(record)
            global_index += 1

        for joint_index, joint in enumerate(composite.joints):
            if joint.kind != "corner":
                continue
            side_records = local_records[:2]
            corners.append(
                {
                    "facet_index": int(facet_index),
                    "joint_index": int(joint_index),
                    "source_name": composite.source_name,
                    "apex": _point_metadata(joint.point),
                    "p_left": _point_metadata(side_records[0]["p_left"]),
                    "p_right": _point_metadata(side_records[-1]["p_right"]),
                    "primitive_indices": [int(record["index"]) for record in side_records],
                    "left_primitive": side_records[0],
                    "right_primitive": side_records[-1],
                }
            )

    metadata_path = _facet_metadata_path(path)
    with open(metadata_path, "w") as metadata_file:
        json.dump(
            {
                "schema_version": 2,
                "source": "util.plotting.vtk_utils.writeFacets",
                "primitives": primitives,
                "corners": corners,
            },
            metadata_file,
            indent=2,
        )


def _write_corner_tip_metadata(facets, path: str):
    corner_tips = []
    for facet_index, facet in enumerate(facets):
        if facet is None:
            continue
        composite = composite_from_facet(facet)
        for joint_index, joint in enumerate(composite.joints):
            if joint.kind != "corner":
                continue
            corner_tips.append(
                {
                    "point": [float(joint.point[0]), float(joint.point[1])],
                    "kind": joint.kind,
                    "source_name": composite.source_name,
                    "facet_index": facet_index,
                    "joint_index": joint_index,
                }
            )

    metadata_path = _corner_tip_metadata_path(path)
    if not corner_tips:
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        return

    with open(metadata_path, "w") as metadata_file:
        json.dump(
            {
                "schema_version": 1,
                "source": "util.plotting.vtk_utils.writeFacets",
                "corner_tips": corner_tips,
            },
            metadata_file,
            indent=2,
        )


# Plot facets as vtk file
def writeFacets(facets, path):
    base_path = '/'.join(path.split('/')[:-1])
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    facets = list(facets)
    _write_facet_metadata(facets, path)
    _write_corner_tip_metadata(facets, path)

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
