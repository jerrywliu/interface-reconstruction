import pytest

from experiments.baselines.external_runner import (
    UnresolvedExternalCell,
    run_external_static_baseline,
)
from main.algos.baselines.external_geometry import (
    ExternalCellReconstruction,
    ExternalInterfaceComponent,
    ExternalLinePrimitive,
    ExternalReconstructionStatus,
)
from main.structs.meshes.base_mesh import BaseMesh


def _points(nx, ny):
    return [[[float(x), float(y)] for y in range(ny + 1)] for x in range(nx + 1)]


def test_runner_records_each_original_mixed_cell_and_boundary_outcome():
    fractions = [[0.5, 0.5, 0.5] for _ in range(3)]
    mesh = BaseMesh(_points(3, 3), 1.0e-10, fractions=fractions)

    def method(context):
        if not context.complete_3x3:
            raise UnresolvedExternalCell(
                "paper stencil is incomplete", {"boundary": True}
            )
        primitive = ExternalLinePrimitive(
            (context.polygon_points[0][0], context.polygon_points[0][1] + 0.5),
            (context.polygon_points[1][0], context.polygon_points[1][1] + 0.5),
        )
        return ExternalCellReconstruction(
            context.cell_index,
            context.polygon_points,
            (ExternalInterfaceComponent((primitive,)),),
            "example",
            "published",
            ExternalReconstructionStatus.RECONSTRUCTED,
            target_phase_area=context.target_phase_area,
            exact_area_callback=lambda polygon: context.target_phase_area,
        )

    result = run_external_static_baseline(
        mesh, method, source_method="example", source_variant="published"
    )

    assert len(result.cells) == 9
    assert result.metadata["status_counts"]["reconstructed"] == 1
    assert result.metadata["status_counts"]["unresolved"] == 8
    assert result.cells[(0, 0)].diagnostics["boundary"] is True


def test_runner_rejects_perturbed_mesh_before_method_execution():
    points = _points(2, 2)
    points[1][1] = [1.1, 1.0]
    mesh = BaseMesh(points, 1.0e-10, fractions=[[0.5, 0.5], [0.5, 0.5]])
    called = False

    def method(context):
        nonlocal called
        called = True

    with pytest.raises(ValueError, match="Cartesian"):
        run_external_static_baseline(
            mesh, method, source_method="example", source_variant="published"
        )
    assert called is False
