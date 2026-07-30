import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


RUN_CONTEXT_FIELDS = [
    "experiment",
    "algo",
    "resolution",
    "wiggle",
    "seed",
    "save_name",
    "source_commit",
    "source_branch",
    "plic_fallback",
    "rescue_profile",
    "corner_behavior_profile",
]

CSV_ARTIFACTS = {
    "case_metrics.csv": "metrics/case_metrics.csv",
    "cell_metrics.csv": "metrics/cell_metrics.csv",
    "merge_events.csv": "metrics/merge_events.csv",
    "unresolved_plic_fallbacks.csv": "metrics/unresolved_plic_fallbacks.csv",
}

OUTPUT_FILENAMES = [
    "run_inventory.csv",
    "run_manifests.jsonl",
    "case_geometry.jsonl",
    *CSV_ARTIFACTS,
]


class DiagnosticBundleError(RuntimeError):
    pass


def _make_tree_read_only(root):
    root = Path(root)
    for path in sorted(
        root.rglob("*"), key=lambda item: len(item.parts), reverse=True
    ):
        if path.is_symlink():
            continue
        path.chmod(path.stat().st_mode & ~0o222)
    root.chmod(root.stat().st_mode & ~0o222)


def _make_tree_writable(root):
    root = Path(root)
    if not root.exists():
        return
    root.chmod(root.stat().st_mode | 0o700)
    for path in root.rglob("*"):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o700)


def archive_run_bundle(run_dir, raw_bundle_root, save_name=None):
    """Copy one run bundle into a collision-proof, read-only release location."""
    run_dir = Path(run_dir)
    raw_bundle_root = Path(raw_bundle_root)
    relative_bundle = Path(save_name or run_dir.name)
    if relative_bundle.is_absolute() or ".." in relative_bundle.parts:
        raise DiagnosticBundleError(
            f"Invalid release bundle name: {relative_bundle}"
        )
    if not run_dir.is_dir():
        raise DiagnosticBundleError(f"Run bundle does not exist: {run_dir}")

    destination = raw_bundle_root / relative_bundle
    staging = destination.with_name(f".{destination.name}.copying-{os.getpid()}")
    if destination.exists():
        raise DiagnosticBundleError(
            f"Release run bundle already exists: {destination}"
        )
    if staging.exists():
        raise DiagnosticBundleError(
            f"Incomplete release staging path already exists: {staging}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(run_dir, staging, copy_function=shutil.copy2)
        _make_tree_read_only(staging)
        if destination.exists():
            raise DiagnosticBundleError(
                f"Release run bundle already exists: {destination}"
            )
        staging.replace(destination)
    except Exception:
        if staging.exists():
            _make_tree_writable(staging)
            shutil.rmtree(staging)
        raise
    return destination


def prepare_diagnostic_bundle(output_dir):
    """Initialize a clean set of consolidated diagnostic files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in OUTPUT_FILENAMES:
        path = output_dir / filename
        if path.exists():
            path.unlink()
    return output_dir


def _append_csv(path, fieldnames, rows):
    path = Path(path)
    needs_header = not path.exists() or path.stat().st_size == 0
    if not needs_header:
        with path.open("r", newline="", encoding="utf-8") as stream:
            existing_header = next(csv.reader(stream), [])
        if existing_header != list(fieldnames):
            raise DiagnosticBundleError(
                f"CSV schema mismatch for {path}: {existing_header} != {fieldnames}"
            )

    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise DiagnosticBundleError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _read_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DiagnosticBundleError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
    return records


def _append_jsonl(path, records):
    with Path(path).open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")


def _run_context(run_metadata, manifest):
    parameters = manifest.get("parameters", {})
    context = {
        "experiment": run_metadata["experiment"],
        "algo": run_metadata["algo"],
        "resolution": run_metadata["resolution"],
        "wiggle": run_metadata["wiggle"],
        "seed": run_metadata["seed"],
        "save_name": run_metadata["save_name"],
        "source_commit": manifest.get("source_commit", ""),
        "source_branch": manifest.get("source_branch", ""),
        "plic_fallback": parameters.get("plic_fallback", ""),
        "rescue_profile": parameters.get("rescue_profile", ""),
        "corner_behavior_profile": parameters.get(
            "corner_behavior_profile",
            run_metadata.get("corner_behavior_profile", ""),
        ),
    }
    return {field: context.get(field, "") for field in RUN_CONTEXT_FIELDS}


def _with_context(context, rows):
    return [{**context, **row} for row in rows]


def _numeric_value(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _cell_row_priority(row):
    """Prefer the active fallback/reconstruction row over a historical placeholder."""
    return (
        row.get("construction_path") == "plic_fallback",
        row.get("final_facet_class") not in (None, "", "missing"),
        _numeric_value(row.get("event_count")),
        _numeric_value(row.get("merge_id"), default=-1),
    )


def _deduplicate_cell_rows(rows):
    selected = {}
    for row in rows:
        key = (row.get("case_index", ""), row.get("cell_id", ""))
        current = selected.get(key)
        if current is None or _cell_row_priority(row) > _cell_row_priority(current):
            selected[key] = row
    return list(selected.values())


def _cell_summary(rows):
    class_counts = Counter(row.get("final_facet_class", "") for row in rows)
    merge_ids = {row.get("merge_id", "") for row in rows}
    merged_ids = {
        row.get("merge_id", "")
        for row in rows
        if _numeric_value(row.get("is_merged"))
    }
    num_cells = len(rows)

    def count(field):
        return sum(_numeric_value(row.get(field)) for row in rows)

    def fraction(value):
        return value / num_cells if num_cells else 0.0

    num_merged_cells = count("is_merged")
    plic_count = sum(
        row.get("construction_path") == "plic_fallback" for row in rows
    )
    used_circular = count("used_circular")
    used_linear_corner = count("used_linear_corner")
    used_curved_corner = count("used_curved_corner")
    used_curved_corner_rescue = count("used_curved_corner_rescue")
    return {
        "num_mixed_cells": num_cells,
        "num_merge_components": len(merge_ids),
        "num_merged_cells": num_merged_cells,
        "num_merged_components": len(merged_ids),
        "num_plic_fallback_cells": plic_count,
        "num_used_circular_cells": used_circular,
        "num_used_linear_corner_cells": used_linear_corner,
        "num_used_curved_corner_cells": used_curved_corner,
        "num_used_curved_corner_rescue_cells": used_curved_corner_rescue,
        "num_final_linear_cells": class_counts["linear"],
        "num_final_circular_cells": class_counts["circular"],
        "num_final_linear_corner_cells": class_counts["linear_corner"],
        "num_final_curved_corner_cells": class_counts["curved_corner"],
        "num_final_missing_cells": class_counts["missing"],
        "fraction_merged_cells": fraction(num_merged_cells),
        "fraction_plic_fallback_cells": fraction(plic_count),
        "fraction_used_circular_cells": fraction(used_circular),
        "fraction_used_linear_corner_cells": fraction(used_linear_corner),
        "fraction_used_curved_corner_cells": fraction(used_curved_corner),
        "fraction_used_curved_corner_rescue_cells": fraction(
            used_curved_corner_rescue
        ),
        "fraction_final_linear_cells": fraction(class_counts["linear"]),
        "fraction_final_circular_cells": fraction(class_counts["circular"]),
        "fraction_final_linear_corner_cells": fraction(
            class_counts["linear_corner"]
        ),
        "fraction_final_curved_corner_cells": fraction(
            class_counts["curved_corner"]
        ),
    }


def _repair_case_summaries(case_rows, cell_rows):
    rows_by_case = defaultdict(list)
    for row in cell_rows:
        rows_by_case[row.get("case_index", "")].append(row)
    for row in case_rows:
        summary = _cell_summary(rows_by_case.get(row.get("case_index", ""), []))
        for field, value in summary.items():
            if field in row:
                row[field] = value
    return case_rows


def consolidate_run_diagnostics(
    run_dir, output_dir, run_metadata, inventory_root=None
):
    """Append one static run bundle to the sweep-level diagnostic tables."""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    manifest_path = run_dir / "run_manifest.json"
    geometry_path = run_dir / "metrics" / "case_geometry.jsonl"
    required_paths = [
        manifest_path,
        geometry_path,
        *(run_dir / relative_path for relative_path in CSV_ARTIFACTS.values()),
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise DiagnosticBundleError(
            "Run bundle is missing required diagnostics: " + ", ".join(missing)
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DiagnosticBundleError(f"Invalid run manifest: {manifest_path}") from exc

    context = _run_context(run_metadata, manifest)
    _append_jsonl(
        output_dir / "run_manifests.jsonl",
        [{**context, "manifest": manifest}],
    )

    geometry_records = _read_jsonl(geometry_path)
    _append_jsonl(
        output_dir / "case_geometry.jsonl",
        _with_context(context, geometry_records),
    )

    diagnostic_tables = {}
    for output_filename, relative_path in CSV_ARTIFACTS.items():
        source_path = run_dir / relative_path
        diagnostic_tables[output_filename] = _read_csv(source_path)

    cell_fields, cell_rows = diagnostic_tables["cell_metrics.csv"]
    cell_rows = _deduplicate_cell_rows(cell_rows)
    diagnostic_tables["cell_metrics.csv"] = (cell_fields, cell_rows)
    case_fields, case_rows = diagnostic_tables["case_metrics.csv"]
    diagnostic_tables["case_metrics.csv"] = (
        case_fields,
        _repair_case_summaries(case_rows, cell_rows),
    )

    row_counts = {"case_geometry_rows": len(geometry_records)}
    for output_filename, (source_fields, rows) in diagnostic_tables.items():
        output_fields = RUN_CONTEXT_FIELDS + [
            field for field in source_fields if field not in RUN_CONTEXT_FIELDS
        ]
        _append_csv(
            output_dir / output_filename,
            output_fields,
            _with_context(context, rows),
        )
        count_key = output_filename.removesuffix(".csv") + "_rows"
        row_counts[count_key] = len(rows)

    inventory_fields = RUN_CONTEXT_FIELDS + [
        "run_bundle",
        "case_geometry_rows",
        "case_metrics_rows",
        "cell_metrics_rows",
        "merge_events_rows",
        "unresolved_plic_fallbacks_rows",
    ]
    if inventory_root is None:
        inventory_bundle = str(run_dir.resolve())
    else:
        try:
            inventory_bundle = run_dir.resolve().relative_to(
                Path(inventory_root).resolve()
            ).as_posix()
        except ValueError as exc:
            raise DiagnosticBundleError(
                f"Run bundle {run_dir} is outside release root {inventory_root}"
            ) from exc

    _append_csv(
        output_dir / "run_inventory.csv",
        inventory_fields,
        [
            {
                **context,
                "run_bundle": inventory_bundle,
                **row_counts,
            }
        ],
    )
    return row_counts
