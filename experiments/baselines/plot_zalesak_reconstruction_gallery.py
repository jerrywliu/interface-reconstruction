"""Render the appendix's 3-by-4 Zalesak gallery from deposited geometry."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle

from experiments.baselines.project_benchmarks import canonical_benchmark_cases
from experiments.plotting import (
    PAPER_HIGH_ORDER_COLORS, PAPER_METHOD_COLORS, PAPER_METHOD_LINESTYLES,
    apply_paper_serif_style,
)
from main.algos.baselines.external_geometry import ExternalBaselineResult
from main.algos.baselines.project_facet_adapter import external_primitives_from_facet_metadata

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / 'experiments/baselines/results/zalesak_25case_20261004'
METHODS = ('plvira', 'pcic', 'quasi', 'ours_graph', 'ours_corner', 'ours_corner_c0')
LABELS = ('PLVIRA', 'PCIC', 'QUASI', 'Ours: circular\n(graph-coordinated)',
          'Ours: circular\n(graph-coordinated\n+ corner)',
          'Ours: circular\n(graph-coordinated\n+ corner + joint $C^0$)')


def geometry_path(value: str) -> Path:
    """Resolve the original recorded workspace path beneath the current clone."""
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    if path.is_file():
        return path
    marker = '/interface-reconstruction/'
    if marker in value:
        path = ROOT / value.split(marker, 1)[1]
    if not path.is_file():
        raise FileNotFoundError(f'Deposited geometry is missing: {value}')
    return path


def draw(axis, primitives, color, style, width):
    for primitive in primitives:
        points = np.asarray(primitive.sample(min(500, max(24, int(primitive.length() * 40)))))
        axis.plot(points[:, 0], points[:, 1], color=color, ls=style, lw=width)


def number(value):
    if float(value) == 0:
        return '0'
    mantissa, exponent = f'{float(value):.2e}'.split('e')
    return mantissa + r'\times10^{' + str(int(exponent)) + '}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--case-index', type=int, default=12)
    args = parser.parse_args()
    csv_path = args.input_root / 'case_metrics.csv'
    manifest = json.loads((args.input_root / 'manifest.json').read_text())
    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != manifest['case_csv_sha256']:
        raise ValueError('The frozen Zalesak case table has changed')
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    index = {(r['method'], int(r['N']), int(r['case_index'])): r for r in rows}
    if len(rows) != 1400 or len(index) != len(rows):
        raise ValueError('Expected 1,400 unique matched case rows')
    truth = canonical_benchmark_cases('zalesak', (args.case_index,))[0].truth_primitives()
    join = np.asarray(next(p for p in truth if p.kind == 'arc').p_left)
    colors = [PAPER_HIGH_ORDER_COLORS[k] for k in ('plvira', 'pcic_center', 'quasi')]
    colors += [PAPER_METHOD_COLORS[k] for k in ('circular', 'circular+corner', 'circular+corner+C0')]
    styles = ['-.', '--', '-'] + [PAPER_METHOD_LINESTYLES[k] for k in ('circular', 'circular+corner', 'circular+corner+C0')]
    apply_paper_serif_style()
    fig = plt.figure(figsize=(7.25, 10.0), layout='constrained')
    grid = fig.add_gridspec(3, 4)
    fig.set_constrained_layout_pads(w_pad=.06, h_pad=.08, wspace=.04, hspace=.04)
    for i, (method, label, color, style) in enumerate(zip(METHODS, LABELS, colors, styles)):
        pair, row = divmod(i, 3)
        for j, n in enumerate((64, 300)):
            col = 2 * pair + j
            cell = grid[row, col].subgridspec(2, 1, height_ratios=[1, .65], hspace=.08)
            axis, zoom = fig.add_subplot(cell[0]), fig.add_subplot(cell[1])
            record = index[(method, n, args.case_index)]
            path = geometry_path(record['geometry_file'])
            if method in METHODS[:3]:
                result = ExternalBaselineResult.from_json(path)
                primitives = [p for c in result.cells.values() for p in c.primitives()]
            else:
                primitives = external_primitives_from_facet_metadata(json.loads(path.read_text()))
            draw(axis, truth, '#777777', '-', 1.15)
            draw(axis, primitives, color, style, .8)
            axis.set(xlim=(33, 68), ylim=(33, 68), aspect='equal', xticks=[35, 50, 65], yticks=[35, 50, 65])
            title = f'({chr(97 + 4 * row + col)}) {label}\n$N={n}$\n'
            title += '$d_H=' + number(record['hausdorff']) + '$\n$g_h=' + number(record['facet_gap']) + '$'
            axis.set_title('\n' * (6 - len(title.split('\n'))) + title, fontsize=8.8, pad=5)
            axis.tick_params(labelsize=8.2, labelbottom=False, labelleft=(col == 0))
            if col == 0:
                axis.set_ylabel('$y$', fontsize=9, labelpad=3)
            half = 100 / n
            draw(zoom, truth, '#777777', '-', 1.3)
            draw(zoom, primitives, color, '-', 1.05)
            zoom.set(xlim=(join[0] - half, join[0] + half), ylim=(join[1] - half, join[1] + half), aspect='equal', xticks=[], yticks=[])
            for coordinate in (0, 1):
                for k in range(int(np.floor((join[coordinate] - half) * n / 100)), int(np.ceil((join[coordinate] + half) * n / 100)) + 1):
                    (zoom.axvline if coordinate == 0 else zoom.axhline)(k * 100 / n, color='#dddddd', lw=.35, zorder=0)
            axis.add_patch(Rectangle(join - half, 2 * half, 2 * half, fill=False, edgecolor='#7C5AA6', linestyle='--', lw=.75))
            for spine in zoom.spines.values():
                spine.set_edgecolor('#7C5AA6'); spine.set_linewidth(.75)
            fig.add_artist(ConnectionPatch(xyA=(join[0], join[1] - half), coordsA=axis.transData, xyB=(.5, 1), coordsB=zoom.transAxes, color='#7C5AA6', lw=.55, linestyle=':', clip_on=False))
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / 'zalesak_all_methods_gallery_3x4.pdf'
    fig.savefig(path, bbox_inches='tight', pad_inches=.08)
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(path)


if __name__ == '__main__':
    main()
