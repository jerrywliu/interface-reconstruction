# Proposed figure color and marker map

## Design rules

1. Color identifies the reconstruction family before it identifies an implementation detail.
2. Light and dark shades distinguish our per-cell and graph-coordinated variants.
3. Marker shape redundantly distinguishes per-cell, graph-coordinated, corner-enabled, and joint-`C0` variants.
4. In the reported experiments, graph coordination includes orientation propagation and selective cell merging. Merging therefore does not receive a separate color.
5. PLVIRA retains the LVIRA green because it extends the LVIRA fitting objective to a parabolic representation.

## Baselines

| Method | Role | Color | Hex | Marker | Line |
|---|---|---|---|---|---|
| Youngs | linear baseline | muted rose | `#B14E5E` | circle | solid |
| ELVIRA | linear baseline | ochre | `#B3811B` | square | dashed |
| LVIRA | linear baseline | green | `#2D7D64` | diamond | dash-dot |
| PLVIRA | higher-order LVIRA family | green | `#2D7D64` | circle | solid |
| PCIC | higher-order baseline | purple | `#7C5AA6` | square | dashed |
| QUASI | higher-order baseline | charcoal | `#4B5563` | diamond | dash-dot |

## Proposed methods

| Method | Role | Color | Hex | Marker | Line |
|---|---|---|---|---|---|
| Ours (linear, per-cell) | linear, no graph coordination | light blue | `#74A9CF` | up triangle | dashed |
| Ours (linear, graph-coordinated) | linear, propagation and merging | blue | `#2F6FA3` | down triangle | solid |
| Ours (linear + corners, graph-coordinated) | linear and line-line corners | teal | `#008C95` | X | solid |
| Ours (circular, per-cell) | circular, no graph coordination | gold | `#E6AB02` | up triangle | dashed |
| Ours (circular, graph-coordinated) | circular, propagation and merging | vermilion | `#D55E00` | down triangle | solid |
| Ours (circular + corners, graph-coordinated) | circular and sharp corners | crimson | `#B91C1C` | X | solid |

## Joint continuity refinement

The recommended default is to retain the underlying primitive-family color and encode joint `C0` refinement with a filled plus marker and a dotted line. This keeps a refined linear method visibly related to the linear family and a refined circular method related to the circular family.

| Method | Color | Marker | Line |
|---|---|---|---|
| Ours (linear, graph-coordinated + joint `C0`) | blue, `#2F6FA3` | filled plus | dotted |
| Ours (circular, graph-coordinated + joint `C0`) | vermilion, `#D55E00` | filled plus | dotted |
| Ours (circular + corners, graph-coordinated + joint `C0`) | crimson, `#B91C1C` | filled plus | dotted |

An alternative for plots whose sole purpose is to isolate continuity refinement is to use berry `#9C2F5F` as a temporary `C0` accent. It gives stronger visual separation in Figure 14, but it should not be used in panels that compare both linear-`C0` and circular-`C0` variants because it hides their primitive-family relationship.

## Figure-wide conventions

- Reference interface: black dashed.
- Numerical or geometric fitting floor: medium gray dotted.
- Interquartile range: transparent fill in the line color.
- Per-cell variants: upward triangles and dashed lines.
- Graph-coordinated variants: downward triangles and solid lines.
- Corner-enabled variants: X markers.
- Joint-`C0` variants: filled plus markers and dotted lines.

