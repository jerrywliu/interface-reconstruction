# Interface reconstruction

An example implementation in Python of an interface reconstruction method, using linear/circular elements and cusps.

## Contributors

Jerry Liu, jwl50@stanford.edu

## Table of Contents

- [Algorithms](#algorithms)
- [Static experiments](#static-experiments)
- [Advection experiments](#advection-experiments)

## Algorithms

Our algorithms consist of two main features:
- Circular facets
- Corner facets (either linear or circular), which requires merging cells

We currently support the following algorithms:
- Baselines:
  - Youngs
  - ELVIRA
- Our algorithms:
  - No merging.
    - "Safe" linear
    - "Safe" circular
  - With merging.
    - Linear+corners
    - Circular+corners

## Static experiments

Randomly oriented lines:
```
./experiments/static/run_lines.sh 
```

Randomly oriented circles:
```
./experiments/static/run_circles.sh
```

TODO Randomly oriented ellipses (circles)

TODO Randomly generated polygons (corners)

TODO Pac-man (circular corners)

## Advection experiments

Zalesak:
```
python3 run.py --config advection/zalesak/50/zalesak_50_ccorner
python3 run.py --config advection/zalesak/100/zalesak_100_ccorner
```

x+o:
```
python3 run.py --config advection/x+o/50/x+o_50_safecircle # Works
python3 run.py --config advection/x+o/50/x+o_50_circular
python3 run.py --config advection/x+o/50/x+o_50_ccorner # Producing circular facets with inverted curvature and incorrect corners... TODO
python3 run.py --config advection/x+o/100/x+o_100_ccorner
python3 run.py --config advection/x+o/150/x+o_150_ccorner # Mostly ok, but need to adjust corner threshold. Some corners failing and reforming. TODO
```

Vortex:
Algos: safecirclec0, safecircle, safelinear
Resolutions: 32, 64, 128
```
python3 run.py --config advection/vortex/32/vortex_32_safecirclec0
python3 run.py --config advection/vortex/50/vortex_50_safecircle
python3 run.py --config advection/vortex/100/vortex_100_safecircle
```

## TODO
Mainly two lingering bugs with the full algorithm.
- Corners: sometimes fail to generate, other times generate in completely wrong places. Maybe because of 1. threshold used to generate corner, 2. failing to find a proper facet to extend (resolution issue).
- Circles: very rarely (usually low resolution) will choose the wrong curvature solution. Likely because of heuristics I use to choose the orientation.

C0 doesn't seem to work with merge-less algorithms: e.g. safe-circle.

Fixes:
- Corner threshold.
- findOrientation() in MergeMesh needs to be fixed.

Checkpointing: a bug relating to circular references within MergeMesh. Likely because of neighbor storing.

<div align="right"><a href="#table-of-contents">back to top </a></div>