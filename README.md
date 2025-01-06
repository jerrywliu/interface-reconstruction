# Interface reconstruction

An example implementation of an interface reconstruction method, using linear/circular elements and cusps.

## Contributors

Jerry Liu, jwl50@stanford.edu

## Table of Contents

1. [Stuff](#stuff)
    - [Stuff](#stuff)
1. [Stuff]

<div align="right"><a href="#table-of-contents">back to top </a></div>

```bash
$ python3 run.py --config ${setting}_${resolution}_${algo}
```

python3 run.py --config advection/zalesak/50/zalesak_50_linear
python3 -m experiments.static.lines --config static/line --num_lines 10 --facet_algo [Youngs/LVIRA/safe_linear] --save_name [line_youngs/line_lvira/line_safelinear]