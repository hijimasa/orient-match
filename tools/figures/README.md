# Figure generators

The SVGs in [`figs/`](../../figs) are generated, not drawn by hand. Every arrow, score map
and number in them comes from running the OrientMatch computation on a small synthetic
scene, so the pictures cannot drift away from what the library actually does.

## Requirements

Python 3 with `numpy` and `opencv-python`. Nothing here is part of the C++ build.

```sh
pip install numpy opencv-python
```

## Usage

```sh
cd tools/figures
python3 fig_principle.py        # figs/principle.svg
python3 fig_coarse_to_fine.py   # figs/coarse-to-fine.svg
python3 fig_pipeline.py         # figs/pipeline.svg
python3 data.py                 # print the numbers the figures quote
```

Each script overwrites its own file under `figs/` and is deterministic: re-running with an
unchanged source produces a byte-identical SVG.

## Files

| File | Purpose |
| --- | --- |
| `core.py` | Python mirror of the library math (orientation field, canvas, rotation, normalized score) plus the synthetic template and scene |
| `data.py` | Runs the coarse and fine stages once and caches every intermediate result; also the multiply-add cost estimate |
| `shapes.py` | Vector drawing of the synthetic part used as the template |
| `svgutil.py` | Small SVG writer: cards, arrows, sampled vector fields, run-length encoded heatmaps |
| `fig_*.py` | One script per figure |

`core.py` intentionally repeats the formulas of `src/orient_match.cpp` rather than binding
to the C++ code. If the algorithm changes there, update `core.py` and regenerate.
