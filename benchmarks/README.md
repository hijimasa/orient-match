# Comparison against other rotation-capable matchers

This directory measures OrientMatch next to the approaches people normally reach for when
a template has to be found under an unknown in-plane rotation. Everything here runs on the
same images, the same ground truth and the same success rule, so the numbers are
comparable to each other -- and only to each other.

## Methods

| Name in the CSV | What it is |
| --- | --- |
| `orient_match` | This library, at its default settings. |
| `bfncc_pyr` | The same search structure -- 0.5x coarse level, 3 degree bank, 1 degree refinement, 5 candidates -- scoring masked zero-mean NCC on **intensity** instead of the orientation field. This is the control that isolates the scoring function from the search. |
| `bfncc` | `cv::matchTemplate` with a rotated template bank at 1 degree over the full circle: the textbook way to add rotation, and the accuracy reference. |
| `fmt` | Fourier-Mellin: rotation from phase correlation of log-polar amplitude spectra, then a masked NCC for translation. |
| `orb` | ORB features, Lowe ratio test, RANSAC similarity fit. |
| `sift` | The same with SIFT. |

The Radon/sinogram method of
[radon-template-matching](https://github.com/hijimasa/radon-template-matching) is
deliberately not included. It is not an approach in common use, so it says little about
where OrientMatch stands; the baselines here are adapted from that repository's evaluation
harness (MIT, same author).

## Protocol

Per case: take a Kodak image in grayscale, rotate the source by theta and cut a 384x256
window from its centre; take the template from the theta = 0 window at a fixed offset from
its centre. The template therefore appears in the scene at angle theta, at a position the
protocol knows exactly. The scene is then optionally degraded.

- 12 images, 3 template offsets, 9 angles, 11 conditions: 324 cases per condition.
- Template 96x96 in a 384x256 scene.
- The 9 angles all have a fractional part (37.25, 78.5, ...). Angles on the 1 degree
  refinement grid would measure the grid rather than the method.
- Conditions: clean, Gaussian noise (sigma 25, 50), occlusion (25%, 50% of the rotated
  template's area), illumination change, JPEG quality 20, scale mismatch (0.95, 1.05), and
  two negative conditions where the template comes from a different image.
- A case succeeds when the angle error is at most 5 degrees **and** the position error is
  at most 5 pixels.

## How the timings are measured

Two columns, because the two questions are different.

- **`ms_total`** -- one shot: prepare the template, then search. This is what a program
  pays when it sees a template once.
- **`ms_frame`** -- the search alone, with per-template work hoisted out. This is what a
  stream of frames against a fixed template pays, and it is the case OrientMatch's
  reusable `Matcher` is built for.

Every method gets a prepared form, not just OrientMatch: the NCC baselines precompute
their whole rotated bank, Fourier-Mellin its template spectrum, ORB and SIFT their template
keypoints and descriptors. Charging a method for work a real application would do once
would understate all of them, and unevenly.

Two further choices keep the comparison honest:

- **One warm-up call per method** before the measured run. The first call in a process pays
  for thread pools, OpenCV's lazily allocated buffers and page faults -- a property of the
  process, not of the method. It is applied uniformly, once, and over thousands of cases it
  changes almost nothing; it exists so that no single method absorbs those costs because it
  happened to run first.
- **The order of methods within a case is rotated.** In a fixed order, one method is always
  measured right after the same neighbour, so that neighbour's memory traffic biases it the
  same way every time.

The machine must be otherwise idle while it runs; anything else competing for cores
lands in the timing columns, and not evenly across methods.

Timings come from one machine, one OpenCV build, and `OMP_NUM_THREADS=8`. That thread
count is not arbitrary: on the 22-core machine used here, every correlation-based method
runs *slower* with 20 threads than with 8, because the per-frame work is bound by memory
bandwidth rather than arithmetic, and the methods lose throughput at different rates. A
badly oversubscribed run does not just make the numbers larger, it changes the ranking.
Treat the ratios as indicative and re-measure on your own target, at your own thread count,
before choosing on speed.

## Running it

```sh
benchmarks/fetch_kodak.sh                   # downloads 24 images into datasets/
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DORIENT_MATCH_BUILD_BENCHMARKS=ON
cmake --build build -j
OMP_NUM_THREADS=8 ./build/orient_match_compare 12 96 > results.csv
python3 benchmarks/summarize.py results.csv
```

`orient_match_compare <images> <template size> [only method]`. The dataset is not committed;
`datasets/` is ignored. `results.csv.gz` here is the run the README tables were built from,
so the numbers can be checked without re-running anything.

## What the comparison does and does not show

It covers one setting: a single grayscale template, known scale, one instance to find, in
384x256 scenes from a photographic dataset. Within that setting the ranking is meaningful.
Outside it -- multiple instances, unknown scale, 3-D pose, textureless industrial parts,
much larger images -- it says nothing, and a method that does poorly here may be the right
choice there. The feature-based methods in particular are built for a wider problem than
this one, and are handicapped by a 96x96 template; Fourier-Mellin assumes the two images
are related by one global rotation, which a small template inside a large scene is not.
