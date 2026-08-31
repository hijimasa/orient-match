# OrientMatch

[日本語](README.ja.md) | **English**

OrientMatch packages dense gradient-orientation correlation and coarse-to-fine pose
search into a small C++17/OpenCV library. It is an engineering-oriented combination of
established techniques, with a reusable matcher API and explicit operating assumptions.

It is intended as a transparent, reproducible baseline for the following setting:

- one grayscale template;
- known, fixed scale;
- one best match in a larger grayscale image;
- translation and in-plane rotation;
- CPU execution, optionally parallelized with OpenMP.

The matcher uses the softly gated orientation field

```text
z = (gx, gy) / (sqrt(gx^2 + gy^2) + eps)
```

and normalized, non-centered correlation between the two vector-field components.
It searches all positions and coarse angles in a downsampled image, retains the best
angle candidates, and refines their positions and neighboring angles at full resolution.

## Positioning

OrientMatch is an early reference-library release. It makes **no claim of a novel
algorithm or state-of-the-art performance**. Gradient-direction similarity, rotated
template banks, and pyramidal pose search are established ideas with precedents including
Steger's shape-based matching and later oriented-gradient template methods:

- C. Steger, “Similarity Measures for Occlusion, Clutter, and Illumination Invariant
  Object Recognition,” DAGM 2001.
- C. Steger, “Occlusion, Clutter, and Illumination Invariant Object Recognition,”
  ISPRS 2002.
- S. Hinterstoisser et al., “Gradient Response Maps for Real-Time Detection of
  Texture-Less Objects,” TPAMI 2012.
- Y. Konishi et al., “Fast and Precise Template Matching Based on Oriented Gradients,”
  ECCV Workshops 2012.

The contribution here is packaging and engineering. In particular, OrientMatch provides:

- one reusable `Matcher` that builds a rotation bank from a single template;
- a dense, continuous two-component field rather than sparse quantized orientations;
- a soft magnitude gate and whole-patch energy-normalized correlation;
- a CPU/OpenCV implementation with optional OpenMP parallelism;
- documented coordinate and angle conventions, validation, tests, and CMake installation;
- an intentionally narrow API for fixed-scale, single-best-pose matching.

These choices form a useful implementation point in the design space; the individual
ingredients and the overall method family are not presented as new.

The implementation was extracted from the image-space control method evaluated in
[radon-template-matching](https://github.com/hijimasa/radon-template-matching).

## Related implementations

The following projects overlap with OrientMatch, but optimize for different
representations or use cases:

| Implementation | Shared ground | Main difference from OrientMatch |
|---|---|---|
| [OpenCV `matchTemplate`](https://docs.opencv.org/4.x/de/da9/tutorial_template_matching.html) | Dense image-space correlation | Searches translation for a fixed template; rotation and coarse-to-fine orchestration are left to the caller. |
| [OpenCV LINEMOD](https://docs.opencv.org/4.x/d7/d07/classcv_1_1linemod_1_1Detector.html) | Gradient orientation and template pyramids | Uses selected, quantized orientation features and normally registers templates for the required views or poses. |
| [`shape_based_matching`](https://github.com/meiqua/shape_based_matching) | C++/OpenCV, gradient orientation, pyramids, rotated/scaled templates | A LINEMOD/HALCON-style sparse-feature matcher with pose-template generation, SIMD, and NMS-oriented detection features. |
| [OpenCV `GeneralizedHoughGuil`](https://docs.opencv.org/4.x/d3/d20/classcv_1_1GeneralizedHoughGuil.html) | Edge gradients and position/rotation/scale estimation | Uses generalized-Hough voting instead of dense normalized correlation. |
| [`batchmatch`](https://github.com/wlruys/batchmatch) | Normalized gradient fields and transformation search | A PyTorch/FFT-oriented registration toolkit for batched affine grids, rather than a small C++ local-refinement matcher. |
| [`corrmatch-rs`](https://github.com/VitalyVorobyev/corrmatch-rs) | Rotation-aware coarse-to-fine template search | Uses intensity ZNCC/SSD in Rust rather than dense gradient-orientation correlation in OpenCV. |

No claim is made that this list is exhaustive. The practical distinction is the complete
combination exposed by this library: a single-template C++ API, dense continuous
orientation fields, normalized correlation, and coarse-to-fine translation/rotation
search. For a new application, benchmark the alternatives above on representative data
rather than assuming one method is universally faster or more accurate.

## Requirements

- CMake 3.16 or newer
- a C++17 compiler
- OpenCV (`core` and `imgproc`; `imgcodecs` is needed only for the example CLI)
- OpenMP (optional)

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run the example CLI:

```bash
./build/orient_match_cli search.png template.png
./build/orient_match_cli search.png template.png -45 90
```

The output is a JSON object containing the absolute template-center coordinates, rotation
angle in degrees, and correlation score.

## C++ API

```cpp
#include <opencv2/imgcodecs.hpp>
#include <orient_match/orient_match.hpp>

cv::Mat image = cv::imread("search.png", cv::IMREAD_GRAYSCALE);
cv::Mat templ = cv::imread("template.png", cv::IMREAD_GRAYSCALE);

orient_match::MatcherOptions options;
options.coarse_scale = 0.5;
options.coarse_angle_step_deg = 3.0;
options.fine_angle_step_deg = 1.0;
options.refine_top_k = 5;

// Construction precomputes the template field and coarse rotated-template bank.
orient_match::Matcher matcher(templ, options);
orient_match::MatchResult result = matcher.match(image);

if (result) {
    std::cout << result.center.x << ", " << result.center.y << "\n";
    std::cout << result.angle_deg << " deg, score=" << result.score << "\n";
}
```

An optional single-channel template mask is supported:

```cpp
orient_match::Matcher matcher(templ, mask, options);
```

Mask pixels equal to zero are excluded. Gradients are computed before masking so the mask
boundary does not create a synthetic template edge. After rotation, the interpolated mask
is converted back to a binary support before energy normalization; this keeps the cosine
score bounded despite interpolation at the mask boundary.

### CMake consumer

After installing OrientMatch, a consuming project can use:

```cmake
find_package(OrientMatch CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE OrientMatch::orient_match)
```

## Coordinates and angle convention

- Pixel centers follow OpenCV coordinates: the top-left pixel center is `(0, 0)`.
- `MatchResult::center` is the absolute center of the transformed template in the input.
- Positive angles are counter-clockwise in the convention used by
  `cv::getRotationMatrix2D`.
- Returned angles are normalized to `[0, 360)`.
- `angle_extent_deg` defines the half-open interval
  `[angle_start_deg, angle_start_deg + angle_extent_deg)`.

## Algorithm outline

1. Blur the grayscale image and template, then compute Sobel gradients.
2. Normalize each vector using a soft, image-adaptive gate.
3. Apply a Gaussian window and optional mask to the template field.
4. At the coarse image level, correlate every configured coarse rotation over all valid
   translations.
5. Keep the best `refine_top_k` coarse-angle candidates.
6. At full resolution, search a local position ROI and neighboring fine angles.
7. Return the highest normalized vector-field correlation.

Template construction precomputes the coarse rotated-template bank, making a `Matcher`
suitable for repeated use on multiple frames. The object is immutable after construction;
`match()` uses only local working buffers.

## Scope and known limitations

- Scale is assumed known. A scale mismatch is not searched.
- Only the single best match is returned; there is no NMS or multi-instance output yet.
- Position and angle are discrete. There is no subpixel/sub-degree peak interpolation yet.
- Coarse-to-fine search is heuristic and can miss a narrow global optimum.
- Excessively fine angle grids are rejected instead of allocating an unbounded template
  bank; the current limits are 4,096 coarse angles, 4,097 fine offsets, and 1,000,000 fine
  tasks per match.
- `coarse_scale` must leave the square template canvas at least 3 x 3 pixels.
- The correlation score is not a calibrated probability or universal detection threshold.
- A square rotation canvas must fit entirely inside the search image.
- The current implementation rotates full-resolution templates during refinement; very
  large templates or wide fine searches can be expensive.
- This project does not currently attempt to replace feature-based matching, learned
  detectors, or full affine registration outside its stated fixed-scale setting.

## License

MIT. See [LICENSE](LICENSE).
