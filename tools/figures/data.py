"""Search results used by the figures.

Everything here mirrors what src/orient_match.cpp does, so the maps, peaks and scores
drawn in figs/ are real numbers rather than sketches.
"""
import numpy as np
import cv2

from core import (image_planes, make_canvas, make_scene, rotate_canvas, score_map)

COARSE_SCALE = 0.5
COARSE_STEP_DEG = 3.0
FINE_OFFSETS = [-3, -2, -1, 0, 1, 2, 3]
REFINE_TOP_K = 5

_cache = {}


def scene():
    """(image, template, true angle, true centre) plus the template canvas."""
    if "scene" not in _cache:
        img, tmpl, angle, centre = make_scene()
        canvas = make_canvas(tmpl)
        _cache["scene"] = (img, tmpl, angle, centre, canvas)
    return _cache["scene"]


def coarse_to_fine():
    """Run the two-stage search once and keep every intermediate result."""
    if "c2f" in _cache:
        return _cache["c2f"]

    img, tmpl, true_angle, _, (cx, cy, mask, size) = scene()
    cimg = cv2.resize(img, None, fx=COARSE_SCALE, fy=COARSE_SCALE, interpolation=cv2.INTER_AREA)
    csize = int(round(size * COARSE_SCALE))
    ccx, ccy, cmask = (cv2.resize(a, (csize, csize), interpolation=cv2.INTER_AREA)
                       for a in (cx, cy, mask))
    cix, ciy, cen = image_planes(cimg)

    angles = np.arange(0.0, 360.0, COARSE_STEP_DEG)
    maps, peaks = {}, []
    for angle in angles:
        tx, ty, support = rotate_canvas(ccx, ccy, cmask, csize, angle)
        scores = score_map(cix, ciy, cen, tx, ty, support)
        peaks.append((float(scores.max()), *np.unravel_index(scores.argmax(), scores.shape)))
        maps[angle] = scores
    peaks = np.array(peaks)
    order = np.argsort(-peaks[:, 0])[:REFINE_TOP_K]

    ix, iy, en = image_planes(img)
    padding = int(np.ceil(2.0 / COARSE_SCALE)) + 3
    fine, rois = {}, {}
    for rank, index in enumerate(order):
        base = angles[index]
        # Inverse of OpenCV's half-pixel resize mapping, as in Matcher::match().
        centre_x = (peaks[index, 2] + (csize - 1) / 2.0 + 0.5) / COARSE_SCALE - 0.5
        centre_y = (peaks[index, 1] + (csize - 1) / 2.0 + 0.5) / COARSE_SCALE - 0.5
        w = min(img.shape[1], size + 2 * padding)
        h = min(img.shape[0], size + 2 * padding)
        x0 = int(np.clip(round(centre_x - (w - 1) / 2.0), 0, img.shape[1] - w))
        y0 = int(np.clip(round(centre_y - (h - 1) / 2.0), 0, img.shape[0] - h))
        rois[rank] = (x0, y0, w, h)
        roi = (slice(y0, y0 + h), slice(x0, x0 + w))
        for offset in FINE_OFFSETS:
            angle = base + offset
            tx, ty, support = rotate_canvas(cx, cy, mask, size, angle)
            scores = score_map(ix[roi], iy[roi], en[roi], tx, ty, support)
            fine[(rank, offset)] = (float(scores.max()), scores, angle)

    result = dict(img=img, size=size, csize=csize, angles=angles, maps=maps, peaks=peaks,
                  order=order, fine=fine, rois=rois, true_angle=true_angle, padding=padding,
                  img_shape=img.shape, coarse_shape=maps[0.0].shape)
    _cache["c2f"] = result
    return result


def cost_estimate():
    """Multiply-adds for an exhaustive search versus the two staged searches."""
    d = coarse_to_fine()
    size, csize = d["size"], d["csize"]
    coarse_positions = d["coarse_shape"][0] * d["coarse_shape"][1]
    coarse = len(d["angles"]) * coarse_positions * csize * csize
    x0, y0, w, h = d["rois"][0]
    fine_positions = (h - size + 1) * (w - size + 1)
    fine = len(d["order"]) * len(FINE_OFFSETS) * fine_positions * size * size
    rows, cols = d["img_shape"]
    exhaustive = 360 * (rows - size + 1) * (cols - size + 1) * size * size
    return dict(coarse=coarse, fine=fine, exhaustive=exhaustive,
                ratio=exhaustive / (coarse + fine), fine_positions=fine_positions)


if __name__ == "__main__":
    d = coarse_to_fine()
    top = [(d["angles"][i], round(d["peaks"][i, 0], 3)) for i in d["order"]]
    best = max(d["fine"].items(), key=lambda kv: kv[1][0])
    print("coarse map", d["coarse_shape"], "canvas", d["size"], "coarse canvas", d["csize"])
    print("top K (angle, score):", top)
    print("best fine pose: angle %.1f score %.4f" % (best[1][2], best[1][0]))
    print("cost:", {k: (round(v, 1) if k == "ratio" else v) for k, v in cost_estimate().items()})
