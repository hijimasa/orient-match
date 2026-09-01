"""Search results used by the figures.

Everything here mirrors what src/orient_match.cpp does, so the maps, peaks and scores
drawn in figs/ are real numbers rather than sketches.
"""
import numpy as np
import cv2

from core import (image_planes, make_canvas, make_scene, rotate_canvas, score_map)

COARSE_SCALE = 0.5
COARSE_STEP_DEG = 3.0
SCAN_STEP_DEG = 12.0                  # kScanStepDeg: the global scan skips to this step
SCAN_STRIDE = int(SCAN_STEP_DEG / COARSE_STEP_DEG)
FINE_OFFSETS = [-3, -2, -1, 0, 1, 2, 3]
OUTER_FINE_OFFSETS = [-6, -5, -4, 4, 5, 6]   # searched for the leading candidate only
REFINE_TOP_K = 5

_cache = {}


def scene():
    """(image, template, true angle, true centre) plus the template canvas."""
    if "scene" not in _cache:
        img, tmpl, angle, centre = make_scene()
        canvas = make_canvas(tmpl)
        _cache["scene"] = (img, tmpl, angle, centre, canvas)
    return _cache["scene"]


def dense_scan():
    """Every coarse angle scored over the whole coarse image.

    Matcher::match only visits every SCAN_STRIDE-th of these. The full set is computed
    here because the figures show what the skipped angles would have looked like, which
    is what makes the sparse scan legible.
    """
    if "dense" in _cache:
        return _cache["dense"]

    img, _, _, _, (cx, cy, mask, size) = scene()
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

    _cache["dense"] = dict(angles=angles, maps=maps, peaks=np.array(peaks), csize=csize,
                           coarse=(cix, ciy, cen), canvas=(ccx, ccy, cmask), cimg=cimg)
    return _cache["dense"]


def coarse_to_fine():
    """Run the staged search once and keep every intermediate result."""
    if "c2f" in _cache:
        return _cache["c2f"]

    img, tmpl, true_angle, _, (cx, cy, mask, size) = scene()
    dense = dense_scan()
    angles, maps, peaks, csize = dense["angles"], dense["maps"], dense["peaks"], dense["csize"]
    cix, ciy, cen = dense["coarse"]
    ccx, ccy, cmask = dense["canvas"]
    cimg = dense["cimg"]

    # Stage A: every position, but only every SCAN_STRIDE-th angle.
    scanned = list(range(0, len(angles), SCAN_STRIDE))

    # Candidates are kept at least half a canvas apart, so each one is a distinct place.
    separation = max(1, csize // 2)
    order = []
    for index in sorted(scanned, key=lambda i: -peaks[i, 0]):
        if len(order) >= REFINE_TOP_K:
            break
        if any(abs(peaks[j, 2] - peaks[index, 2]) < separation and
               abs(peaks[j, 1] - peaks[index, 1]) < separation for j in order):
            continue
        order.append(int(index))
    order = np.array(order)

    # Stage B: the angles the global scan stepped over, on a window around each candidate.
    radius = SCAN_STRIDE // 2
    pad = int(np.ceil(csize * np.sin(np.radians(SCAN_STEP_DEG / 2.0)))) + 2
    bases, locations = [], []
    for index in order:
        y0 = int(peaks[index, 1]); x0 = int(peaks[index, 2])
        wy0, wx0 = max(0, y0 - pad), max(0, x0 - pad)
        wy1 = min(cimg.shape[0], y0 + csize + pad)
        wx1 = min(cimg.shape[1], x0 + csize + pad)
        win = (slice(wy0, wy1), slice(wx0, wx1))
        best = (-2.0, int(index), y0, x0)
        for step in range(-radius, radius + 1):
            j = (int(index) + step) % len(angles)
            tx, ty, support = rotate_canvas(ccx, ccy, cmask, csize, angles[j])
            scores = score_map(cix[win], ciy[win], cen[win], tx, ty, support)
            py, px = np.unravel_index(scores.argmax(), scores.shape)
            if float(scores.max()) > best[0]:
                best = (float(scores.max()), j, wy0 + py, wx0 + px)
        bases.append(best[1])
        locations.append((best[2], best[3]))

    # Stage C: full resolution, around each refined candidate.
    ix, iy, en = image_planes(img)
    padding = int(np.ceil(2.0 / COARSE_SCALE)) + 3
    fine, rois = {}, {}
    for rank, (index, (py, px)) in enumerate(zip(bases, locations)):
        base = angles[index]
        # Inverse of OpenCV's half-pixel resize mapping, as in Matcher::match().
        centre_x = (px + (csize - 1) / 2.0 + 0.5) / COARSE_SCALE - 0.5
        centre_y = (py + (csize - 1) / 2.0 + 0.5) / COARSE_SCALE - 0.5
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

    # The leading candidate is searched one ring wider, because the coarse level can be a
    # step or two off the full-resolution optimum.
    leader = max(fine, key=lambda k: fine[k][0])[0]
    x0, y0, w, h = rois[leader]
    roi = (slice(y0, y0 + h), slice(x0, x0 + w))
    for offset in OUTER_FINE_OFFSETS:
        angle = angles[bases[leader]] + offset
        tx, ty, support = rotate_canvas(cx, cy, mask, size, angle)
        scores = score_map(ix[roi], iy[roi], en[roi], tx, ty, support)
        fine[(leader, offset)] = (float(scores.max()), scores, angle)

    result = dict(img=img, size=size, csize=csize, angles=angles, scanned=scanned, maps=maps,
                  peaks=peaks, order=order, bases=bases, fine=fine, rois=rois,
                  true_angle=true_angle, padding=padding, img_shape=img.shape,
                  coarse_shape=maps[0.0].shape, leader=leader)
    _cache["c2f"] = result
    return result


def cost_estimate():
    """Multiply-adds for an exhaustive search versus the two staged searches."""
    d = coarse_to_fine()
    size, csize = d["size"], d["csize"]
    coarse_positions = d["coarse_shape"][0] * d["coarse_shape"][1]
    coarse = len(d["scanned"]) * coarse_positions * csize * csize
    x0, y0, w, h = d["rois"][0]
    fine_positions = (h - size + 1) * (w - size + 1)
    fine = (len(d["order"]) * len(FINE_OFFSETS) + len(OUTER_FINE_OFFSETS)) * \
        fine_positions * size * size
    rows, cols = d["img_shape"]
    exhaustive = 360 * (rows - size + 1) * (cols - size + 1) * size * size
    return dict(coarse=coarse, fine=fine, exhaustive=exhaustive,
                ratio=exhaustive / (coarse + fine), fine_positions=fine_positions)


if __name__ == "__main__":
    d = coarse_to_fine()
    top = [(d["angles"][i], round(d["peaks"][i, 0], 3)) for i in d["order"]]
    best = max(d["fine"].items(), key=lambda kv: kv[1][0])
    print("coarse map", d["coarse_shape"], "canvas", d["size"], "coarse canvas", d["csize"])
    print("scanned angles: %d of %d" % (len(d["scanned"]), len(d["angles"])))
    print("top K (angle, score):", top)
    print("after local angle refinement:", [d["angles"][i] for i in d["bases"]])
    print("best fine pose: angle %.1f score %.4f" % (best[1][2], best[1][0]))
    print("cost:", {k: (round(v, 1) if k == "ratio" else v) for k, v in cost_estimate().items()})
