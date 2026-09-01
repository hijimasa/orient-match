"""Shared scene + OrientMatch-equivalent math for figure generation."""
import numpy as np, cv2

def orientation_field(img, blur_sigma=1.0, eps_ratio=0.1):
    v = img.astype(np.float32)
    if blur_sigma > 0:
        v = cv2.GaussianBlur(v, (0, 0), blur_sigma, blur_sigma)
    gx = cv2.Sobel(v, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(v, cv2.CV_32F, 0, 1, 3)
    mag = cv2.magnitude(gx, gy)
    eps = eps_ratio * float(mag.mean()) + 1e-6
    d = mag + eps
    return gx / d, gy / d

def gaussian_window(h, w, sigma=1.0):
    ny = (2.0 * np.arange(h) / (h - 1.0) - 1.0)[:, None]
    nx = (2.0 * np.arange(w) / (w - 1.0) - 1.0)[None, :]
    return np.exp(-(nx**2 + ny**2) / (2 * sigma**2)).astype(np.float32)

def make_canvas(tmpl, blur_sigma=1.0, eps_ratio=0.1, window_sigma=1.0):
    fx, fy = orientation_field(tmpl, blur_sigma, eps_ratio)
    w = gaussian_window(*tmpl.shape, window_sigma)
    fx, fy = fx * w, fy * w
    h, wd = tmpl.shape
    size = int(np.ceil(np.hypot(h, wd)))
    if (h & 1) == (wd & 1) and (size & 1) != (h & 1):
        size += 1
    tx, ty = (size - wd) / 2.0, (size - h) / 2.0
    M = np.array([[1, 0, tx], [0, 1, ty]], np.float32)
    cx = cv2.warpAffine(fx, M, (size, size))
    cy = cv2.warpAffine(fy, M, (size, size))
    mask = cv2.warpAffine(np.ones_like(fx), M, (size, size))
    return cx, cy, mask, size

def rotate_canvas(cx, cy, mask, size, angle_deg, rotate_values=True):
    c = (size - 1) / 2.0
    M = cv2.getRotationMatrix2D((c, c), angle_deg, 1.0)
    wx = cv2.warpAffine(cx, M, (size, size))
    wy = cv2.warpAffine(cy, M, (size, size))
    wm = cv2.warpAffine(mask, M, (size, size))
    support = (wm > 0).astype(np.float32)
    if rotate_values:
        r = np.deg2rad(-angle_deg)
        co, si = np.cos(r), np.sin(r)
        wx, wy = wx * co - wy * si, wx * si + wy * co
    return wx, wy, support

ENERGY_FLOOR_RATIO = 1e-6

def score_map(ix, iy, energy, tx, ty, support):
    nx = cv2.matchTemplate(ix, tx, cv2.TM_CCORR)
    ny = cv2.matchTemplate(iy, ty, cv2.TM_CCORR)
    ie = cv2.matchTemplate(energy, support, cv2.TM_CCORR)
    te = max(float((tx * tx).sum() + (ty * ty).sum()), 1e-12)
    # The support area is the largest energy a position can hold; anything far below it
    # is textureless, and dividing by it would amplify round-off. See kEnergyFloorRatio.
    floor = max(ENERGY_FLOOR_RATIO * float(support.sum()), 1e-12)
    den = np.sqrt(np.maximum(ie, floor) * te)
    s = (nx + ny) / den
    s[~np.isfinite(s)] = -1.0
    s[(s < -1.001) | (s > 1.001)] = -1.0
    return s

def image_planes(img):
    ix, iy = orientation_field(img)
    return ix, iy, ix * ix + iy * iy

# ---------------- synthetic scene ----------------
def make_template(size=64):
    t = np.full((size, size), 235, np.uint8)
    cv2.rectangle(t, (10, 18), (54, 44), 70, -1)
    cv2.rectangle(t, (36, 6), (52, 22), 70, -1)
    cv2.circle(t, (22, 31), 7, 225, -1)
    return cv2.GaussianBlur(t, (0, 0), 0.6)

def paste_rotated(canvas, patch, center, angle_deg, gain=1.0, bias=0.0):
    h, w = patch.shape
    big = int(np.ceil(np.hypot(h, w))) + 4
    M = np.array([[1, 0, (big - w) / 2.0], [0, 1, (big - h) / 2.0]], np.float32)
    p = cv2.warpAffine(patch.astype(np.float32), M, (big, big), borderValue=255)
    c = (big - 1) / 2.0
    R = cv2.getRotationMatrix2D((c, c), angle_deg, 1.0)
    p = cv2.warpAffine(p, R, (big, big), borderValue=255)
    alpha = (p < 250).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.6)
    p = p * gain + bias
    x0 = int(round(center[0] - c)); y0 = int(round(center[1] - c))
    roi = canvas[y0:y0 + big, x0:x0 + big].astype(np.float32)
    canvas[y0:y0 + big, x0:x0 + big] = np.clip(roi * (1 - alpha) + p * alpha, 0, 255)
    return canvas

def make_scene(w=360, h=240, angle=58.0, center=(232, 130)):
    yy = np.linspace(0, 1, h)[:, None] * np.ones((1, w))
    xx = np.linspace(0, 1, w)[None, :] * np.ones((h, 1))
    img = (120 + 55 * xx + 25 * yy).astype(np.float32)
    rng = np.random.default_rng(7)
    img += rng.normal(0, 3.0, img.shape)
    tmpl = make_template()
    img = paste_rotated(img, tmpl, center, angle, gain=0.55, bias=45)   # target: low contrast
    img = paste_rotated(img, tmpl, (95, 78), -30.0, gain=0.9, bias=-15) # distractor pose
    cv2.circle(img, (300, 46), 17, 210, -1)
    return np.clip(img, 0, 255).astype(np.uint8), tmpl, angle, center
