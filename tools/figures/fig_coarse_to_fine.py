import numpy as np
import cv2

from data import coarse_to_fine, cost_estimate
from svgutil import *

D = coarse_to_fine()
angles, maps, peaks, order = D["angles"], D["maps"], D["peaks"], D["order"]
size, csize, fine, rois = D["size"], D["csize"], D["fine"], D["rois"]
H_, W_ = D["img_shape"]
ORANGE, BLUE, GREEN = "#e08a3c", "#457b9d", "#18864b"

W, H = 1400, 858
g = Svg(W, H, "OrientMatch coarse-to-fine pose search",
        "The coarse stage scores every position for every coarse angle on a half-resolution image. "
        "The best K poses are refined at full resolution over a small position window and neighbouring "
        "angles, which costs a small fraction of an exhaustive full-resolution search.")
g.text(W/2, 50, "Why the search is split in two stages", 28, INK, "700", "middle")
g.text(W/2, 80, "the score is a 3-D volume over (x, y, θ) — sample all of it cheaply, then look closely "
       "only where it is promising", 15, MUTED, "400", "middle")

# ------------------------------------------------ stage 1
y0 = 104
card(g, 40, y0, 640, 500, "STAGE 1   coarse — all positions × all 3° angles, on the 0.5× image",
     accent="#f4a261", head_fill="#f4a261", title_fill="#3d2608", title_size=15)
lo, hi = 0.0, float(peaks[:, 0].max())
DS = 0.5
mh, mw = cv2.resize(maps[0.0], None, fx=DS, fy=DS, interpolation=cv2.INTER_AREA).shape
cell = 176.0/mw
for i, a in enumerate((0.0, 30.0, 57.0)):
    x = 76 + i*192
    g.text(x + 88, y0+72, f"θ = {a:.0f}°", 13, "#8a4a04" if a == 57.0 else MUTED, "700", "middle")
    heat_rects(g, cv2.resize(maps[a], None, fx=DS, fy=DS, interpolation=cv2.INTER_AREA),
               x, y0+82, cell, HEAT, lo=0.04, hi=hi, levels=14)
    g.rect(x, y0+82, mw*cell, mh*cell, rx=3, fill="none",
           stroke="#d9a066" if a == 57.0 else "#c3cbd8", stroke_width=2 if a == 57.0 else 1)
    r, c = peaks[list(angles).index(a), 1]*DS, peaks[list(angles).index(a), 2]*DS
    g.add(f'<circle cx="{x+(c+0.5)*cell:.1f}" cy="{y0+82+(r+0.5)*cell:.1f}" r="5" fill="none" '
          f'stroke="#7a3d00" stroke-width="2"/>')
    g.text(x + 88, y0+82+mh*cell+18, f"peak {peaks[list(angles).index(a),0]:.2f}", 12, MUTED, "400", "middle")
# shared colour scale legend
cbx, cby = 236, y0+82+mh*cell+34
for k in range(28):
    g.rect(cbx + k*8, cby, 8.4, 12, fill=HEAT(k/27.0))
g.rect(cbx, cby, 224, 12, rx=0, fill="none", stroke="#d3dae4")
g.text(cbx-8, cby+11, "0.0", 11, "#9aa4b5", "400", "end")
g.text(cbx+232, cby+11, f"{hi:.2f}", 11, "#9aa4b5", "400", "start")
g.text(360, cby+34, "one complete translation score map per coarse angle — 120 of them",
       13, MUTED, "400", "middle")

# angle profile
px0, py0, pw, ph = 90, y0+276, 552, 128
g.rect(px0, py0, pw, ph, rx=8, fill="#ffffff", stroke="#e2e7ef")
for t in (0.25, 0.5, 0.75):
    yy = py0 + ph - t*ph
    g.add(f'<path d="M{px0} {yy:.1f} H{px0+pw}" stroke="#eef1f6" stroke-width="1"/>')
    g.text(px0-8, yy+4, f"{t:.2f}", 11, "#9aa4b5", "400", "end")
pts = " ".join(f"{px0 + a/360.0*pw:.1f},{py0 + ph - max(p,0)*ph:.1f}"
               for a, p in zip(angles, peaks[:, 0]))
g.add(f'<polyline points="{pts}" fill="none" stroke="{ORANGE}" stroke-width="2.4" stroke-linejoin="round"/>')
for i in order:
    ax = px0 + angles[i]/360.0*pw; ay = py0 + ph - peaks[i, 0]*ph
    g.add(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="4.5" fill="#ffffff" stroke="#b85c00" stroke-width="2.2"/>')
for a in (0, 90, 180, 270, 360):
    g.text(px0 + a/360.0*pw, py0+ph+18, f"{a}°", 11, "#9aa4b5", "400", "middle")
g.text(px0, py0-10, "best score found at each coarse angle", 12.5, MUTED, "600")
g.text(px0+pw, py0-10, "○ = kept for refinement", 12, "#b85c00", "600", "end")
bx = px0 + 330/360.0*pw
g.add(f'<path d="M{bx:.1f} {py0+ph-peaks[110,0]*ph-12:.1f} v-22" stroke="#9aa4b5" stroke-width="1.4"/>')
g.text(bx, py0+ph-peaks[110,0]*ph-38, "second object", 11.5, MUTED, "400", "middle")
g.text(90, y0+468, "cheap because the image, the template and the angle grid are all coarse",
       13, MUTED)

# ------------------------------------------------ top-K
g.add(f'<path d="M690 330 H726" stroke="#6f7c90" stroke-width="2.5"/>')
arrow(g, 690, 330, 730, 330, "#6f7c90", 2.5, 10)
card(g, 742, 250, 156, 210, "TOP K", accent="#2f3e5c", head_fill="#2f3e5c", title_size=14, head_h=36)
g.text(820, 306, "K = 5 poses", 12.5, MUTED, "400", "middle")
for j, i in enumerate(list(order)[:4]):
    yy = 326 + j*30
    g.rect(760, yy, 120, 24, rx=6, fill="#fff3e6")
    g.text(770, yy+16, f"θ {angles[i]:.0f}°", 12, "#7a4200", "700", font=MONO)
    g.text(870, yy+16, f"{peaks[i,0]:.2f}", 12, "#7a4200", "400", "end", font=MONO)
g.text(820, 452, "⋮", 15, MUTED, "700", "middle")
g.text(820, 486, "here they are neighbours", 11.5, MUTED, "400", "middle")
g.text(820, 502, "of the same peak", 11.5, MUTED, "400", "middle")
arrow(g, 900, 330, 940, 330, "#6f7c90", 2.5, 10)

# ------------------------------------------------ stage 2
card(g, 952, y0, 408, 500, "STAGE 2   fine — full resolution, only around those poses",
     accent="#72a8c7", head_fill="#457b9d", title_size=15)
fs = fine[(0, -2)][1]
fh, fw = fs.shape
fcell = 150.0/fw
g.text(1080, y0+72, "position score map at full resolution", 12.5, MUTED, "400", "middle")
heat_rects(g, fs, 1005, y0+82, fcell, COOL, levels=14)
g.rect(1005, y0+82, fw*fcell, fh*fcell, rx=3, fill="none", stroke="#8fb0c5", stroke_width=1.5)
fr, fc = np.unravel_index(fs.argmax(), fs.shape)
g.add(f'<circle cx="{1005+(fc+0.5)*fcell:.1f}" cy="{y0+82+(fr+0.5)*fcell:.1f}" r="6" fill="none" '
      f'stroke="#16455f" stroke-width="2.5"/>')
g.text(1180, y0+112, f"only {fw}×{fh} offsets", 12.5, "#20516e", "700")
g.text(1180, y0+134, "around each coarse peak", 12.5, MUTED)
g.text(1180, y0+164, "the coarse peak is at best", 12.5, MUTED)
g.text(1180, y0+184, "half-pixel accurate, so a small", 12.5, MUTED)
g.text(1180, y0+204, "window is enough", 12.5, MUTED)

# fine angle bars
by0 = y0+256
g.text(1005, by0-12, "and the neighbouring 1° angles", 12.5, MUTED, "600")
offs = [-3, -2, -1, 0, 1, 2, 3]
bw, gap = 34, 12
BASE = 0.76
for k, o in enumerate(offs):
    sc_, _, a = fine[(0, o)]
    x = 1005 + k*(bw+gap)
    hgt = max(sc_ - BASE, 0.002)/(0.84 - BASE)*104
    best = (o == -2)
    g.rect(x, by0+14+104-hgt, bw, hgt, rx=4, fill=GREEN if best else "#a8c6d8")
    g.text(x+bw/2, by0+140, f"{a:.0f}°", 11.5, INK if best else MUTED, "700" if best else "400", "middle")
    g.text(x+bw/2, by0+8+104-hgt, f"{sc_:.3f}", 10, INK if best else MUTED, "700" if best else "400", "middle")
g.add(f'<path d="M1005 {by0+118} H1327" stroke="#c9d1dc" stroke-width="1"/>')
g.text(1327, by0+158, "score axis starts at 0.76", 10.5, "#9aa4b5", "400", "end")
g.rect(1005, by0+170, 330, 62, rx=12, fill="#e9f7ef", stroke="#56a878")
g.add('<circle cx="1036" cy="' + str(by0+201) + '" r="17" fill="#18864b"/>')
g.add(f'<path d="M1027 {by0+201} l7 7 13-15" fill="none" stroke="#ffffff" stroke-width="3.4" '
      f'stroke-linecap="round" stroke-linejoin="round"/>')
g.text(1062, by0+196, "best pose", 12, "#31704b", "700")
g.text(1062, by0+216, "center (232.5, 129.5) · θ 58° · 0.82", 13, "#145c35", "700", font=MONO)

# ------------------------------------------------ cost band
cy0 = 632
card(g, 40, cy0, 1320, 176, None, accent="#dfe4ec")
g.text(76, cy0+34, "What that buys, on this example", 16, INK, "700")
COST = cost_estimate()
ex, c1, c2 = COST["exhaustive"], COST["coarse"], COST["fine"]
rows = [("full-resolution exhaustive search, 1° steps", ex, "#b6bfcd", f"{ex/1e9:.0f} G"),
        ("coarse stage (0.5× image, 3° steps)", c1, ORANGE, f"{c1/1e9:.1f} G"),
        ("fine stage (5 poses × 7 angles × 15×15 offsets)", c2, BLUE, f"{c2/1e9:.2f} G")]
for i, (lbl, v, col, tag) in enumerate(rows):
    yy = cy0 + 56 + i*32
    g.text(76, yy+16, lbl, 13, INK if i else MUTED)
    g.rect(500, yy+3, 640, 18, rx=9, fill="#eef1f6")
    g.rect(500, yy+3, max(640*v/ex, 4), 18, rx=9, fill=col)
    g.text(1160, yy+16, tag, 12.5, MUTED, "700", font=MONO)
g.text(1210, cy0+92, f"≈ {COST['ratio']:.0f}× less", 20, GREEN, "800")
g.text(1210, cy0+114, "multiply–adds", 12.5, MUTED)
g.text(76, cy0+160, "Counted as template pixels × evaluated poses. This is a heuristic: a narrow score "
       "peak that the coarse grid misses cannot be recovered by the fine stage.", 12, "#8b95a5")

write(g, "coarse-to-fine.svg")
