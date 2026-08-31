import numpy as np
import cv2

from core import image_planes, orientation_field, rotate_canvas, score_map
from data import scene
from shapes import part_group
from svgutil import *

TMPL_C, IMG_C = "#5b52d6", "#77839a"
ANG = 58.0

img, tmpl, _, _, (cx, cy, mask, size) = scene()
ix, iy, en = image_planes(img)
tfx, tfy = orientation_field(tmpl)
rx, ry, sup = rotate_canvas(cx, cy, mask, size, ANG)
nx_, ny_, sup2 = rotate_canvas(cx, cy, mask, size, ANG, rotate_values=False)
s_ok = score_map(ix, iy, en, rx, ry, sup)
best = np.unravel_index(s_ok.argmax(), s_ok.shape)
py, px = int(best[0]), int(best[1])
s_norot = score_map(ix, iy, en, nx_, ny_, sup2)[py, px]
w30x, w30y, w30s = rotate_canvas(cx, cy, mask, size, 30.0)
s_w30 = score_map(ix, iy, en, w30x, w30y, w30s)[py, px]
s_bg = float(score_map(ix, iy, en, rx, ry, sup)[10, 10])
S_OK = float(s_ok[py, px])
print("pose (%d, %d)  correct %.3f  values not rotated %.3f  wrong angle %.3f  background %.3f"
      % (px, py, S_OK, s_norot, s_w30, s_bg))

W, H = 1300, 940
g = Svg(W, H, "How OrientMatch scores a pose",
        "Each pixel is turned into a unit gradient-direction vector. Rotating the template rotates "
        "both the sample positions and the vector values. The score is the energy-normalized sum of "
        "dot products between template and image direction vectors.")
g.text(W/2, 50, "What OrientMatch actually compares", 28, INK, "700", "middle")
g.text(W/2, 80, "gradient direction at every pixel — not brightness, not edges, not keypoints",
       15, MUTED, "400", "middle")

# ---------------------------------------------------------------- card 1
y0 = 104
card(g, 40, y0, 596, 392, "1  Every pixel becomes a direction vector", accent="#00a896", head_fill="#00a896")
g.text(165, y0+72, "template image", 13, MUTED, "600", "middle")
g.text(451, y0+72, "orientation field  z", 13, "#087f72", "600", "middle")
g.rect(70, y0+84, 190, 190, rx=8, fill="#eef1f6", stroke="#c7cedb")
g.add(part_group(165, y0+179, 190/64.0))
g.rect(356, y0+84, 190, 190, rx=8, fill="#ffffff", stroke="#b7e4de")
g.add(f'<g opacity="0.16">{part_group(451, y0+179, 190/64.0)}</g>')
field_arrows(g, tfx, tfy, 356, y0+84, 190/64.0, step=4, color=TMPL_C, length=13.5,
             thresh=0.25, width=1.9, head=5.6)
g.add(f'<path d="M275 {y0+179} H341" stroke="#9aa4b5" stroke-width="2.5" fill="none"/>')
arrow(g, 275, y0+179, 344, y0+179, "#9aa4b5", 2.5, 10)
g.rect(70, y0+288, 476, 44, rx=9, fill="#effaf8")
g.text(308, y0+317, "z = ∇I / (‖∇I‖ + ε)", 19, "#087f72", "400", "middle",
       font="Georgia,serif", style="italic")
g.text(70, y0+352, "• direction only — unchanged if the part is dark on light or light on dark", 13, MUTED)
g.text(70, y0+371, "• ε gates weak, noisy gradients toward zero: flat areas contribute nothing", 13, MUTED)

# ---------------------------------------------------------------- card 2
card(g, 664, y0, 596, 392, "2  A rotated template rotates its vectors too",
     accent="#f4a261", head_fill="#f4a261", title_fill="#3d2608")
g.text(789, y0+72, "θ = 0°", 13, MUTED, "600", "middle")
g.text(1135, y0+72, "θ = 58°", 13, "#8a4a04", "600", "middle")
sc = 190/float(size)
for cxp, fxx, fyy, ang_ in ((694, cx, cy, 0.0), (1040, rx, ry, ANG)):
    g.rect(cxp, y0+84, 190, 190, rx=8, fill="#ffffff", stroke="#f0cba2")
    g.add(f'<g opacity="0.2">{part_group(cxp+95, y0+179, 190/float(size), ang_)}</g>')
    field_arrows(g, fxx, fyy, cxp, y0+84, sc, step=5, color=TMPL_C, length=13.5,
                 thresh=0.2, width=1.9, head=5.6)

# highlight two probe vectors before / after the rotation
M = cv2.getRotationMatrix2D(((size-1)/2.0, (size-1)/2.0), ANG, 1.0)
r_ = np.deg2rad(-ANG); R = np.array([[np.cos(r_), -np.sin(r_)], [np.sin(r_), np.cos(r_)]])
for (pr, pc) in ((30, 46), (58, 30)):
    v = np.array([cx[pr, pc], cy[pr, pc]], float)
    q = M @ np.array([pc, pr, 1.0])
    vr = R @ v
    for base, (qc, qr), vec in ((694, (pc, pr), v), (1040, (q[0], q[1]), vr)):
        ax, ay = base + (qc+0.5)*sc, y0+84 + (qr+0.5)*sc
        n = np.hypot(*vec)
        g.add(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="12" fill="#fde8cf" stroke="#d1873c" stroke-width="1.6"/>')
        arrow(g, ax - vec[0]/n*13, ay - vec[1]/n*13, ax + vec[0]/n*13, ay + vec[1]/n*13,
              "#c25e0a", 3.0, 9)
g.add(f'<path d="M898 {y0+150} a54 54 0 0 1 118 0" stroke="#d1873c" stroke-width="2.6" fill="none"/>')
arrow(g, 1010, y0+126, 1020, y0+150, "#d1873c", 2.6, 9)
g.text(957, y0+120, "rotate by θ", 13, "#8a4a04", "700", "middle")
g.rect(694, y0+288, 496, 78, rx=10, fill="#fff6ec", stroke="#f0cba2")
g.text(714, y0+313, "positions rotate — and every vector value rotates with them.", 13.5, "#6b3a00", "700")
g.text(714, y0+336, f"rotating positions only would scale every dot product by cos θ:", 13, MUTED)
g.text(714, y0+358, f"score {S_OK:.2f}  →  {s_norot:.2f}   at θ = 58°", 13, "#b34a10", "700")

# ---------------------------------------------------------------- card 3
y1 = 524
card(g, 40, y1, 1220, 372, "3  The score of one pose = normalized sum of dot products",
     accent="#457b9d", head_fill="#457b9d")

# --- (a) scene
S = 330/360.0
g.text(76, y1+72, "search image with one candidate pose", 13, MUTED, "600")
g.defs.append('<linearGradient id="scenebg" x1="0" y1="0" x2="1" y2="1">'
              '<stop offset="0" stop-color="#8d97a8"/><stop offset="1" stop-color="#c3cad6"/></linearGradient>')
g.defs.append(f'<clipPath id="sceneclip"><rect x="76" y="{y1+84}" width="330" height="220" rx="6"/></clipPath>')
g.rect(76, y1+84, 330, 220, rx=6, fill="url(#scenebg)", stroke="#aeb7c6")
g.add('<g clip-path="url(#sceneclip)">')
g.add(part_group(76+95*S, y1+84+78*S, S, -30.0, fill_dark="#5c6577", fill_light="#aab2c0"))
g.add(part_group(76+232*S, y1+84+130*S, S, ANG, fill_dark="#7b8494", fill_light="#b9c0cc"))
g.add(f'<circle cx="{76+300*S:.1f}" cy="{y1+84+46*S:.1f}" r="{17*S:.1f}" fill="#cdd4de"/>')
g.add('</g>')
tcx, tcy, box = 76+232*S, y1+84+130*S, 92*S
g.add(f'<rect x="{tcx-box/2:.1f}" y="{tcy-box/2:.1f}" width="{box:.1f}" height="{box:.1f}" '
      f'fill="none" stroke="#457b9d" stroke-width="2"/>')
g.add(f'<g transform="translate({tcx:.1f} {tcy:.1f}) rotate({-ANG})">'
      f'<rect x="{-32*S:.1f}" y="{-32*S:.1f}" width="{64*S:.1f}" height="{64*S:.1f}" fill="none" '
      f'stroke="#f4a261" stroke-width="2.5" stroke-dasharray="6 4"/></g>')
g.add(f'<circle cx="{tcx:.1f}" cy="{tcy:.1f}" r="3.4" fill="#ffffff" stroke="#20516e" stroke-width="2"/>')
g.add(f'<rect x="82" y="{y1+270}" width="150" height="28" rx="6" fill="#ffffffdd"/>')
g.add(f'<rect x="90" y="{y1+276}" width="12" height="9" fill="none" stroke="#f4a261" stroke-width="2" stroke-dasharray="3 2"/>')
g.text(108, y1+285, "template at θ", 11.5, "#3d4756")
g.add(f'<rect x="90" y="{y1+288}" width="12" height="9" fill="none" stroke="#457b9d" stroke-width="2"/>')
g.text(108, y1+297, "square rotation canvas", 11.5, "#3d4756")
g.text(241, y1+326, "the same test is repeated for every (x, y, θ)", 12.5, MUTED, "400", "middle")

# --- (b) overlay
ozx, ozy, osz = 450, y1+84, 220
g.text(ozx, y1+72, "the two fields, superposed", 13, MUTED, "600")
g.rect(ozx, ozy, osz, osz, rx=8, fill="#ffffff", stroke="#9fb4c4")
sc2 = osz/float(size)
win = (slice(py, py+size), slice(px, px+size))
field_arrows(g, ix[win], iy[win], ozx, ozy, sc2, step=7, color="#b9c1cd", length=19,
             thresh=0.20, width=3.4, head=9)
field_arrows(g, rx, ry, ozx, ozy, sc2, step=7, color=TMPL_C, length=15,
             thresh=0.25, width=2.3, head=7)
arrow(g, ozx+2, y1+330, ozx+26, y1+330, "#b9c1cd", 3.4, 9)
g.text(ozx+30, y1+334, "image", 12, MUTED)
arrow(g, ozx+80, y1+330, ozx+104, y1+330, TMPL_C, 1.9, 6)
g.text(ozx+108, y1+334, "template", 12, MUTED)
g.text(ozx+osz/2, y1+356, "at the true pose the pairs point the same way", 12.5, MUTED, "400", "middle")

# --- (c) formula + bars
fx0 = 726
g.rect(fx0, y1+80, 518, 82, rx=10, fill="#eef5fa", stroke="#bfd6e4")
g.text(fx0+259, y1+114, "score  =  Σ (tₓ iₓ + tᵧ iᵧ) ⁄ √(Eₜ Eᵢ)", 20, "#20516e", "400",
       "middle", font="Georgia,serif", style="italic")
g.text(fx0+259, y1+140, "every pixel pair contributes t · i = cos Δ, so the sum measures "
       "direction agreement", 12.5, MUTED, "400", "middle")
bars = [("correct pose  θ = 58°", S_OK, "#18864b"),
        ("template vectors not rotated", s_norot, "#c9752f"),
        ("wrong angle  θ = 30°", s_w30, "#f4a261"),
        ("empty background", max(s_bg, 0.0), "#a8b1bf")]
by = y1+186
g.text(fx0, by, "same window, different hypotheses", 12.5, MUTED, "600")
for i, (lbl, v, col) in enumerate(bars):
    yy = by + 14 + i*32
    g.text(fx0, yy+18, lbl, 12.5, INK)
    g.rect(fx0+238, yy+5, 220, 17, rx=8, fill="#e5e9f0")
    g.rect(fx0+238, yy+5, max(220*v, 4), 17, rx=8, fill=col)
    g.text(fx0+470, yy+18, f"{v:.2f}", 12.5, INK, "700", font=MONO)
g.text(fx0, by+156, "scores live in [-1, 1]; the search keeps the largest one",
       12.5, MUTED)

g.text(40, 922, "Synthetic example — the arrows and every number above are computed with the "
       "library's own formulas.", 12, "#8b95a5")

write(g, "principle.svg")
