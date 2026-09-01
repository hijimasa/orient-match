import numpy as np
import cv2

from core import image_planes, orientation_field, rotate_canvas
from data import coarse_to_fine, scene
from shapes import part_group
from svgutil import *

D = coarse_to_fine()
maps, angles, peaks, fine = D["maps"], D["angles"], D["peaks"], D["fine"]
scanned, order, bases = D["scanned"], D["order"], D["bases"]
LEAD = int(bases[0])
img, tmpl, ANG, _, (cx, cy, mask, size) = scene()
tfx, tfy = orientation_field(tmpl)
ix, iy, en = image_planes(img)
rx, ry, _ = rotate_canvas(cx, cy, mask, size, ANG)

TEAL, ORANGE, BLUE, GREEN = "#00a896", "#e08a3c", "#457b9d", "#18864b"
TMPL_C, IMG_C = "#5b52d6", "#8b95a5"

W, H = 1380, 690
g = Svg(W, H, "OrientMatch pipeline",
        "Building a Matcher precomputes the template orientation field, its square rotation canvas "
        "and a bank of coarse rotated copies. Each frame is converted to an orientation field, "
        "scanned globally at a sparse set of angles, and the best separated places are then refined "
        "locally - first over the skipped angles, then at full resolution.")
g.text(W/2, 48, "OrientMatch pipeline", 28, INK, "700", "middle")
g.text(W/2, 76, "what is paid for once, and what happens on every frame", 15, MUTED, "400", "middle")

def lane(x, y, w, h, label, sub, col):
    g.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{col}"/>')
    g.text(x+w/2, y+h/2-6, label, 13, "#ffffff", "800", "middle")
    g.text(x+w/2, y+h/2+14, sub, 11.5, "#ffffffcc", "400", "middle")

def box(x, y, w, h, n, title, accent):
    g.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="#ffffff" '
          f'stroke="{accent}" stroke-width="1.6"/>')
    g.add(f'<circle cx="{x+24}" cy="{y+24}" r="13" fill="{accent}"/>')
    g.text(x+24, y+29, str(n), 13, "#ffffff", "800", "middle")
    g.text(x+46, y+29, title, 14, INK, "700")

def link(x1, y, x2):
    arrow(g, x1, y, x2, y, "#8b95a5", 2.4, 10)

BW, BH = 246, 196
XS = [200, 486, 772, 1058]
# ------------------------------------------------------------------ lane A
ya = 106
lane(40, ya, 140, BH, "BUILT ONCE", "Matcher(template)", "#7a6cf0")
for i, x in enumerate(XS):
    if i: link(x-34, ya+BH/2, x-6)

box(XS[0], ya, BW, BH, 1, "template", "#7a6cf0")
g.rect(XS[0]+56, ya+48, 134, 134, rx=6, fill="#eef1f6", stroke="#c7cedb")
g.add(part_group(XS[0]+123, ya+115, 134/64.0))

box(XS[1], ya, BW, BH, 2, "orientation field", TEAL)
g.rect(XS[1]+56, ya+48, 134, 134, rx=6, fill="#ffffff", stroke="#b7e4de")
g.add(f'<g opacity="0.14">{part_group(XS[1]+123, ya+115, 134/64.0)}</g>')
field_arrows(g, tfx, tfy, XS[1]+56, ya+48, 134/64.0, step=5, color=TMPL_C, length=10,
             thresh=0.25, width=1.5, head=4.6)

box(XS[2], ya, BW, BH, 3, "square canvas", "#9a8cf5")
g.rect(XS[2]+56, ya+48, 134, 134, rx=6, fill="#ffffff", stroke="#cfd6e2")
sc = 134.0/size
g.add(f'<g opacity="0.14">{part_group(XS[2]+123, ya+115, 134/float(size))}</g>')
field_arrows(g, cx, cy, XS[2]+56, ya+48, sc, step=6, color=TMPL_C, length=10,
             thresh=0.2, width=1.5, head=4.6)
g.add(f'<circle cx="{XS[2]+123}" cy="{ya+115}" r="3" fill="#ffffff" stroke="#4a5468" stroke-width="1.6"/>')
g.text(XS[2]+123, ya+192, "windowed · masked · padded to hypot", 10.5, MUTED, "400", "middle")

box(XS[3], ya, BW, BH, 4, "coarse rotated bank", ORANGE)
for k, a in enumerate((0.0, 24.0, 48.0)):
    ox, oy = XS[3]+40+k*22, ya+56+k*16
    g.rect(ox, oy, 96, 96, rx=6, fill="#fffaf3", stroke="#efc79a")
    g.add(part_group(ox+48, oy+48, 96/float(size), a, fill_dark="#dda15e", fill_light="#f6e2c8"))
g.text(XS[3]+123, ya+192, f"{len(angles)} angles at 0.5× — reused for every frame", 10.5, MUTED, "400", "middle")

# ------------------------------------------------------------------ lane B
yb = 386
lane(40, yb, 140, BH, "EVERY FRAME", "match(image)", BLUE)
for i, x in enumerate(XS):
    if i: link(x-34, yb+BH/2, x-6)

box(XS[0], yb, BW, BH, 5, "search image", "#7b879b")
S = 195/360.0
g.defs.append('<linearGradient id="sbg" x1="0" y1="0" x2="1" y2="1">'
              '<stop offset="0" stop-color="#8d97a8"/><stop offset="1" stop-color="#c3cad6"/></linearGradient>')
g.defs.append(f'<clipPath id="sclip"><rect x="{XS[0]+26}" y="{yb+46}" width="195" height="130" rx="5"/></clipPath>')
g.rect(XS[0]+26, yb+46, 195, 130, rx=5, fill="url(#sbg)", stroke="#aeb7c6")
g.add('<g clip-path="url(#sclip)">')
g.add(part_group(XS[0]+26+95*S, yb+46+78*S, S, -30.0, fill_dark="#5c6577", fill_light="#aab2c0"))
g.add(part_group(XS[0]+26+232*S, yb+46+130*S, S, ANG, fill_dark="#7b8494", fill_light="#b9c0cc"))
g.add(f'<circle cx="{XS[0]+26+300*S:.1f}" cy="{yb+46+46*S:.1f}" r="{17*S:.1f}" fill="#cdd4de"/></g>')
g.text(XS[0]+123, yb+190, "larger grayscale frame", 10.5, MUTED, "400", "middle")

box(XS[1], yb, BW, BH, 6, "orientation field", TEAL)
g.rect(XS[1]+26, yb+46, 195, 130, rx=5, fill="#ffffff", stroke="#b7e4de")
field_arrows(g, ix, iy, XS[1]+26, yb+46, 195/360.0, step=22, color="#cfd5df", length=11,
             thresh=0.16, width=1.4, head=4.6)
field_arrows(g, ix, iy, XS[1]+26, yb+46, 195/360.0, step=22, color=IMG_C, length=13,
             thresh=0.45, width=1.8, head=5.4)
g.text(XS[1]+123, yb+190, "two float planes + local energy", 10.5, MUTED, "400", "middle")

box(XS[2], yb, BW, BH, 7, "global scan", ORANGE)
cmap = cv2.resize(maps[angles[LEAD]], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
mh, mw = cmap.shape
c2 = 200.0/mw
heat_rects(g, cmap, XS[2]+23, yb+52, c2, HEAT, lo=0.16, hi=float(peaks[:,0].max()), levels=14)
g.rect(XS[2]+23, yb+52, mw*c2, mh*c2, rx=4, fill="none", stroke="#e6c39a")
r_, c_ = peaks[LEAD,1]*0.5, peaks[LEAD,2]*0.5
g.add(f'<circle cx="{XS[2]+23+(c_+0.5)*c2:.1f}" cy="{yb+52+(r_+0.5)*c2:.1f}" r="6" fill="none" '
      f'stroke="#7a3d00" stroke-width="2"/>')
g.text(XS[2]+123, yb+190, f"all positions × {len(scanned)} of {len(angles)} angles, 0.5× image", 10.5, MUTED, "400", "middle")

box(XS[3], yb, BW, BH, 8, "refine candidates", BLUE)
fs = fine[(0,-2)][1]
fc = 108.0/fs.shape[1]
heat_rects(g, fs, XS[3]+22, yb+58, fc, COOL, levels=14)
g.rect(XS[3]+22, yb+58, 108, 108, rx=4, fill="none", stroke="#9fb4c4")
g.text(XS[3]+140, yb+80, "K = 5 places", 11.5, "#20516e", "700")
g.text(XS[3]+140, yb+100, "skipped angles", 11.5, MUTED)
g.text(XS[3]+140, yb+120, "±7 px, ±3° at 1°", 11.5, MUTED)
g.text(XS[3]+140, yb+140, "full resolution", 11.5, MUTED)
g.text(XS[3]+123, yb+190, "highest fine score wins", 10.5, MUTED, "400", "middle")

# top-K chip on the arrow
g.rect(XS[2]+BW-2, yb+BH/2-34, 44, 22, rx=8, fill="#fff3e6", stroke=ORANGE)
g.text(XS[2]+BW+20, yb+BH/2-19, "top K", 11, "#7a4200", "700", "middle")

# result
arrow(g, XS[3]+BW/2, yb+BH+6, XS[3]+BW/2, yb+BH+34, "#8b95a5", 2.4, 10)
g.rect(XS[3]-120, yb+BH+38, 386, 52, rx=14, fill="#e9f7ef", stroke="#56a878")
g.add(f'<circle cx="{XS[3]-92}" cy="{yb+BH+64}" r="15" fill="{GREEN}"/>')
g.add(f'<path d="M{XS[3]-100} {yb+BH+64} l6 6 12-13" fill="none" stroke="#ffffff" stroke-width="3" '
      f'stroke-linecap="round" stroke-linejoin="round"/>')
g.text(XS[3]-70, yb+BH+69, "MatchResult: center (x, y) · angle θ · score", 14, "#145c35", "700")

# cross-lane feeds
gapy0, gapy1 = ya+BH+4, yb-6
g.add(f'<path d="M{XS[3]+BW/2} {gapy0} L{XS[2]+BW/2} {gapy1}" fill="none" stroke="{ORANGE}" '
      f'stroke-width="2.2" stroke-dasharray="7 5"/>')
arrow(g, XS[2]+BW/2+8, gapy1-11, XS[2]+BW/2, gapy1, ORANGE, 2.2, 9)
g.add(f'<path d="M{XS[2]+BW/2} {gapy0} L{XS[3]+BW/2} {gapy1}" fill="none" stroke="{BLUE}" '
      f'stroke-width="2.2" stroke-dasharray="7 5"/>')
arrow(g, XS[3]+BW/2-8, gapy1-11, XS[3]+BW/2, gapy1, BLUE, 2.2, 9)
lx, ly = 210, ya+BH+30
g.add(f'<path d="M{lx} {ly-4} h26" stroke="{ORANGE}" stroke-width="2.2" stroke-dasharray="7 5"/>')
g.text(lx+34, ly, "the precomputed 0.5× bank is what both 0.5× stages slide", 12, "#a2621f", "600")
g.add(f'<path d="M{lx} {ly+22} h26" stroke="{BLUE}" stroke-width="2.2" stroke-dasharray="7 5"/>')
g.text(lx+34, ly+26, "the full-resolution canvas is rotated on demand, one fine angle at a time",
       12, "#2b6285", "600")

g.text(40, H-18, "Each stage transforms its image window once, then correlates every angle "
       "against it. Fixed scale, one best match per frame.", 12, "#8b95a5")
write(g, "pipeline.svg")
