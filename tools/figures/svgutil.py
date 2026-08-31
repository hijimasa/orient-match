import os

import numpy as np

FONT = "Inter,Segoe UI,Helvetica,Arial,sans-serif"
MONO = "SFMono-Regular,Menlo,Consolas,monospace"
INK, MUTED = "#172033", "#5f6b7a"

class Svg:
    def __init__(self, w, h, title, desc, bg="#f7f9fc"):
        self.w, self.h, self.parts = w, h, []
        self.head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
                     f'viewBox="0 0 {w} {h}" role="img" aria-labelledby="ttl dsc">'
                     f'<title id="ttl">{title}</title><desc id="dsc">{desc}</desc>')
        self.defs = []
        self.parts.append(f'<rect width="{w}" height="{h}" fill="{bg}"/>')
    def add(self, s): self.parts.append(s); return self
    def text(self, x, y, s, size=14, fill=INK, weight="400", anchor="start", font=None, style=""):
        f = font or FONT
        st = f' font-style="{style}"' if style else ""
        return self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{f}" font-size="{size}" '
                        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{st}>{s}</text>')
    def rect(self, x, y, w, h, **kw):
        a = " ".join(f'{k.replace("_","-")}="{v}"' for k, v in kw.items())
        return self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" {a}/>')
    def dump(self):
        return self.head + ("<defs>" + "".join(self.defs) + "</defs>" if self.defs else "") + \
               "".join(self.parts) + "</svg>"

def card(svg, x, y, w, h, title=None, accent="#d9deea", head_fill=None, r=16, head_h=44,
         title_size=17, title_fill="#ffffff", fill="#ffffff"):
    svg.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" '
            f'stroke="{accent}" stroke-width="1.5"/>')
    if title is not None:
        hf = head_fill or accent
        svg.add(f'<path d="M{x} {y+head_h} V{y+r} a{r} {r} 0 0 1 {r} -{r} h{w-2*r} '
                f'a{r} {r} 0 0 1 {r} {r} v{head_h-r} Z" fill="{hf}"/>')
        svg.text(x + w / 2, y + head_h - 15, title, size=title_size, weight="700",
                 fill=title_fill, anchor="middle")

def arrow(svg, x1, y1, x2, y2, color="#6f7c90", width=2.4, head=9):
    dx, dy = x2 - x1, y2 - y1
    n = np.hypot(dx, dy)
    if n < 1e-6: return
    ux, uy = dx / n, dy / n
    bx, by = x2 - ux * head, y2 - uy * head
    px, py = -uy, ux
    svg.add(f'<path d="M{x1:.1f} {y1:.1f} L{bx:.1f} {by:.1f}" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round" fill="none"/>')
    svg.add(f'<path d="M{x2:.1f} {y2:.1f} L{bx+px*head*0.42:.1f} {by+py*head*0.42:.1f} '
            f'L{bx-px*head*0.42:.1f} {by-py*head*0.42:.1f} Z" fill="{color}"/>')

def field_arrows(svg, fx, fy, ox, oy, scale, step=6, color="#4c6ef5", length=13.0,
                 thresh=0.22, width=2.0, head=6.5, opacity=None):
    """Draw a sampled orientation field. fx/fy indexed [row, col] in source pixels."""
    h, w = fx.shape
    g = []
    for r in range(step // 2, h, step):
        for c in range(step // 2, w, step):
            vx, vy = float(fx[r, c]), float(fy[r, c])
            m = np.hypot(vx, vy)
            if m < thresh: continue
            cx, cy = ox + (c + 0.5) * scale, oy + (r + 0.5) * scale
            L = length * min(m / 0.9, 1.0)
            g.append((cx - vx / m * L / 2, cy - vy / m * L / 2,
                      cx + vx / m * L / 2, cy + vy / m * L / 2))
    op = f' opacity="{opacity}"' if opacity else ""
    svg.add(f'<g{op}>')
    for x1, y1, x2, y2 in g:
        arrow(svg, x1, y1, x2, y2, color=color, width=width, head=head)
    svg.add('</g>')
    return len(g)

def heat_rects(svg, m, ox, oy, cell, ramp, lo=None, hi=None, levels=12, opacity=1.0):
    """Row-wise run-length encoded heatmap; m is 2D."""
    lo = float(m.min()) if lo is None else lo
    hi = float(m.max()) if hi is None else hi
    q = np.clip((m - lo) / max(hi - lo, 1e-9), 0, 1)
    q = np.round(q * (levels - 1)).astype(int)
    svg.add(f'<g opacity="{opacity}" shape-rendering="crispEdges">')
    for r in range(q.shape[0]):
        c = 0
        while c < q.shape[1]:
            v = q[r, c]; c2 = c
            while c2 + 1 < q.shape[1] and q[r, c2 + 1] == v: c2 += 1
            svg.rect(ox + c * cell, oy + r * cell, (c2 - c + 1) * cell + 0.4, cell + 0.4,
                     fill=ramp(v / (levels - 1)))
            c = c2 + 1
    svg.add('</g>')

def ramp_maker(stops):
    """stops: list of (t, (r,g,b))."""
    def f(t):
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]; t1, c1 = stops[i + 1]
            if t <= t1 or i == len(stops) - 2:
                u = 0 if t1 == t0 else (t - t0) / (t1 - t0)
                u = min(max(u, 0), 1)
                return "#%02x%02x%02x" % tuple(int(round(c0[k] + (c1[k] - c0[k]) * u)) for k in range(3))
    return f

HEAT = ramp_maker([(0.0, (241, 244, 249)), (0.45, (253, 230, 199)),
                   (0.75, (244, 162, 97)), (1.0, (214, 93, 12))])
COOL = ramp_maker([(0.0, (240, 245, 250)), (0.5, (198, 226, 240)), (1.0, (49, 110, 148))])


def figure_path(name):
    """figs/<name> at the repository root, whatever the working directory is."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "figs", name)


def write(svg, name):
    path = figure_path(name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(svg.dump())
    print(f"wrote {path} ({len(svg.dump())/1024:.0f} KB)")
