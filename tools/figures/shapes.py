"""Vector drawing of the synthetic parts used in the figures (matches core.make_template)."""
def part_paths(fill_dark="#4a5468", fill_light="#c9d0dc", stroke="#2f3846"):
    # template pixel coords, 64x64
    return (f'<rect x="10" y="18" width="44" height="26" rx="3" fill="{fill_dark}"/>'
            f'<rect x="36" y="6" width="16" height="16" rx="2.5" fill="{fill_dark}"/>'
            f'<circle cx="22" cy="31" r="7" fill="{fill_light}"/>')

def part_group(cx, cy, scale, angle_deg=0.0, **kw):
    """Place the 64x64 part centred at (cx, cy), rotated by angle_deg (CCW, y-down canvas)."""
    return (f'<g transform="translate({cx:.1f} {cy:.1f}) rotate({-angle_deg:.2f}) '
            f'scale({scale:.4f}) translate(-32 -32)">{part_paths(**kw)}</g>')
