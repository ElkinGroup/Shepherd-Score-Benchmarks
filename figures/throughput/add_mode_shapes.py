"""Give every mode marker in the hand-edited fig2_alt.svg a distinct shape (panels C, D and the key)
and set the panel-D stem widths, editing marker paths in place so positions and text survive.
Idempotent. Run: python add_mode_shapes.py
"""
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = {
    '#5554ae': ('vol', 'circle', 'st49'),
    '#7a2b73': ('vol_esp', 'diamond', 'st31'),
    '#d35d7f': ('vol_lipo', 'triangle-up', 'st25'),
    '#a52f2f': ('vol_color', 'square', 'st22'),
    '#007d00': ('vol_and_surf_esp', 'hexagon', 'st62'),
    '#006d72': ('surf_esp', 'triangle-down', 'st52'),
    '#1a619f': ('surf', 'plus', 'st14'),
    '#ff8b00': ('pharm', 'pentagon', 'st50'),
}


def shape_path(shape, radius, cx=0, cy=0):
    def point(x, y):
        return f'{cx + x * radius:.5f} {cy + y * radius:.5f}'

    if shape == 'circle':
        return (f'M {point(0, 1)} A {radius} {radius} 0 1 1 {point(0, -1)} '
                f'A {radius} {radius} 0 1 1 {point(0, 1)} Z')
    if shape == 'square':
        pts = [(-.9, -.9), (.9, -.9), (.9, .9), (-.9, .9)]
    elif shape == 'diamond':
        pts = [(0, -1.25), (1.1, 0), (0, 1.25), (-1.1, 0)]
    elif shape.startswith('triangle'):
        flip = 1 if shape == 'triangle-up' else -1
        pts = [(0, -1.15 * flip), (1.05, .85 * flip), (-1.05, .85 * flip)]
    elif shape == 'plus':
        pts = [(-.38, -1.05), (.38, -1.05), (.38, -.38), (1.05, -.38),
               (1.05, .38), (.38, .38), (.38, 1.05), (-.38, 1.05),
               (-.38, .38), (-1.05, .38), (-1.05, -.38), (-.38, -.38)]
    else:
        n = 6 if shape == 'hexagon' else 5
        angle = 0 if n == 6 else -math.pi / 2
        pts = [(1.08 * math.cos(angle + i * 2 * math.pi / n),
                1.08 * math.sin(angle + i * 2 * math.pi / n)) for i in range(n)]
    return 'M ' + ' L '.join(point(x, y) for x, y in pts) + ' Z'


def update(svg):
    if 'data-speed-spacing="v1"' in svg:
        # A layout pass that already carries these shapes has translated coordinates; leave it alone.
        return svg
    counts = Counter()
    seen = {'C': set(), 'D': set(), 'key': set()}

    def replace(match):
        raw = match.group()
        node = ET.fromstring(raw)
        d = node.get('d', '')
        transform = node.get('transform', '')
        region = None
        cx = cy = 0
        # The key uses Illustrator classes and absolute path coordinates.
        for color, (mode, shape, css) in MODES.items():
            if node.get('class') == css and 'translate(500 637)' in transform:
                region, radius = 'key', 4.5
                coords = re.match(r'M\s*([\d.]+)[,\s]+([\d.]+)', d)
                cx, bottom = map(float, coords.groups())
                cy = bottom - radius
                break
        else:
            style = dict(re.findall(r'([\w-]+)\s*:\s*([^;]+)', node.get('style', '')))
            color = style.get('fill')
            if color not in MODES:
                color = style.get('stroke')
            if color not in MODES:
                return raw
            mode, shape, _ = MODES[color]
            if node.get('data-panel') == 'B':
                counts['B'] += 1
                return raw
            swatch = re.fullmatch(r'M ([\d.]+) ([\d.]+)\s+L ([\d.]+) ([\d.]+)\s+L ([\d.]+) ([\d.]+)\s+L ([\d.]+) ([\d.]+)\s+z\s*', d)
            if swatch and not transform:
                coords = list(map(float, swatch.groups()))
                xs, ys = coords[::2], coords[1::2]
                if min(xs) > 780 and 165 < min(ys) < max(ys) < 173:
                    counts['B'] += 1
                    path = shape_path(shape, 3.25, sum(xs)/4, sum(ys)/4)
                    raw = re.sub(r'\bd="[^"]*"', f'd="{path}"', raw, count=1)
                    return raw.replace('<path ', f'<path data-panel="B" data-mode="{mode}" data-mode-shape="{shape}" ', 1)
            stem = re.fullmatch(r'M ([\d.]+) ([\d.]+)\s+L ([\d.]+) ([\d.]+)\s*', d)
            if stem and not transform and style.get('fill') == 'none':
                x1, y1, x2, y2 = map(float, stem.groups())
                if x1 == x2 and 360 <= x1 <= 900 and min(y1, y2) >= 339:
                    counts['D stems'] += 1
                    return re.sub(r'stroke-width: [\d.]+', 'stroke-width: 2.2', raw)
            location = re.fullmatch(r'translate\(([\d.]+) ([\d.]+)\)', transform)
            if not location or not d.strip().lower().endswith('z'):
                return raw
            x, y = map(float, location.groups())
            if 40 <= x <= 314 and 315 <= y <= 579:
                region, radius = 'C', 3.6
            elif 360 <= x <= 900 and 339 <= y <= 590:
                region = 'D'
                original = re.match(r'M 0 (5\.85|7\.15)\s+C', d)
                if original is None and 'data-mode-shape' not in node.attrib:
                    return raw
                hollow = style.get('fill') == '#ffffff'
                radius = (5.85 if hollow else 7.15) * 1.1
                width = 1.32 if hollow else .66
                raw = re.sub(r'stroke-width: [\d.]+', f'stroke-width: {width}', raw)
            else:
                return raw

        counts[region] += 1
        seen[region].add(mode)
        if region == 'key' and 'data-mode-shape' in node.attrib:
            return raw  # Idempotent, including the key's absolute coordinates.
        path = shape_path(shape, radius, cx, cy)
        raw = re.sub(r'\bd="[^"]*"', f'd="{path}"', raw, count=1)
        if 'data-mode-shape' in node.attrib:
            return raw
        return raw.replace('<path ', f'<path data-mode="{mode}" data-mode-shape="{shape}" ', 1)

    output = re.sub(r'<path\b[^>]*?/>', replace, svg)
    def lower_caption(match):
        raw = match.group()
        node = ET.fromstring(raw)
        if ''.join(node.itertext()).strip() != 'alignment mode':
            return raw
        counts['D captions'] += 1
        if node.get('data-caption-offset') == '5':
            return raw
        y = float(node.get('y')) + 5
        raw = re.sub(r'\by="[^"]*"', f'y="{y:.2f}"', raw, count=1)
        raw = re.sub(r'transform="[^"]*"', f'transform="rotate(-0 {node.get("x")} {y:.2f})"', raw, count=1)
        return raw.replace('<text ', '<text data-caption-offset="5" ', 1)

    output = re.sub(r'<text\b[^>]*>.*?</text>', lower_caption, output, flags=re.S)
    output = output.replace(
        'other tool, coloured by the ShEPhERD-score mode it is compared with, numbered in the list at right',
        'external tool; shape and color match the compared mode; numbers identify tools at right')
    expected = {m[0] for m in MODES.values()}
    assert all(modes == expected for modes in seen.values()), seen
    assert counts['key'] == 8 and counts['D'] >= 24, counts
    assert counts['D stems'] == 24, counts
    assert counts['B'] == 8 and counts['D captions'] == 3, counts
    before, after = ET.fromstring(svg), ET.fromstring(output)
    assert len(list(before.iter())) == len(list(after.iter()))
    assert not any(e.tag.split('}')[-1] in ('image', 'use', 'clipPath') for e in after.iter())
    print(f'Mode shapes verified: {dict(counts)}; all eight modes in C, D and key.')
    return output


if __name__ == '__main__':
    target = HERE / 'fig2_alt.svg'
    source = target.read_text(encoding='utf-8')
    result = update(source)
    assert update(result) == result
    target.write_text(result, encoding='utf-8')
