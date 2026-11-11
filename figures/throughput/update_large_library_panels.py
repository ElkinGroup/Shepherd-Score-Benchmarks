"""Redraw panel C (scaling) and the two GPU comparison charts of panel D inside the hand-edited
fig2_alt.svg from paper_results; every other element is preserved byte for byte. Also writes
large_library_plot_sources.json listing the cells drawn. Run: python update_large_library_panels.py
"""
import json
import math
import re
from pathlib import Path
import xml.etree.ElementTree as ET
from add_mode_shapes import MODES, shape_path
import paper_results as results

HERE = Path(__file__).resolve().parent
DATA = results.DATA
PALETTE = {mode: (color, shape) for color, (mode, shape, _) in MODES.items()}
ORDER = list(PALETTE)


def cells(name, size=None, tool='fss'):
    doc = json.loads((DATA / name).read_text())
    out = []
    for key, c in doc['cells'].items():
        if c.get('tool') != tool or c.get('workload') != 'screen' or 'aligns_per_sec' not in c:
            continue
        if size is not None and c.get('size') != size:
            continue
        assert c.get('valid') and not c.get('effort_truncated'), (name, key)
        assert c['env']['device'] == 'NVIDIA L40S' or c['backend'] == 'numba'
        if c.get('spread_flagged'):
            assert c['size'] < 1_000_000, (name, key)
            print('Retaining existing small-N spread flag:', name, key)
        out.append(dict(c, source=name, cell_key=key))
    return out


def line(x1, y1, x2, y2, color='#1b1b1b', width=.8):
    return f'<path d="M{x1:.5f} {y1:.5f} L{x2:.5f} {y2:.5f}" fill="none" stroke="{color}" stroke-width="{width}"/>'


def text(x, y, value, size=12.5, color='#1b1b1b', anchor='middle', rotate=None, bold=False):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ''
    weight = ' font-weight="700"' if bold else ''
    return f'<text x="{x:.5f}" y="{y:.5f}" font-family="Arial" font-size="{size}" fill="{color}" text-anchor="{anchor}"{transform}{weight}>{value}</text>'


def marker(m, x, y, radius, hollow=False, opacity=1, width=.66):
    color, shape = PALETTE[m]
    fill, stroke = ('white', color) if hollow else (color, 'white')
    return f'<path data-mode="{m}" data-mode-shape="{shape}" d="{shape_path(shape, radius, x, y)}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"/>'


def scaling():
    gpu = results.gpu_ladder()
    cpu = results.cpu_ladder()
    # Match the saved axis rectangle and y scale; extend only the x range.
    max_power = math.ceil(math.log10(max(c['size'] for c in gpu + cpu)))
    x = lambda n: 41.51 + (math.log10(n)-1)*272.1/(max_power - 1 + .25)
    y = lambda rate: 592.26 - math.log10(rate)*41.862414
    art = []
    for power in range(1,max_power+1):
        xx = x(10**power)
        art.append(line(xx, 592.26, xx, 595.76))
        art.append(text(xx, 611.76, f'10<tspan baseline-shift="super" font-size="9">{power}</tspan>'))
        for minor in range(2,10):
            xm = x(minor*10**power)
            if xm < 313.61:
                art.append(line(xm, 592.26, xm, 594.26, width=.6))
    for power in range(7):
        yy = y(10**power)
        art.append(line(41.51, yy, 38.01, yy))
        art.append(text(34.51, yy+4, f'10<tspan baseline-shift="super" font-size="9">{power}</tspan>', anchor='end'))
        for minor in range(2,10):
            ym = y(minor*10**power)
            if ym > 329.36:
                art.append(line(41.51, ym, 39.51, ym, width=.6))
    for mode in ORDER:
        for collection, dashed in ((gpu, False), (cpu, True)):
            points = sorted((c for c in collection if c['mode']==mode), key=lambda c:c['size'])
            assert len({c['size'] for c in points}) == len(points)
            coords = [(x(c['size']), y(c['aligns_per_sec'])) for c in points]
            color, _ = PALETTE[mode]
            d = 'M' + ' L'.join(f'{a:.5f} {b:.5f}' for a,b in coords)
            extra = ' stroke-dasharray="5.55 2.4" opacity=".85"' if dashed else ''
            art.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{1.5 if dashed else 2.1}"{extra}/>')
            for c,(xx,yy) in zip(points,coords):
                # C markers: the saved size and the per-mode shapes.
                node=marker(mode,xx,yy,3.6,opacity=.85 if dashed else 1,width=0)
                art.append(node.replace('<path ',f'<path data-series="{ "cpu" if dashed else "gpu"}" data-size="{c["size"]}" data-rate="{c["aligns_per_sec"]}" '))
    art += [line(41.51,592.26,41.51,329.36),line(41.51,592.26,313.61,592.26)]
    return art, gpu


def gpu_screen(pair=False):
    rows = results.pair_comparison() if pair else results.screen_comparison()
    fss = {c['mode']:c for c in rows if c['tool']=='fss'}
    comp = {c['mode']:c for c in rows if c['tool']=='roshambo2'}
    assert set(fss)==set(ORDER) and set(comp)=={'vol','vol_color'}
    left, bottom, ymax = (747.26,589.74,125_000) if pair else (558.67,593.34,1_100_000)
    y = lambda rate: bottom - rate*236/ymax
    art=[line(left,bottom,left+150.58,bottom,width=1.1),line(left,bottom,left,bottom-236,width=1.1)]
    if pair:
        ratios={m:fss[m]['aligns_per_sec']/comp[m]['aligns_per_sec'] for m in comp}
        art.append(text(left,bottom-252,f"vol: {ratios['vol']:.1f}&#215;; color: {ratios['vol_color']:.1f}&#215;",size=10,anchor='start'))
    for value,label in ((0,'0'),(ymax,'125k' if pair else '1.1M')):
        art += [line(left,y(value),left-3,y(value)),text(left-8.18,y(value)+3.23,label,anchor='end')]
    for i,m in enumerate(ORDER):
        xx=left+9.82+i*18.00413
        rate=fss[m]['aligns_per_sec']; yy=y(rate)
        other=comp.get(m)
        lo=y(other['aligns_per_sec']) if other else bottom
        art.append(line(xx,lo,xx,yy,PALETTE[m][0],2.2))
        art.append(marker(m,xx,yy,7.865).replace('<path ',f'<path data-tool="fss" data-rate="{rate}" '))
        label=f'{rate/1e6:.2f}M' if rate>=1e6 else f'{rate/1e3:.0f}k'
        near_top = yy < bottom - 236 + 45
        art.append(text(xx+11.5 if near_top else xx+3.8777,
                        yy+4 if near_top else yy-9.15,label,
                        size=15,anchor='start',rotate=None if near_top else -90))
        if other:
            oy=y(other['aligns_per_sec'])
            art.append(marker(m,xx,oy,6.435,hollow=True,width=1.32).replace('<path ',f'<path data-tool="roshambo2" data-rate="{other["aligns_per_sec"]}" '))
            art.append(text(xx,oy+2.190399,'9' if m=='vol' else '10',size=8.45,color=PALETTE[m][0],bold=True))
    return art, list(fss.values())+list(comp.values())


def layer_spans(source):
    depth,start,spans=0,None,[]
    for match in re.finditer(r'<!--.*?-->|<[^>]+>',source,re.S):
        t=match.group()
        if t.startswith(('<?','<!')): continue
        if t.startswith('</'):
            depth-=1
            if depth==2: spans.append((start,match.end()))
        else:
            if depth==2: start=match.start()
            if t.endswith('/>'):
                if depth==2: spans.append((start,match.end()))
            else: depth+=1
    # defs/style is also at depth 2; retain only spans inside Layer_1.
    begin=source.index('<g id="Layer_1"')
    return [(a,b) for a,b in spans if a>begin]


def main():
    from update_benchmark_panels import main as update_current_layout
    update_current_layout()


if __name__=='__main__': main()
