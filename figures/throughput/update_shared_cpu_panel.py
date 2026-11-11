"""Replace only the CPU comparison chart of panel D inside the hand-edited fig2_alt.svg, drawn from
paper_results.cpu_comparison(); every other element is preserved verbatim.
Run: python update_shared_cpu_panel.py
"""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from update_large_library_panels import ORDER, PALETTE, line, marker, text, layer_spans
import paper_results

HERE = Path(__file__).resolve().parent
NS = '{http://www.w3.org/2000/svg}'
COMPARATORS = {'vol': [('phesa', 1), ('rdkit', 2), ('shaep', 3)],
               'vol_color': [('rdkit', 4)],
               'vol_and_surf_esp': [('espsim', 5), ('shaep', 6)],
               'pharm': [('pharao', 7), ('phesa', 8)]}


def layout_centers(values, gap=10):
    """Pool-adjacent-violators layout: move ``values`` as little as possible (least squares) so that
    consecutive centers are at least ``gap`` apart."""
    blocks = []
    for i, value in enumerate(values):
        blocks.append(([i], value - i * gap))
        while len(blocks) > 1 and blocks[-2][1] > blocks[-1][1]:
            b, yb = blocks.pop()
            a, ya = blocks.pop()
            blocks.append((a + b, (len(a)*ya + len(b)*yb)/(len(a)+len(b))))
    result = [0.] * len(values)
    for indices, mean in blocks:
        for i in indices:
            result[i] = mean + i * gap
    return result


def artwork(doc):
    rows = [c for c in doc['cells'].values() if c.get('workload') == 'screen']
    assert len(rows) == 17 and all(c['valid'] and c['size'] == 1000 for c in rows)
    assert len({c['shared_input_manifest_sha256'] for c in rows if c['tool'] != 'fss'}) == 1
    data = {(c['tool'], c['mode']): c for c in rows}
    # Requested top tick; retain measured points slightly above it without clipping.
    ymax = 2500
    yy = lambda rate: 595.51 - rate * 236/ymax
    art = [line(360.58,595.51,511.16,595.51,width=1.1),
           line(360.58,595.51,360.58,359.51,width=1.1)]
    for v, label in [(0,'0'),(ymax,f'{ymax/1000:g}k')]:
        art += [line(360.58,yy(v),357.58,yy(v)), text(352.4,yy(v)+3.23,label,anchor='end')]
    for i, mode in enumerate(ORDER):
        xx = 370.4 + i * 18.00413
        fss = data['fss', mode]
        entries = [('fss', None, fss)] + [(tool, num, data[tool, mode]) for tool,num in COMPARATORS.get(mode, [])]
        entries.sort(key=lambda e: -e[2]['aligns_per_sec'])
        centers = [yy(e[2]['aligns_per_sec']) for e in entries]
        external = [j for j,e in enumerate(entries) if e[0] != 'fss']
        packed = layout_centers([centers[j] for j in external])
        shift = max(0, max(packed, default=595.51)-595.51)
        for j, cy in zip(external, packed):
            centers[j] = cy-shift
        # Keep the complete 5 visible above the foreground filled hexagon.
        # This is a display offset only; data-axis-y retains the measured value.
        if mode == 'vol_and_surf_esp':
            centers[next(j for j, e in enumerate(entries) if e[0] == 'espsim')] -= 5
        # Stem links compared modes; singleton stems terminate at the baseline.
        art.append(line(xx,min(centers),xx,max(centers) if len(centers)>1 else 595.51,PALETTE[mode][0],2.2))
        # Preserve the manual top-to-bottom stacking, including each number.
        # Rate captions sit behind the markers, as in the manually edited SVG.
        labels = []
        shapes = []
        manual_stack = mode == 'vol_and_surf_esp'
        ordered = list(zip(entries, centers))
        if not manual_stack:
            ordered.sort(key=lambda pair: pair[0][0] != 'fss')
        for (tool,number,row),cy in ordered:
            node = marker(mode,xx,cy,7.865 if tool=='fss' else 6.435,hollow=tool!='fss',width=.66 if tool=='fss' else 1.32)
            shapes.append(node.replace('<path ', f'<path data-tool="{tool}" data-rate="{row["aligns_per_sec"]:.12g}" data-axis-y="{yy(row["aligns_per_sec"]):.5f}" '))
            if tool=='fss':
                rate=row['aligns_per_sec']
                value=f'{rate/1000:.1f}k' if rate>=1000 else (f'{rate:.1f}' if rate<10 else f'{rate:.0f}')
                lx=xx-5.12 if mode=='pharm' else xx+3.88
                near_top = cy < 359.51 + 45
                label_y = cy - (17 if mode == 'vol_and_surf_esp' else 9.15)
                labels.append(text(xx+11.5 if near_top else lx,
                                   cy+4 if near_top else label_y,value,
                                   size=15,anchor='start',rotate=None if near_top else -90))
            else:
                (shapes if manual_stack else labels).append(text(xx,cy+2.19,str(number),size=8.45,color=PALETTE[mode][0],bold=True))
        art += labels + shapes if manual_stack else shapes + labels
    art[0] = art[0].replace('<path ', '<path id="shared-cpu-start" ', 1)
    art[-1] = art[-1].replace('<text ', '<text id="shared-cpu-end" ', 1)
    return '\n    '.join(art)


def main():
    from update_benchmark_panels import main as update_current_layout
    update_current_layout()


if __name__=='__main__':
    main()
