"""Render the hand-edited fig2_alt.svg to fig2_alt_flat.png and fig2_alt_preview.png without
rebuilding its layout; a missing SVG is recovered from the committed copy. Data updates use make_figure.py and preserve this layout.
"""
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import svg2png


def saved_svg():
    return subprocess.check_output(
        ['git', 'show', 'HEAD:figures/throughput/fig2_alt.svg'], cwd=HERE)


def main():
    source = HERE / 'fig2_alt.svg'
    if not source.exists():
        data = saved_svg()
        ET.fromstring(data)
        source.write_bytes(data)
        print('Recovered the committed manual layout from Git.')
    before = source.read_bytes()
    ET.fromstring(before)
    svg2png.render(str(source), str(HERE / 'fig2_alt_flat.png'), 4200)
    svg2png.render(str(source), str(HERE / 'fig2_alt_preview.png'), 1800)
    assert source.read_bytes() == before, 'Rendering must not modify the editable SVG'
    print('Rendered publication and preview PNGs; manual SVG unchanged.')


if __name__ == '__main__':
    main()
