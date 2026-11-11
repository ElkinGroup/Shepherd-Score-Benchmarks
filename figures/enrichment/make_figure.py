#!/usr/bin/env python3
"""Update enrichment data in the hand-edited SVG without rebuilding its layout.

Run python figures/enrichment/make_figure.py; --prefix NAME writes a separate copy.
The SVG contains its reference measurements and geometry. Target/method changes
or values outside the fixed axes require a reviewed layout update.
"""
import argparse
import html
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import svg2png
import enrichment_data as data

SVG = HERE / "fig3.svg"
META = "enrichment-data-layout"
NUM = r"-?\d+(?:\.\d+)?"


def measurements(targets):
    tags = [m[0] for m in data.METHODS]
    names = sorted(targets)
    if len(names) != 43 or any(tag not in targets[t] for t in names for tag in tags):
        raise ValueError("Expected complete results for 43 targets and all plotted methods")
    result = {}
    for tag in tags:
        v = np.array([targets[t][tag]["auc"] for t in names], dtype=float)
        if not np.isfinite(v).all() or ((v < 0) | (v > 1)).any():
            raise ValueError(f"Invalid ROC-AUC values for {tag}")
        result[tag] = [float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))]
    return result


def elements(svg):
    return {e.get("id"): e for e in ET.fromstring(svg).iter() if e.get("id")}


def attributes(svg, ident, changes):
    pattern = r'<[a-z]+\b[^>]*\bid="' + re.escape(ident) + r'"[^>]*>'
    def replace(match):
        text = match[0]
        for key, value in changes.items():
            if value is None:
                text = re.sub(r'\s'+re.escape(key)+r'="[^"]*"', '', text)
                continue
            value = html.escape(str(value), quote=True)
            attr = r'\b' + re.escape(key) + r'="[^"]*"'
            if re.search(attr, text):
                text = re.sub(attr, lambda _: f'{key}="{value}"', text)
            else:
                text = text.replace(' ', f' {key}="{value}" ', 1)
        return text
    out, count = re.subn(pattern, replace, svg)
    if count != 1:
        raise ValueError(f"Missing or duplicate SVG element: {ident}")
    return out


def static_signature(svg):
    root = ET.fromstring(svg)
    for node in root.iter():
        ident = node.get("id", "")
        if ident.startswith(("bar-", "sem-", "cap-low-", "cap-high-")):
            node.attrib.pop("d", None)
        elif ident.startswith("comparison-"):
            node.attrib.pop("transform", None)
        elif ident.startswith("significance-"):
            node.attrib.pop("transform", None)
            for child in node.iter():
                child.text = None
                if "style" in child.attrib:
                    child.set("style", re.sub(r"font-size:[\d.]+px;", "", child.get("style")))
        elif ident.startswith("rank-"):
            node.attrib.pop("cy", None)
            node.attrib.pop("class", None)
    for node in root.findall("{*}metadata"):
        if node.get("id") == META:
            root.remove(node)
    return hashlib.sha256(ET.tostring(root)).hexdigest()


def update(svg, targets):
    nodes = elements(svg)
    if META not in nodes:
        raise ValueError("SVG lacks reference layout metadata; refusing to rebuild it")
    ref = json.loads(nodes[META].text)
    if static_signature(svg) != ref["static_sha256"]:
        raise ValueError("Manual layout changed; rebind data marks before updating")
    if sorted(targets) != ref["targets"]:
        raise ValueError("Target set changed; review the layout and caption first")
    stats = measurements(targets)
    if sorted(stats) != sorted(ref["stats"]):
        raise ValueError("Method set changed; review the layout first")
    scale, base = ref["scale"], ref["base"]
    def y(value):
        return base - scale * (value - .5)
    for tag, (mean, sem) in stats.items():
        if y(mean + sem) < ref["top"] or mean - sem < .5:
            raise ValueError(f"{tag} exceeds the fixed plot bounds; review axes")
        if stats[tag] == ref["stats"][tag]:
            for ident in ["bar-"+tag, "sem-"+tag, "cap-low-"+tag, "cap-high-"+tag]:
                if ident in ref["paths"]:
                    svg = attributes(svg, ident, {"d": ref["paths"][ident]})
            continue
        x, width = ref["bars"][tag]
        height = scale * (mean - .5)
        svg = attributes(svg, "bar-"+tag, {"d": f"M{x},{base}h{width}v{-height}h{-width}Z"})
        cx = x + width/2
        svg = attributes(svg, "sem-"+tag, {"d": f"M{cx},{y(mean-sem)}V{y(mean+sem)}"})
        for side, value in [("low",mean-sem),("high",mean+sem)]:
            ident = "cap-"+side+"-"+tag
            if ident not in ref["paths"]:
                continue
            original = list(map(float, re.findall(NUM, ref["paths"][ident])))
            cap_x, _, cap_w = original[:3]
            path = f"M{cap_x},{y(value)}h{cap_w}"
            if len(original) == 6:
                path += f"M{original[3]},{y(mean+sem)}h{original[5]}"
            svg = attributes(svg, ident, {"d": path})
    for a,b,_ in data.COMPARISONS:
        key = a+"-"+b
        info = ref["comparisons"][key]
        delta = -scale * (max(sum(stats[a]),sum(stats[b])) - max(sum(ref["stats"][a]),sum(ref["stats"][b])))
        if info["bracket_y"] + delta < ref["annotation_top"]:
            raise ValueError(f"Comparison {key} would overlap the header/key; review layout")
        # Translation keeps the manually chosen bracket spacing and typography.
        ident = "comparison-"+key
        svg = attributes(svg, ident, {"transform": f"translate(0 {delta})"}) if delta else attributes(svg, ident, {"transform": None})
        mark = data._stars(data.paired_test(targets,ref["targets"],a,b)[2])
        ident = "significance-"+key
        # A star sits above its text baseline; n.s. sits on it. Keep the visible
        # gap to the bracket constant when a result changes significance.
        x = float(re.findall(NUM, info["transform"])[0])
        baseline = info["bracket_y"] - (3.76 if mark == "n.s." else 0.10)
        transform = f"translate({x:g} {baseline:g})"
        svg = attributes(svg, ident, {"transform": (f"translate(0 {delta}) " if delta else "") + transform})
        pattern = r'(<text\b[^>]*id="'+re.escape(ident)+r'"[^>]*>)(.*?)(</text>)'
        def label(m):
            current = "".join(ET.fromstring(m[0]).itertext()).strip()
            if current == mark:
                return m[0]
            body = re.sub(r'(<tspan\b[^>]*>).*?(</tspan>)',lambda z:z[1]+mark+z[2],m[2],flags=re.S)
            # The saved layout uses 4 pt for n.s. and 5 pt for stars.
            block = m[1]+body+m[3]
            if mark != "".join(ET.fromstring(block).itertext()).strip():
                raise ValueError("Significance label replacement failed")
            size = ref["significance_font_sizes"]["ns" if mark == "n.s." else "stars"]
            return re.sub(r"font-size:[\d.]+px", f"font-size:{size}px", block)
        svg = re.sub(pattern,label,svg,flags=re.S)
    modes = ref["modes"]
    for target in ref["targets"]:
        ranked = sorted(modes,key=lambda m:-targets[target][m]["auc"])[:3]
        for rank,mode in enumerate(ranked,1):
            ident = f"rank-{rank}-{target}"
            svg = attributes(svg,ident,{"cy":ref["rows"][mode]["cy"],"class":ref["rows"][mode]["class"]})
    # Return the original bytes when every data-dependent attribute is unchanged.
    if static_signature(svg) != ref["static_sha256"]:
        raise ValueError("Update changed static enrichment artwork")
    return svg


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix")
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--force",action="store_true",help="compatibility option; never rebuilds the layout")
    ap.add_argument("--png-only",action="store_true")
    a=ap.parse_args()
    if a.data_dir: data.DATA=str(a.data_dir.resolve())
    if a.png_only and not a.prefix: ap.error("--png-only requires --prefix")
    original=SVG.read_text(encoding="utf-8")
    output=update(original,data.load_targets())
    out=HERE/(a.prefix+".svg") if a.prefix else SVG
    out.write_text(output,encoding="utf-8")
    pub=HERE/(a.prefix+(".png" if a.png_only else "_pub.png")) if a.prefix else HERE/"fig3_comprehensive_enrichment_2panel_auc_JCIM.png"
    preview=HERE/(a.prefix+"_preview.png") if a.prefix else HERE/"fig3_preview.png"
    svg2png.render(str(out),str(pub),4200)
    if not a.png_only: svg2png.render(str(out),str(preview),2200)
    elif a.prefix: out.unlink()
    print("Updated enrichment values using the saved manual layout")

if __name__ == "__main__":
    main()
