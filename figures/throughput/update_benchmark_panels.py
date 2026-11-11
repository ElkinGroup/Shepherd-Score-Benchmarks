"""Update data marks in the current throughput SVG; never redraw its layout."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import paper_results as results

HERE = Path(__file__).resolve().parent
SVG = HERE / "fig2_alt.svg"
META = "throughput-data-layout"


def measurements():
    values = {}
    def add(key, value):
        value = float(value)
        if key in values or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid or duplicate measurement: {key}")
        values[key] = value
    cells = results.document()['preparation']
    shared = [c for c in cells if c['stage'] == 'embed']
    if len(shared) != 1 or shared[0]['n'] != 5000:
        raise ValueError('Expected one 5,000-conformer embedding measurement')
    base = shared[0]['seconds']
    add('prep/shared:embed', 300000 / base)
    for c in cells:
        if c['stage'] != 'convert':
            continue
        if c['input_sha256'] != shared[0]['output_sdf_sha256']:
            raise ValueError('Preparation inputs differ')
        config = c['config']
        tool, mode = ('roshambo2', 'shape' if config.endswith('shape') else 'shape+colour') if config.startswith('roshambo2_') else ('fss', config)
        if config == 'sdf':
            continue
        add(f'prep/{tool}:{mode}', 300000 / (base + c['seconds']))
    for series, rows in [("gpu", results.gpu_ladder()), ("cpu", results.cpu_ladder())]:
        for c in rows:
            add(f"scaling/{series}/{c['mode']}/{c['size']}", c["aligns_per_sec"])
    for panel, rows in [("cpu", results.cpu_comparison()),
                        ("screen", results.screen_comparison()),
                        ("pair", results.pair_comparison())]:
        for c in rows:
            # ShaEP ESP-only is not one of the displayed comparisons.
            if panel == "cpu" and c["tool"] == "shaep" and c["mode"] == "vol_esp":
                continue
            add(f"{panel}/{c['tool']}/{c['mode']}", c["aligns_per_sec"])
    return values


def attributes(svg, ident, changes):
    pattern = r'<[a-z]+\b[^>]*\bid="' + re.escape(ident) + r'"[^>]*>'
    def replace(m):
        tag = m[0]
        for key, value in changes.items():
            pat = r'\s' + re.escape(key) + r'="[^"]*"'
            if value is None:
                tag = re.sub(pat, "", tag)
            else:
                value = html.escape(str(value), quote=True)
                attr = f' {key}="{value}"'
                tag = re.sub(pat, lambda _: attr, tag) if re.search(pat, tag) else tag.replace(" ", attr+" ", 1)
        return tag
    out, count = re.subn(pattern, replace, svg)
    if count != 1:
        raise ValueError(f"Missing/duplicate data element {ident}; review the saved layout")
    return out


def static_signature(svg, rules):
    root = ET.fromstring(svg)
    by_id = {e.get("id"): e for e in root.iter() if e.get("id")}
    for rule in rules:
        node = by_id[rule["id"]]
        for name in rule["mutable"]:
            if name == "text":
                for e in node.iter():
                    e.text = None
            else:
                node.attrib.pop(name, None)
    for e in root.findall("{*}metadata"):
        if e.get("id") == META:
            root.remove(e)
    return hashlib.sha256(ET.tostring(root)).hexdigest()


def update(svg, values):
    root = ET.fromstring(svg)
    meta = root.find(f"{{*}}metadata[@id='{META}']")
    if meta is None:
        raise ValueError("Missing saved layout metadata; refusing to redraw")
    ref = json.loads(meta.text)
    if set(values) != set(ref["values"]):
        raise ValueError("Measurement set changed; review modes, sizes and layout")
    if static_signature(svg, ref["rules"]) != ref["static_sha256"]:
        raise ValueError("Manual layout changed; rebind data marks before updating")
    for key, value in values.items():
        low, high = ref["limits"][key]
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} exceeds fixed axes; review layout")
    shared_rate = values["prep/shared:embed"]
    if any(v > shared_rate for k,v in values.items() if k.startswith("prep/")):
        raise ValueError("Total preparation cannot be faster than its shared stage")
    def delta(key):
        old, new = ref["values"][key], values[key]
        if math.isclose(old, new, rel_tol=1e-10, abs_tol=1e-10):
            return 0.
        return ref["slopes"][key] * (math.log10(new/old) if key.startswith("scaling/") else new-old)
    for rule in ref["rules"]:
        kind, ident = rule["kind"], rule["id"]
        changes = dict(rule["original"])
        if kind in ("move", "label"):
            dy = delta(rule["key"]) / rule["parent_y_scale"]
            if dy:
                changes["transform"] = f"translate(0 {dy:.9f}) " + (changes.get("transform") or "")
        elif kind == "bar":
            key = rule["key"]
            if delta(key):
                height = values[key] * 242 / 1200
                changes.update(y=f"{304-height:.6f}", height=f"{height:.6f}",
                               **{"data-total-seconds": str(300000/values[key])})
            if delta(key) or delta("prep/shared:embed"):
                conversion = 0. if key == "prep/shared:embed" else 300000/values[key] - 300000/shared_rate
                changes["data-conversion-seconds"] = str(conversion)
        elif kind == "curve":
            points = rule["points"]
            changes["points"] = rule["original"]["points"] if not any(delta(k) for k in rule["keys"]) else " ".join(
                f"{x:.8f} {y+delta(k)/rule['y_scale']:.8f}" for (x,y),k in zip(points,rule["keys"]))
        elif kind == "stem":
            d1 = delta(rule["ends"][0]) if rule["ends"][0] else 0.
            d2 = delta(rule["ends"][1]) if rule["ends"][1] else 0.
            if d1 or d2:
                x,y,v = rule["geometry"]
                changes["d"] = f"M{x},{y+d1/rule['y_scale']:.8f}v{v+(d2-d1)/rule['y_scale']:.8f}"
        if "data-rate" in changes and "key" in rule:
            if delta(rule["key"]):
                changes["data-rate"] = str(values[rule["key"]])
        svg = attributes(svg, ident, changes)
        if kind == "label":
            value = values[rule["key"]]
            label = rule["label"]
            if delta(rule["key"]):
                suffix = rule["suffix"]
                divisor = {"":1,"k":1000,"M":1000000}[suffix]
                label = f"{value/divisor:.{rule['decimals']}f}"+suffix
            # The id sits on the <text> or on a group wrapping it.
            pattern = r'(<(?:text|g)\b[^>]*\bid="'+re.escape(ident)+r'"[^>]*>)(.*?)(</text>)'
            def replace(m):
                body = re.sub(r'(<tspan\b[^>]*>).*?(</tspan>)',
                              lambda t:t[1]+html.escape(label)+t[2],m[2],flags=re.S)
                return m[1]+body+m[3]
            svg, count = re.subn(pattern, replace, svg, flags=re.S)
            if count != 1:
                raise ValueError(f"Missing label text {ident}; review the saved layout")
    if static_signature(svg, ref["rules"]) != ref["static_sha256"]:
        raise ValueError("Update changed static artwork")
    return svg


def y_scale(transform):
    """Vertical scale of an SVG transform list (data marks only move vertically)."""
    scale = 1.
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", transform or ""):
        v = [float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", args)]
        if name == "scale":
            scale *= v[-1]
        elif name == "matrix" and v[1] == v[2] == 0:
            scale *= v[3]
        elif name != "translate":
            raise ValueError(f"Cannot rebind under a rotated or skewed transform: {transform}")
    return scale


def rebind(svg, values):
    """Re-anchor the saved data layout to reviewed manual artwork.

    Keeps each mark's measurement key, axis slopes and limits from the saved
    metadata and re-reads everything a manual edit can change: mark attributes,
    curve points, stem geometry, label text, transform scales and the static
    artwork hash. The artwork must already show `values`; review it first.
    """
    root = ET.fromstring(svg)
    meta = root.find(f"{{*}}metadata[@id='{META}']")
    if meta is None:
        raise ValueError("No saved layout metadata; copy it from the last committed SVG that has it")
    ref = json.loads(meta.text)
    if set(values) != set(ref["values"]):
        raise ValueError("Measurement set changed; review modes, sizes and layout")
    parent = {child: node for node in root.iter() for child in node}
    by_id = {e.get("id"): e for e in root.iter() if e.get("id")}
    for rule in ref["rules"]:
        node = by_id[rule["id"]]
        rule["original"] = {k: node.get(k) for k in rule["mutable"] if k != "text"}
        scale, up = 1., parent.get(node)
        while up is not None:
            scale *= y_scale(up.get("transform"))
            up = parent.get(up)
        rule["parent_y_scale"] = scale
        if "y_scale" in rule:
            rule["y_scale"] = scale * y_scale(node.get("transform"))
        if rule["kind"] == "curve":
            v = [float(x) for x in node.get("points").split()]
            rule["points"] = [list(p) for p in zip(v[::2], v[1::2])]
        elif rule["kind"] == "stem":
            m = re.fullmatch(r"M([-\d.]+),([-\d.]+)v([-\d.]+)", node.get("d"))
            if m is None:
                raise ValueError(f"Stem {rule['id']} is not one vertical segment")
            rule["geometry"] = [float(g) for g in m.groups()]
        elif rule["kind"] == "label":
            spans = [e for e in node.iter() if e.tag.endswith("}tspan")]
            if len(spans) != 1:
                raise ValueError(f"Label {rule['id']} must hold exactly one tspan")
            rule["label"] = spans[0].text or ""
            divisor = {"": 1, "k": 1000, "M": 1000000}[rule["suffix"]]
            expected = f"{values[rule['key']]/divisor:.{rule['decimals']}f}" + rule["suffix"]
            if rule["label"] != expected:
                raise ValueError(f"Label {rule['id']} ({rule['key']}) reads {rule['label']!r} but "
                                 f"the value formats as {expected!r}; fix the artwork or the data")
    ref["values"] = dict(values)
    ref["static_sha256"] = static_signature(svg, ref["rules"])
    block = json.dumps(ref, separators=(",", ":"))
    return re.sub(r'(<metadata id="' + META + r'">).*?(</metadata>)',
                  lambda m: m[1] + block + m[2], svg, count=1, flags=re.S)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, help="directory containing the result JSON files")
    ap.add_argument("--prefix", help="write a review copy beside the publication SVG")
    ap.add_argument("--prep", choices=["shared"], default="shared")
    ap.add_argument("--force", action="store_true", help="accepted for compatibility; never redraws")
    ap.add_argument("--rebind", action="store_true",
                    help="re-anchor the saved layout to reviewed manual edits; drawing unchanged")
    args = ap.parse_args()
    if args.data_dir:
        results.DATA = args.data_dir.resolve()
    original = SVG.read_text(encoding="utf-8")
    if args.rebind:
        SVG.write_text(rebind(original, measurements()), encoding="utf-8")
        print("Rebound the saved layout to the current artwork")
        return
    output = update(original, measurements())
    target = HERE/(args.prefix+".svg") if args.prefix else SVG
    target.write_text(output, encoding="utf-8")
    import sys
    sys.path.insert(0, str(HERE.parent))
    import svg2png
    stem = target.with_suffix("")
    svg2png.render(str(target), str(stem)+"_flat.png", 4200)
    svg2png.render(str(target), str(stem)+"_preview.png", 1800)
    print("Updated data marks; retained the current throughput layout")


if __name__ == "__main__":
    main()
