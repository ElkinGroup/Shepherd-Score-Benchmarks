"""The one SVG flattener for every figure in this repo. `svgtools.flatten(path)` rewrites an
editable matplotlib SVG in place into a flat, ungrouped list of <path>/<text>/<image> leaves with
composed transforms, so every object is directly selectable in Illustrator/Inkscape."""
from __future__ import annotations

import copy
import os
import sys
import xml.etree.ElementTree as ET

SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", XLINK)

HERE = os.path.dirname(os.path.abspath(__file__))

_DROP_IN_TREE = {"metadata"}

# Markers a vector editor stamps into a file it saves (matplotlib writes "Created with matplotlib").
_HAND_EDIT_MARKERS = ("Adobe Illustrator", "Inkscape", "inkscape:", "sodipodi")


def guard_svg(path, force=None):
    """True if `path` may be written; an existing hand-edited SVG is refused unless --force is on the command line (or force=True)."""
    if force is None:
        force = "--force" in sys.argv
    if force or not os.path.exists(path):
        return True
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            head = fh.read(4000)
    except OSError:
        return True
    if any(m in head for m in _HAND_EDIT_MARKERS):
        rel = os.path.relpath(path)
        print(f"  SKIP {rel}: hand-edited SVG preserved (contains manual edits this "
              f"script does not reproduce).\n"
              f"       Pass --force to regenerate from data; recover the hand edits "
              f"from git afterward.")
        return False
    return True


def _tag(e):
    return e.tag.split("}")[-1]


def _compose(outer, inner):
    """SVG transform applying `outer` then `inner`."""
    parts = [t for t in (outer, inner) if t and t.strip()]
    return " ".join(parts) if parts else None


def _inline_uses(root):
    """Replace <use> with a concrete copy of the referenced element."""
    by_id = {e.get("id"): e for e in root.iter()
             if e.get("id") and _tag(e) in ("path", "line", "rect", "circle", "polygon", "polyline")}
    parents = {c: p for p in root.iter() for c in p}
    for use in [e for e in root.iter() if _tag(e) == "use"]:
        href = use.get(f"{{{XLINK}}}href") or use.get("href") or ""
        src = by_id.get(href.lstrip("#"))
        if src is None:
            continue
        node = copy.deepcopy(src)
        node.attrib.pop("id", None)
        x, y = use.get("x", "0"), use.get("y", "0")
        node.set("transform", _compose(f"translate({x} {y})", node.get("transform")))
        if use.get("style"):
            node.set("style", use.get("style"))
        p = parents[use]
        p[list(p).index(use)] = node


def _flatten(node, tf, style):
    """Flat list of leaf elements with group transforms/styles pushed down; clip-path and id are dropped."""
    t = _tag(node)
    if t in _DROP_IN_TREE or t == "defs":
        return []
    if t == "g":
        gtf = _compose(tf, node.get("transform"))
        gstyle = "; ".join(s for s in (style, node.get("style")) if s) or None
        out = []
        for ch in list(node):
            out += _flatten(ch, gtf, gstyle)
        return out
    newtf = _compose(tf, node.get("transform"))
    if newtf:
        node.set("transform", newtf)
    if style:  # inherited group style first; the element's own props win
        node.set("style", "; ".join(s for s in (style, node.get("style")) if s))
    node.attrib.pop("clip-path", None)
    node.attrib.pop("id", None)
    return [node]


def flatten(path, out_path=None):
    out_path = out_path or path
    tree = ET.parse(path)
    root = tree.getroot()

    _inline_uses(root)

    # Merge every top-level <defs>'s <style> children into one <defs>.
    merged_style = ET.Element(f"{{{SVG}}}style")
    style_text = []
    for d in [e for e in list(root) if _tag(e) == "defs"]:
        for s in [c for c in d if _tag(c) == "style"]:
            if s.text and s.text.strip():
                style_text.append(s.text.strip())
    merged_style.text = "\n".join(style_text)
    style_defs = ET.Element(f"{{{SVG}}}defs")
    style_defs.append(merged_style)

    flat = []
    for child in list(root):
        flat += _flatten(child, None, None)

    for c in list(root):
        root.remove(c)
    root.append(style_defs)
    for c in flat:
        root.append(c)

    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    n_g = sum(1 for e in root.iter() if _tag(e) == "g")
    n_use = sum(1 for e in root.iter() if _tag(e) == "use")
    n_clip = sum(1 for e in root.iter() if "clip-path" in e.attrib)
    n_text = sum(1 for e in root.iter() if _tag(e) == "text")
    n_img = sum(1 for e in root.iter() if _tag(e) == "image")
    print(f"flattened {os.path.basename(out_path)}: groups={n_g} use={n_use} "
          f"clip-path={n_clip} text={n_text} image={n_img} style_rules_merged={len(style_text)}")
