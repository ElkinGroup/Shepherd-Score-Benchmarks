#!/usr/bin/env python3
"""Figure 1 panel A: the per-mode 3-D alignment-mode gallery for aspirin, rendered headless with
PyVista/VTK into data/render3d/ and composited into fig1.png by hand.

    python render_3d.py        # needs pyvista (+ vtk) and an OpenGL-capable session
"""

from __future__ import annotations

import os

import numpy as np
import pyvista as pv

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
# fast_shepherd_score must be checked out as a sibling of this repo: ../../../fast_shepherd_score
_FORK = os.path.abspath(os.path.join(HERE, "..", "..", "..", "fast_shepherd_score"))
if _FORK not in sys.path:
    sys.path.insert(0, _FORK)

MOL_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
MOL_NAME = "aspirin"
SEED = 42

MODES = ["vol", "vol_esp", "vol_lipo", "surf", "surf_esp", "vol_and_surf_esp", "vol_color", "pharm"]

# 3x3 star used when the panels are composited: base modes on the axes, combinations in the corners.
GRID = {
    "vol_esp": (0, 0),  "vol": (0, 1),  "vol_color": (0, 2),
    "surf":    (1, 0),                  "pharm":     (1, 2),
    "surf_esp": (2, 0),                 "vol_and_surf_esp": (2, 2),
}

VDW = {1: 1.10, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 15: 1.80, 16: 1.80, 17: 1.75}
GREY = "#9aa3ac"          # neutral grey for volume / surface


def _orient(mol):
    """Rotate onto the molecule's principal axes so the ring faces the camera, then centre."""
    p = mol["atom_pos"]
    w, V = np.linalg.eigh(np.cov((p - p.mean(0)).T))
    R = V[:, ::-1]
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    mol = dict(mol)
    for k in ("atom_pos", "surf_pos", "pharm_ancs", "pharm_vecs"):
        if len(mol[k]):
            mol[k] = np.asarray(mol[k]) @ R
    com = mol["atom_pos"].mean(0)
    for k in ("atom_pos", "surf_pos", "pharm_ancs"):
        if len(mol[k]):
            mol[k] = mol[k] - com
    return mol


from shepherd_score.score.constants import P_TYPES
from shepherd_score.pharm_utils.pharmacophore import feature_colors

# Pharmacophore colours from the figure key (override shepherd_score feature_colors).
PHARM_COLORS = {
    "Hydrophobe":       (1.000, 0.545, 0.000),   # #ff8b00 orange
    "LumpedHydrophobe": (1.000, 0.545, 0.000),   # #ff8b00 orange
    "Aromatic":         (0.169, 0.667, 0.192),   # #2baa31 green
    "Acceptor":         (0.525, 0.502, 0.835),   # #8680d5 light indigo
    "Donor":            (0.639, 0.361, 0.608),   # #a35c9b magenta
}
AROM_TILT = 0.45   # fraction to tilt the aromatic ring-normal arrows into the page plane

PNG_DIR = os.path.join(DATA, "render3d")
# Charge/ESP colours matched to the figure key: negative red, white neutral, positive blue.
from matplotlib.colors import LinearSegmentedColormap
ESP_CMAP = LinearSegmentedColormap.from_list(
    "key_esp", ["#d2645b", "#ffffff", "#608cc4"])
LIPO_CMAP = LinearSegmentedColormap.from_list(
    "key_lipo", ["#008c95", "#ffffff", "#d35d7f"])
VIEW = 1000                    # offscreen canvas (square); SSAA upsamples
VOL_GREY = "#d3d7db"
CPK = {1: "#ffffff", 6: "#3b3f44", 7: "#2f6fd0", 8: "#e8483b", 9: "#7fd33f",
       15: "#ff8000", 16: "#e6c12d", 17: "#33d033"}   # element colours (ball-and-stick)
# surf is drawn as the actual surface point cloud (a small sphere per point), matching how the
# algorithm represents the surface.
SURF_POINTS_3D = 300
SURF_PROBE_3D = 0.35           # close to vdW so the shell hugs the molecule
# Fixed parallel-projection camera shared by every panel, for one consistent scale/orientation.
VIEW_DIR = np.array([0.28, -0.45, 1.0])
PARALLEL_SCALE = 6.2
VOL_RSCALE = 0.75
VOL_OPACITY = 1.0
SURF_PT_R = 0.22              # radius of each surface-point sphere
PHARM_R = 0.80
PHARM_ALPHA = 0.6
VOL_COLOR_PHARM_R = 1.7       # vol_color: the aromatic/hydrophobe hybrid sphere
VOL_COLOR_PHARM_R_SMALL = 1.4  # vol_color: all other pharm spheres
VOL_COLOR_PHARM_ALPHA = 0.45   # transparent enough that the grey volume shows through
MAT = dict(smooth_shading=True, ambient=0.55, diffuse=0.55,
           specular=0.0, specular_power=20, show_scalar_bar=False)


def build():
    """Embed aspirin and featurise it, PCA-oriented so every panel shares one orientation."""
    from rdkit import Chem
    from shepherd_score.conformer_generation import embed_conformer_from_smiles
    from shepherd_score.container import Molecule
    np.random.seed(SEED)
    print(f"embedding {MOL_NAME} ...")
    rd = embed_conformer_from_smiles(MOL_SMILES, MMFF_optimize=True, random_seed=SEED)
    m = Molecule(rd, num_surf_points=SURF_POINTS_3D, probe_radius=SURF_PROBE_3D,
                 pharm_multi_vector=False, surface_method="smooth_sdf")
    heavy = Chem.RemoveHs(m.mol)
    Z = np.array([a.GetAtomicNum() for a in heavy.GetAtoms()], dtype=int)
    bonds = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()]
                      for b in heavy.GetBonds()], dtype=int).reshape(-1, 2)
    mol = {
        "atom_pos": np.asarray(m.atom_pos, float), "atom_Z": Z, "bonds": bonds,
        "charges": np.asarray(m.partial_charges, float)[m._nonH_atoms_idx],
        "lipophilicity": np.asarray(m.get_lipophilicity(), float),
        "surf_pos": np.asarray(m.surf_pos, float),
        "surf_esp": np.asarray(m.surf_esp, float),
        "pharm_types": np.asarray(m.pharm_types, int),
        "pharm_ancs": np.asarray(m.pharm_ancs, float),
        "pharm_vecs": np.asarray(m.pharm_vecs, float),
    }
    return _orient(mol)


def _plotter():
    p = pv.Plotter(off_screen=True, window_size=(VIEW, VIEW))
    p.set_background("white")
    p.enable_parallel_projection()
    d = VIEW_DIR / np.linalg.norm(VIEW_DIR)
    p.camera_position = [tuple(d * 50), (0, 0, 0), (0, 1, 0)]
    p.camera.parallel_scale = PARALLEL_SCALE
    p.enable_depth_peeling(12)              # correct translucency ordering
    try:
        p.enable_ssao(radius=1.2, bias=0.01)
    except Exception:
        pass
    p.enable_anti_aliasing("ssaa")
    return p


def _atom_spheres(p, mol, color=None, scalars=None, clim=None, opacity=1.0, rscale=1.0, cmap=ESP_CMAP):
    poly = pv.PolyData(mol["atom_pos"])
    poly["r"] = np.array([VDW.get(int(z), 1.6) for z in mol["atom_Z"]]) * rscale
    if scalars is not None:
        poly["c"] = scalars
    g = poly.glyph(geom=pv.Sphere(radius=1.0, theta_resolution=40, phi_resolution=40),
                   scale="r", orient=False)
    if scalars is not None:
        p.add_mesh(g, scalars="c", cmap=cmap, clim=clim, opacity=opacity, **MAT)
    else:
        p.add_mesh(g, color=color, opacity=opacity, **MAT)


def _surf_points(p, mol, color=None, esp=False, opacity=1.0, radius=SURF_PT_R):
    """The surface as an array of small 3-D spheres (the algorithm's point cloud)."""
    poly = pv.PolyData(mol["surf_pos"])
    if esp:
        poly["esp"] = mol["surf_esp"]
    g = poly.glyph(geom=pv.Sphere(radius=radius, theta_resolution=16, phi_resolution=16),
                   scale=False, orient=False)
    if esp:
        v = np.abs(mol["surf_esp"]).max() or 1.0
        p.add_mesh(g, scalars="esp", cmap=ESP_CMAP, clim=(-v, v), opacity=opacity, **MAT)
    else:
        p.add_mesh(g, color=color, opacity=opacity, **MAT)


def _bonds(p, mol, color="#7c858d", radius=0.05, opacity=1.0):
    pos = mol["atom_pos"]
    for i, j in mol["bonds"]:
        tube = pv.Line(pos[i], pos[j]).tube(radius=radius, n_sides=18)
        p.add_mesh(tube, color=color, opacity=opacity, **MAT)


def _col(name):
    return np.array(PHARM_COLORS.get(name, feature_colors.get(name, (0.5, 0.5, 0.5))), float)


def _pharm(p, mol, arrows, radius=PHARM_R, alpha=PHARM_ALPHA, small_radius=None):
    view_hat = VIEW_DIR / np.linalg.norm(VIEW_DIR)
    right = np.cross(view_hat, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)                    # screen-horizontal: hemisphere split plane

    # Group features sharing an anchor (an aromatic ring is both Aromatic and Hydrophobe) so the
    # shared sphere can be split into per-coding hemispheres.
    groups, order = {}, []
    for t, anc, vec in zip(mol["pharm_types"], mol["pharm_ancs"], mol["pharm_vecs"]):
        name = P_TYPES[int(t)] if int(t) < len(P_TYPES) else "Dummy"
        key = tuple(np.round(anc, 2))
        if key not in groups:
            groups[key] = {"anc": np.asarray(anc, float), "feats": []}
            order.append(key)
        groups[key]["feats"].append((name, np.asarray(vec, float)))

    for key in order:
        anc, feats = groups[key]["anc"], groups[key]["feats"]
        is_arom_hybrid = any(nm == "Aromatic" for nm, _ in feats)
        r = radius if (small_radius is None or is_arom_hybrid) else small_radius
        if len(feats) == 1:
            p.add_mesh(pv.Sphere(radius=r, center=anc, theta_resolution=48,
                                 phi_resolution=48), color=_col(feats[0][0]),
                       opacity=alpha, **MAT)
        else:                                          # two codings -> hemispheres aligned to arrows
            def _sdir(v):
                vp = v - (v @ view_hat) * view_hat     # in-page component of the arrow
                nrm = np.linalg.norm(vp)
                return vp / nrm if nrm > 1e-6 else None
            d0, d1 = _sdir(feats[0][1]), _sdir(feats[1][1])
            if d0 is not None and d1 is not None:
                n = d0 - d1                            # donor+acceptor: align halves to arrows
            else:
                n = right                              # directionless pair: vertical dividing line
            n = n / (np.linalg.norm(n) or 1.0)         # feats[0]'s arrow points toward +n
            full = pv.Sphere(radius=r, center=anc, theta_resolution=72, phi_resolution=72)
            p.add_mesh(full.clip(normal=n, origin=anc, invert=False),
                       color=_col(feats[0][0]), opacity=alpha, **MAT)
            p.add_mesh(full.clip(normal=n, origin=anc, invert=True),
                       color=_col(feats[1][0]), opacity=alpha, **MAT)

        if not arrows:
            continue
        for nm, vec in feats:
            if np.linalg.norm(vec) < 1e-6:
                continue
            u = vec / np.linalg.norm(vec)
            if nm == "Aromatic":
                # the ring normal points mostly out of the page; tilt it toward the page plane
                u_in = u - (u @ view_hat) * view_hat
                if np.linalg.norm(u_in) > 1e-6:
                    u_in /= np.linalg.norm(u_in)
                    u = (1 - AROM_TILT) * u + AROM_TILT * u_in
                    u /= np.linalg.norm(u)
                dirs = (u, -u)                         # bidirectional (both ring faces)
            else:
                dirs = (u,)
            for d in dirs:
                arr = pv.Arrow(start=anc + d * r * 0.9, direction=d, scale=2.3,
                               tip_length=0.36, tip_radius=0.20, shaft_radius=0.08)
                p.add_mesh(arr, color=tuple(_col(nm) * 0.8), opacity=alpha, **MAT)


STICK_R = 0.14                 # aspirin reference: PyMOL-style stick radius
STICK_MAT = dict(smooth_shading=True, ambient=0.30, diffuse=0.82,
                 specular=0.05, specular_power=20, show_scalar_bar=False)


def _render_ball_stick(mol, out_path):
    """CPK sticks-only reference of aspirin, on the same camera as every panel."""
    p = _plotter()
    pos, Z = mol["atom_pos"], mol["atom_Z"]

    def scol(z):                                       # carbons match the vol grey
        return VOL_GREY if int(z) == 6 else CPK.get(int(z), "#cc66cc")

    for i, j in mol["bonds"]:                          # element-split cylinders
        a, b = pos[i], pos[j]
        mid = (a + b) / 2.0
        p.add_mesh(pv.Line(a, mid).tube(radius=STICK_R, n_sides=28, capping=True),
                   color=scol(Z[i]), **STICK_MAT)
        p.add_mesh(pv.Line(mid, b).tube(radius=STICK_R, n_sides=28, capping=True),
                   color=scol(Z[j]), **STICK_MAT)
    for xyz, z in zip(pos, Z):                          # rounded joints / caps
        p.add_mesh(pv.Sphere(radius=STICK_R, center=xyz, theta_resolution=28,
                             phi_resolution=28), color=scol(z), **STICK_MAT)
    p.screenshot(out_path, transparent_background=False)
    p.close()
    return out_path


def render_panel(key, mol, out_path):
    p = _plotter()
    qv = np.abs(mol["charges"]).max() or 1.0
    if key == "vol":                                  # grey vdW volume
        _atom_spheres(p, mol, color=VOL_GREY, opacity=VOL_OPACITY, rscale=VOL_RSCALE)
    elif key == "vol_esp":                            # volume shaded by charge
        _atom_spheres(p, mol, scalars=mol["charges"], clim=(-qv, qv),
                      opacity=VOL_OPACITY, rscale=VOL_RSCALE)
    elif key == "vol_lipo":
        limit = np.abs(mol["lipophilicity"]).max() or 1.0
        _atom_spheres(p, mol, scalars=mol["lipophilicity"], clim=(-limit, limit),
                      opacity=VOL_OPACITY, rscale=VOL_RSCALE, cmap=LIPO_CMAP)
    elif key == "surf":                               # surface point cloud (grey)
        _surf_points(p, mol, color=VOL_GREY)
    elif key == "surf_esp":                           # surface points coloured by ESP
        _surf_points(p, mol, esp=True)
    elif key == "vol_and_surf_esp":                   # volume + ESP surface points
        _atom_spheres(p, mol, color=VOL_GREY, opacity=VOL_OPACITY, rscale=VOL_RSCALE)
        _surf_points(p, mol, esp=True)
    elif key == "vol_color":                          # volume + semi-transparent colour spheres
        _atom_spheres(p, mol, color=VOL_GREY, opacity=VOL_OPACITY, rscale=VOL_RSCALE)
        _pharm(p, mol, arrows=False, radius=VOL_COLOR_PHARM_R, alpha=VOL_COLOR_PHARM_ALPHA,
               small_radius=VOL_COLOR_PHARM_R_SMALL)
    elif key == "pharm":                              # colour features + arrows over sticks
        _bonds(p, mol, opacity=0.9)
        _pharm(p, mol, arrows=True)
    else:
        raise ValueError(key)
    p.screenshot(out_path, transparent_background=False)
    p.close()
    return out_path


def render_all(mol):
    os.makedirs(PNG_DIR, exist_ok=True)
    paths = {}
    print("rendering panels (PyVista offscreen) ...")
    for key in MODES:
        paths[key] = render_panel(key, mol, os.path.join(PNG_DIR, f"{key}.png"))
        print(f"  {key}")
    paths["__ball_stick__"] = _render_ball_stick(mol, os.path.join(PNG_DIR, "ball_stick.png"))
    print("  ball_stick (centre)")
    return paths


def main():
    mol = build()
    render_all(mol)
    print("panels are composited into fig1.svg / fig1.png by hand (Illustrator)")


if __name__ == "__main__":
    main()
