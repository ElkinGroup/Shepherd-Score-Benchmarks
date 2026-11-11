#!/usr/bin/env python3
"""Renders the overlay images the graphical abstract uses: aspirin (reference) plus a second small
molecule, aligned with optimize_ROCS_overlay and drawn headless with PyVista/VTK into
data/overlay_<pair>_volume.png and data/overlay_<pair>_stick.png.

    python render_overlays.py        # needs pyvista(+vtk), rdkit, torch
"""
from __future__ import annotations
import os, sys
import numpy as np
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
# fast_shepherd_score must be checked out as a sibling of this repo: ../../../fast_shepherd_score
_FORK = os.path.abspath(os.path.join(HERE, "..", "..", "..", "fast_shepherd_score"))
if _FORK not in sys.path:
    sys.path.insert(0, _FORK)

SEED = 42
ALPHA = 0.81  # Gaussian-volume width

REF = ("aspirin", "CC(=O)Oc1ccccc1C(=O)O")
FITS = {
    "methylsalicylate": "COC(=O)c1ccccc1O",
    "ibuprofen":        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
}

# Look shared with fig1_modes/render_3d.py.
VDW = {1: 1.10, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 15: 1.80, 16: 1.80, 17: 1.75}
VIEW = 1100
VIEW_DIR = np.array([0.28, -0.45, 1.0])
PARALLEL_SCALE = 6.6
VOL_RSCALE = 0.72
MAT = dict(smooth_shading=True, ambient=0.55, diffuse=0.55,
           specular=0.0, specular_power=20, show_scalar_bar=False)
STICK_MAT = dict(smooth_shading=True, ambient=0.30, diffuse=0.82,
                 specular=0.05, specular_power=20, show_scalar_bar=False)
STICK_R = 0.13

REF_HUE = "#c0e6c2"        # reference: fig 1B green
FIT_HUE = "#7a7ac0"        # fit: fig 1C purple
CPK = {1: "#ffffff", 7: "#2f6fd0", 8: "#e8483b", 9: "#7fd33f",
       15: "#ff8000", 16: "#e6c12d", 17: "#33d033"}


def embed(smiles, seed=SEED):
    from shepherd_score.conformer_generation import embed_conformer_from_smiles
    from rdkit import Chem
    rd = embed_conformer_from_smiles(smiles, MMFF_optimize=True, random_seed=seed)
    heavy = Chem.RemoveHs(rd)
    pos = heavy.GetConformer().GetPositions().astype(float)
    Z = np.array([a.GetAtomicNum() for a in heavy.GetAtoms()], dtype=int)
    bonds = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()]
                      for b in heavy.GetBonds()], dtype=int).reshape(-1, 2)
    return {"pos": pos, "Z": Z, "bonds": bonds}


def align_fit_onto_ref(ref, fit):
    """Optimise the fit heavy-atom cloud onto the reference with the library's SE(3) optimiser."""
    import torch
    from shepherd_score.alignment import optimize_ROCS_overlay
    from shepherd_score.alignment.utils.se3_np import apply_SE3_transform_np
    rp = torch.tensor(ref["pos"], dtype=torch.float32)
    fp = torch.tensor(fit["pos"], dtype=torch.float32)
    _, transform, score = optimize_ROCS_overlay(rp, fp, ALPHA, num_repeats=50)
    T = transform.detach().cpu().numpy()
    out = dict(fit)
    out["pos"] = apply_SE3_transform_np(fit["pos"], T)
    print(f"    aligned  overlap score = {float(score):.3f}")
    return out


def joint_orient(ref, fit):
    """Rotate the scene onto the reference's principal axes and centre on its COM."""
    p = ref["pos"]
    com = p.mean(0)
    w, V = np.linalg.eigh(np.cov((p - com).T))
    R = V[:, ::-1]
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    ref, fit = dict(ref), dict(fit)
    ref["pos"] = (ref["pos"] - com) @ R
    fit["pos"] = (fit["pos"] - com) @ R
    return ref, fit


def _plotter():
    p = pv.Plotter(off_screen=True, window_size=(VIEW, VIEW))
    p.set_background("white")
    p.enable_parallel_projection()
    d = VIEW_DIR / np.linalg.norm(VIEW_DIR)
    p.camera_position = [tuple(d * 50), (0, 0, 0), (0, 1, 0)]
    p.camera.parallel_scale = PARALLEL_SCALE
    p.enable_depth_peeling(16)
    try:
        p.enable_ssao(radius=1.2, bias=0.01)
    except Exception:
        pass
    p.enable_anti_aliasing("ssaa")
    return p


def _spheres(p, mol, color, opacity, rscale):
    poly = pv.PolyData(mol["pos"])
    poly["r"] = np.array([VDW.get(int(z), 1.6) for z in mol["Z"]]) * rscale
    g = poly.glyph(geom=pv.Sphere(radius=1.0, theta_resolution=40, phi_resolution=40),
                   scale="r", orient=False)
    p.add_mesh(g, color=color, opacity=opacity, **MAT)


def _sticks(p, mol, carbon_color, hetero_cpk=True):
    pos, Z, bonds = mol["pos"], mol["Z"], mol["bonds"]

    def scol(z):
        if int(z) == 6 or not hetero_cpk:
            return carbon_color
        return CPK.get(int(z), carbon_color)
    for i, j in bonds:
        a, b = pos[i], pos[j]
        mid = (a + b) / 2.0
        p.add_mesh(pv.Line(a, mid).tube(radius=STICK_R, n_sides=24, capping=True),
                   color=scol(Z[i]), **STICK_MAT)
        p.add_mesh(pv.Line(mid, b).tube(radius=STICK_R, n_sides=24, capping=True),
                   color=scol(Z[j]), **STICK_MAT)
    for xyz, z in zip(pos, Z):
        p.add_mesh(pv.Sphere(radius=STICK_R, center=xyz, theta_resolution=24,
                             phi_resolution=24), color=scol(z), **STICK_MAT)


def render_volume(ref, fit, out):
    p = _plotter()
    _spheres(p, ref, REF_HUE, opacity=0.55, rscale=VOL_RSCALE)
    _spheres(p, fit, FIT_HUE, opacity=0.55, rscale=VOL_RSCALE)
    p.screenshot(out, transparent_background=True)
    p.close()
    print("    wrote", os.path.relpath(out, HERE))


def render_stick(ref, fit, out):
    p = _plotter()
    _sticks(p, ref, REF_HUE, hetero_cpk=False)
    _sticks(p, fit, FIT_HUE, hetero_cpk=False)
    p.screenshot(out, transparent_background=True)
    p.close()
    print("    wrote", os.path.relpath(out, HERE))


def main():
    os.makedirs(DATA, exist_ok=True)
    np.random.seed(SEED)
    print(f"embedding {REF[0]} (reference) ...")
    ref0 = embed(REF[1])
    for name, smi in FITS.items():
        print(f"pair: {REF[0]} + {name}")
        fit0 = embed(smi)
        fit_al = align_fit_onto_ref(ref0, fit0)
        ref, fit = joint_orient(ref0, fit_al)
        render_volume(ref, fit, os.path.join(DATA, f"overlay_{name}_volume.png"))
        render_stick(ref, fit, os.path.join(DATA, f"overlay_{name}_stick.png"))


if __name__ == "__main__":
    main()
