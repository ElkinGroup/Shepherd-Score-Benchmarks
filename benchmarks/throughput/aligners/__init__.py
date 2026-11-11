"""Aligner registry. To add a tool: write aligners/<tool>.py subclassing base.Aligner, add it to
_REGISTRY, and run with --tool <tool> in that tool's environment. Adapters are imported lazily
because the tool packages cannot co-exist in one interpreter."""
from __future__ import annotations

import importlib

from .base import MODES, Aligner

# tool id -> "submodule:ClassName"
_REGISTRY = {
    "fss": "fss:FSSAligner",
    "roshambo2": "roshambo2:Roshambo2Aligner",   # GPU shape+color (env: roshambo2)
    "pharao": "pharao:PharaoAligner",            # CPU pharmacophore align_it (env: align_it)
    "phesa": "phesa:PheSAAligner",               # CPU shape+pharmacophore PheSA/openchemlib (Java)
    "shaep": "shaep:ShaepAligner",               # CPU shape+ESP
    "rdkit": "rdkit_shape:RdkitShapeAligner",    # CPU shape / shape+color: rdkit.Chem.rdShapeAlign (env: fss)
    "espsim": "espsim_tool:EspSimAligner",       # CPU shape+ESP: hesther/espsim (env: fss + COMPARATORS_HOME/python)
}


def get_aligner(tool: str, **kwargs) -> Aligner:
    if tool not in _REGISTRY:
        raise KeyError(f"unknown tool {tool!r}; known: {sorted(_REGISTRY)}")
    modname, clsname = _REGISTRY[tool].split(":")
    mod = importlib.import_module(f"{__name__}.{modname}")
    return getattr(mod, clsname)(**kwargs)


__all__ = ["MODES", "Aligner", "get_aligner"]
