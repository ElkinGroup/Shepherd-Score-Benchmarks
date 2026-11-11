"""Shared ESPsim preparation, alignment and ESP scoring for both benchmarks."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdMolDescriptors


def prepare(mol):
    mol = Chem.AddHs(mol, addCoords=True)
    crippen = rdMolDescriptors._CalcCrippenContribs(mol)
    properties = AllChem.MMFFGetMoleculeProperties(Chem.Mol(mol))
    if properties is None:
        AllChem.ComputeGasteigerCharges(mol)
        charges = np.array([a.GetDoubleProp('_GasteigerCharge') for a in mol.GetAtoms()])
    else:
        charges = np.array([properties.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())])
    if not np.isfinite(charges).all():
        raise ValueError('ESPsim preparation produced nonfinite charges')
    return mol, crippen, charges


def align_and_score(espsim, fit, reference):
    mol, crippen, charges = fit
    ref, ref_crippen, ref_charges = reference
    probe = Chem.Mol(mol)
    rdMolAlign.GetCrippenO3A(probe, ref, crippen, ref_crippen).Align()
    score = float(espsim.GetEspSim(probe, ref, prbCharge=charges, refCharge=ref_charges,
                                  metric='carbo', integrate='gauss', renormalize=False))
    return probe, score
