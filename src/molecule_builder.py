import torch
import numpy as np
from openbabel import openbabel as ob
from rdkit import Chem, Geometry

from src import const


def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer


def build_molecules(one_hot, x, node_mask, is_geom, margins=const.MARGINS_EDM):
    molecules = []
    for i in range(len(one_hot)):
        mask = node_mask[i].squeeze() == 1
        atom_types = one_hot[i][mask].argmax(dim=1).detach().cpu()
        positions = x[i][mask].detach().cpu()
        mol = build_molecule(positions, atom_types, is_geom, margins=margins)
        molecules.append(mol)

    return molecules


def build_molecule(positions, atom_types, is_geom, margins=const.MARGINS_EDM):
    idx2atom = const.GEOM_IDX2ATOM if is_geom else const.IDX2ATOM
    X, A, E = build_xae_molecule(positions, atom_types, is_geom=is_geom, margins=margins)
    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(idx2atom[atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), const.BOND_DICT[E[bond[0], bond[1]].item()])

    mol.AddConformer(create_conformer(positions.detach().cpu().numpy().astype(np.float64)))
    return mol


def build_xae_molecule(positions, atom_types, is_geom, margins=const.MARGINS_EDM):
    """ Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3  (already masked to keep final number nodes)
        atom_types: N
        returns:
        X: N         (int)
        A: N x N     (bool) (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)

    idx2atom = const.GEOM_IDX2ATOM if is_geom else const.IDX2ATOM

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    for i in range(n):
        for j in range(i):

            pair = sorted([atom_types[i], atom_types[j]])
            order = get_bond_order(idx2atom[pair[0].item()], idx2atom[pair[1].item()], dists[i, j], margins=margins)

            # TODO: a batched version of get_bond_order to avoid the for loop
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order

    return X, A, E


def get_bond_order(atom1, atom2, distance, check_exists=True, margins=const.MARGINS_EDM):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in const.BONDS_1:
            return 0
        if atom2 not in const.BONDS_1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of the QM9 true samples
    if distance < const.BONDS_1[atom1][atom2] + margins[0]:

        # Check if atoms in bonds2 dictionary.
        if atom1 in const.BONDS_2 and atom2 in const.BONDS_2[atom1]:
            thr_bond2 = const.BONDS_2[atom1][atom2] + margins[1]
            if distance < thr_bond2:
                if atom1 in const.BONDS_3 and atom2 in const.BONDS_3[atom1]:
                    thr_bond3 = const.BONDS_3[atom1][atom2] + margins[2]
                    if distance < thr_bond3:
                        return 3  # Triple
                return 2  # Double
        return 1  # Single
    return 0  # No bond

def build_molecules_obabel(one_hot, positions, node_mask, is_geom=True):
    if is_geom:
        IDX2ATOMICNUM = {0: 6, 1: 8, 2: 7, 3: 9, 4: 16, 5: 17, 6: 35, 7: 53, 8: 15}
    molecules = []
    for i in range(len(one_hot)):
        mask = node_mask[i].squeeze() == 1
        atom_types = one_hot[i][mask].argmax(dim=1).detach().cpu()
        atom_positions = positions[i][mask].detach().cpu().numpy()

        obmol = ob.OBMol()
        for a, (x,y,z) in zip(atom_types, atom_positions):
            atom = obmol.NewAtom()
            atom.SetAtomicNum(IDX2ATOMICNUM[a.item()])
            atom.SetVector(x.item(), y.item(), z.item())  # 设置原子坐标
        
        obmol.ConnectTheDots()
        obmol.PerceiveBondOrders()

        rdkit_mol = obmol_to_rdkit_mol(obmol, keep_hydrogens=False, embed3d=True) 
        molecules.append(rdkit_mol)
    return molecules

def obmol_to_rdkit_mol(obmol, keep_hydrogens=False, embed3d=True):
    # 1. 构建一个空 RWMol
    rdkit_mol = Chem.RWMol()

    # 2. 原子映射：记录 OBMol 原子索引 -> RDKit atom idx
    ob2rd = {}  # { ob_idx : rdkit_idx }

    # 3. 遍历原子，添加到 RDKit 中
    for ob_atom in ob.OBMolAtomIter(obmol):
        atomic_num = ob_atom.GetAtomicNum()
        if (not keep_hydrogens) and atomic_num == 1:
            continue
        a = Chem.Atom(atomic_num)
        rd_idx = rdkit_mol.AddAtom(a)
        ob2rd[ob_atom.GetIdx()] = rd_idx

    # 4. 遍历 bond，添加到 RDKit 中
    for ob_bond in ob.OBMolBondIter(obmol):
        a1 = ob_bond.GetBeginAtomIdx()
        a2 = ob_bond.GetEndAtomIdx()
        if a1 not in ob2rd or a2 not in ob2rd:
            continue
        order = ob_bond.GetBondOrder()
        # map OpenBabel bond order to RDKit bond type
        if order == 1:
            bt = Chem.BondType.SINGLE
        elif order == 2:
            bt = Chem.BondType.DOUBLE
        elif order == 3:
            bt = Chem.BondType.TRIPLE
        else:
            # 对于芳香键 (aromatic)，以及不常见键，可尝试设为 SINGLE + later kekulize/aromatic
            bt = Chem.BondType.SINGLE
        rdkit_mol.AddBond(ob2rd[a1], ob2rd[a2], bt)

    # 5. 可选：把 OBMol 的 3D 坐标 copy 到 RDKit
    if embed3d:
        conf = Chem.Conformer(rdkit_mol.GetNumAtoms())
        for ob_atom in ob.OBMolAtomIter(obmol):
            idx = ob2rd.get(ob_atom.GetIdx(), None)
            if idx is None:
                continue
            x, y, z = ob_atom.GetX(), ob_atom.GetY(), ob_atom.GetZ()
            conf.SetAtomPosition(idx, Chem.rdGeometry.Point3D(x, y, z))
        rdkit_mol.AddConformer(conf)

    # 6. Sanitize / finalize
    rdkit_mol = rdkit_mol.GetMol()
    # Chem.SanitizeMol(rdkit_mol)

    return rdkit_mol