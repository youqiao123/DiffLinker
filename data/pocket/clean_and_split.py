import sys
sys.path.append('../../')

import argparse
import os
import subprocess
from pathlib import Path

from rdkit import Chem
from tqdm import tqdm
from src.utils import disable_rdkit_logging

# 这个函数是用来过滤溶剂等配体之外的小分子的，可以忽略。以及过滤大分子量的配体
def get_relevant_ligands(mol):
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    ligands = []
    for frag in frags:
        if 10 < frag.GetNumAtoms() <= 40:
            ligands.append(frag)
    return ligands

def process_pdbbind_protein(input_dir, proteins_dir):
    root = Path(input_dir)
    proteins_path = Path(proteins_dir)
    proteins_path.mkdir(parents=True, exist_ok=True) # 允许递归创建父目录

    for protein_path in tqdm(root.rglob('*_protein.pdb')):
        pdb_code = protein_path.stem[:4]

        temp_path_0 = proteins_path / f'{pdb_code}_temp_0.pdb'
        temp_path_1 = proteins_path / f'{pdb_code}_temp_1.pdb'

        out_protein_path = proteins_path / f'{pdb_code}_protein.pdb'

        subprocess.run(f'pdb_selmodel -1 {protein_path} > {temp_path_0}', shell=True)
        subprocess.run(f'pdb_delelem -H {temp_path_0} > {temp_path_1}', shell=True)
        subprocess.run(f'pdb_delhetatm {temp_path_1} > {out_protein_path}', shell=True)

        temp_path_0.unlink()
        temp_path_1.unlink()

def process_pdbbind_ligand(input_dir, ligands_dir):
    root = Path(input_dir)
    ligands_path = Path(ligands_dir)
    ligands_path.mkdir(parents=True, exist_ok=True) # 允许递归创建父目录

    for ligand_path in tqdm(root.rglob('*_ligand.sdf')):
        pdb_code = ligand_path.stem[:4]

        try:
            mol = Chem.SDMolSupplier(str(ligand_path), sanitize=False)[0]
            mol = Chem.RemoveAllHs(mol)
            rw_mol = Chem.RWMol(mol)
            atoms_to_remove = []
            # TODO 预处理 去掉卤素原子这一步有必要吗
            for atom in rw_mol.GetAtoms():
                if atom.GetSymbol() in ["F", "Cl", "Br", "I"]:
                    atoms_to_remove.append(atom.GetIdx())
            for idx in sorted(atoms_to_remove, reverse=True):
                rw_mol.RemoveAtom(idx)
            mol = rw_mol.GetMol()
        except Exception as e:
            print(f'Problem reading ligands PDB={pdb_code}: {e}')
            continue

        out_ligand_path = ligands_path / f'{pdb_code}.mol'
        Chem.MolToMolFile(mol, str(out_ligand_path), kekulize=False)


def run_pdbbind(input_dir, proteins_dir, ligands_dir):
    process_pdbbind_protein(input_dir, proteins_dir)
    process_pdbbind_ligand(input_dir, ligands_dir)
    print('PDBBind processing completed.')


def run(input_dir, proteins_dir, ligands_dir):
    os.makedirs(proteins_dir, exist_ok=True)
    os.makedirs(ligands_dir, exist_ok=True)

    for fname in tqdm(os.listdir(input_dir)):
        if fname.endswith('.bio1'):
            pdb_code = fname.split('.')[0]
            input_path = os.path.join(input_dir, fname)
            temp_path_0 = os.path.join(proteins_dir, f'{pdb_code}_temp_0.pdb')
            temp_path_1 = os.path.join(proteins_dir, f'{pdb_code}_temp_1.pdb')
            temp_path_2 = os.path.join(proteins_dir, f'{pdb_code}_temp_2.pdb')
            temp_path_3 = os.path.join(proteins_dir, f'{pdb_code}_temp_3.pdb')

            out_protein_path = os.path.join(proteins_dir, f'{pdb_code}_protein.pdb')
            out_ligands_path = os.path.join(ligands_dir, f'{pdb_code}_ligands.pdb')

            subprocess.run(f'pdb_selmodel -1 {input_path} > {temp_path_0}', shell=True)
            subprocess.run(f'pdb_delelem -H {temp_path_0} > {temp_path_1}', shell=True)
            subprocess.run(f'pdb_delhetatm {temp_path_1} > {out_protein_path}', shell=True)

            subprocess.run(f'pdb_selhetatm {temp_path_1} > {temp_path_2}', shell=True)
            subprocess.run(f'pdb_delelem -H {temp_path_2} > {temp_path_3}', shell=True)
            subprocess.run(f'pdb_delelem -X {temp_path_3} > {out_ligands_path}', shell=True)

            os.remove(temp_path_0)
            os.remove(temp_path_1)
            os.remove(temp_path_2)
            os.remove(temp_path_3)

            try:
                mol = Chem.MolFromPDBFile(out_ligands_path, sanitize=False)
                os.remove(out_ligands_path)
            except Exception as e:
                print(f'Problem reading ligands PDB={pdb_code}: {e}')
                os.remove(out_ligands_path)
                continue

            try:
                ligands = get_relevant_ligands(mol) 
            except Exception as e:
                print(f'Problem getting relevant ligands PDB={pdb_code}: {e}')
                continue

            for i, lig in enumerate(ligands):
                out_ligand_path = os.path.join(ligands_dir, f'{pdb_code}_{i}.mol')
                Chem.MolToMolFile(lig, out_ligand_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', action='store', type=str, required=True)
    parser.add_argument('--proteins-dir', action='store', type=str, required=True)
    parser.add_argument('--ligands-dir', action='store', type=str, required=True)
    args = parser.parse_args()

    disable_rdkit_logging()
    # run(input_dir=args.in_dir, proteins_dir=args.proteins_dir, ligands_dir=args.ligands_dir)
    # run_pdbbind(input_dir=args.in_dir, proteins_dir=args.proteins_dir, ligands_dir=args.ligands_dir)
    process_pdbbind_ligand(input_dir=args.in_dir, ligands_dir=args.ligands_dir)
