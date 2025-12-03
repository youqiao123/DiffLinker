import argparse
import os
from pathlib import Path
import pandas as pd
import subprocess

from rdkit import Chem
from src.utils import disable_rdkit_logging

from tqdm import tqdm
from pdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument('--samples', action='store', type=str, required=True)
parser.add_argument('--checkpoint', action='store', type=str, required=True)
parser.add_argument('--dataset_type', action='store', type=str, required=True)
parser.add_argument('--test_table_path', action='store', type=str, required=True)
parser.add_argument('--formatted', action='store', type=str, required=True)
parser.add_argument('--linker_size_model_name', action='store', type=str, required=False, default=None)


def load_rdkit_molecule(xyz_path, obabel_path, true_frag_smi):
    if not os.path.exists(obabel_path):
        subprocess.run(f'obabel {xyz_path} -O {obabel_path}', stderr=subprocess.DEVNULL, shell=True)

    supp = Chem.SDMolSupplier(obabel_path, sanitize=False)
    mol = list(supp)[0]

    # Keeping only the biggest connected part
    mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    mol_filtered = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
    try:
        mol_smi = Chem.MolToSmiles(mol_filtered)
    except RuntimeError:
        mol_smi = Chem.MolToSmiles(mol_filtered, canonical=False)

    # Retrieving linker
    true_frag = Chem.MolFromSmiles(true_frag_smi, sanitize=False)
    match = mol_filtered.GetSubstructMatch(true_frag)
    if len(match) == 0:
        linker_smi = ''
    else:
        elinker = Chem.EditableMol(mol_filtered)
        for atom in sorted(match, reverse=True):
            elinker.RemoveAtom(atom)
        linker = elinker.GetMol()
        Chem.Kekulize(linker, clearAromaticFlags=True)
        try:
            linker_smi = Chem.MolToSmiles(linker)
        except RuntimeError:
            linker_smi = Chem.MolToSmiles(linker, canonical=False)

    return mol_filtered, mol_smi, linker_smi


def load_molecules(folder, true_frag_smi):
    obabel_dir = f'{folder}/obabel'
    os.makedirs(obabel_dir, exist_ok=True)

    pred_mols = []
    pred_mols_smi = []
    pred_link_smi = []
    for fname in os.listdir(folder):
        number = fname.split('_')[0]
        if number.isdigit():
            pred_path = f'{folder}/{fname}'
            pred_obabel_path = f'{obabel_dir}/{number}_.sdf'
            mol, mol_smi, link_smi = load_rdkit_molecule(pred_path, pred_obabel_path, true_frag_smi)
            pred_mols.append(mol)
            pred_mols_smi.append(mol_smi)
            pred_link_smi.append(link_smi)

    return pred_mols, pred_mols_smi, pred_link_smi


def load_sampled_dataset(folder, idx2true_mol_smi, idx2true_frag_smi):
    pred_mols = []
    pred_mols_smi = []
    pred_link_smi = []
    true_mols_smi = []
    true_frags_smi = []
    uuids = []

    for fname in tqdm(os.listdir(folder)):
        if fname.isdigit():
            true_mol_smi = idx2true_mol_smi[int(fname)]
            true_frag_smi = idx2true_frag_smi[int(fname)]

            mols, mols_smi, link_smi = load_molecules(f'{folder}/{fname}', true_frag_smi)
            pred_mols += mols
            pred_mols_smi += mols_smi
            pred_link_smi += link_smi
            true_mols_smi += [true_mol_smi] * len(mols)
            true_frags_smi += [true_frag_smi] * len(mols)
            uuids += [fname] * len(mols)

    return pred_mols, pred_mols_smi, pred_link_smi, true_mols_smi, true_frags_smi, uuids

def make_uuid_file(samples_dir:Path, out_path:Path):
    # samples_dir, out_dir = Path(samples_dir), Path(out_dir)
    if not samples_dir.exists():
        raise FileNotFoundError(f"samples_dir not found: {str(samples_dir)}")

    names = [n for n in samples_dir.iterdir() if n.name.isdigit()]
    ids = sorted([int(n.name.split('_')[0]) if '_' in n.name else int(n.name) for n in names])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as fw:
        for i in ids:
            fw.write(f"{i}\n")

# def make_true_smi_file(table_path:Path, out_path:Path, mol_col='molecule', frag_col='fragments'):
#     if not table_path.exists():
#         raise FileNotFoundError(f"Table file not found: {table_path}")
#     df = pd.read_csv(str(table_path))
#     if mol_col not in df.columns or frag_col not in df.columns:
#         raise KeyError(f"Expected columns '{mol_col}' and '{frag_col}' in {table_path}. Found: {list(df.columns)}")
    
#     with open(out_path, 'w') as fw:
#         for _, r in df.iterrows():
#             fw.write(f"{r[mol_col]} {r[frag_col]}\n")
#     print(f"Wrote test smiles to {out_path}")


def reformat(samples, dataset, table_path, checkpoint, formatted, linker_size_model_name):
    samples = Path(samples)
    formatted = Path(formatted)
    # table_path = Path(table_path)
    checkpoint = Path(checkpoint).stem

    if linker_size_model_name is None:
        input_path = samples / dataset / checkpoint
        formatted_output_dir = formatted / checkpoint
    else:
        linker_size_model_name = Path(linker_size_model_name).stem
        input_path = samples / dataset / "sampled_size" / linker_size_model_name / checkpoint
        formatted_output_dir = formatted / checkpoint / "sampled_size" / linker_size_model_name

    out_smi_path = formatted_output_dir / f"{dataset}.smi"
    out_sdf_path = formatted_output_dir / f"{dataset}.sdf"

    print(f"Sampled SMILES will be saved to {out_smi_path}")
    print(f"Sampled molecules will be saved to {out_sdf_path}")

    true_smiles_table = pd.read_csv(table_path, usecols=['molecule', 'fragments'])

    dataset_lower = dataset.lower()

    if ('moad' in dataset_lower) or ('pdbbind' in dataset_lower):
        import numpy as np
        uuid_path = formatted / dataset.split('.')[0] / "uuids.txt"
        if not uuid_path.exists():
            make_uuid_file(input_path, uuid_path)

        uuids = np.loadtxt(str(uuid_path), dtype=int)
        idx2true_mol_smi = dict(zip(uuids, true_smiles_table['molecule'].values))
        idx2true_frag_smi = dict(zip(uuids, true_smiles_table['fragments'].values))
    else:
        idx2true_mol_smi = dict(enumerate(true_smiles_table.molecule.values))
        idx2true_frag_smi = dict(enumerate(true_smiles_table.fragments.values))

    pred_mols, pred_mols_smi, pred_link_smi, true_mols_smi, true_frag_smi, uuids = load_sampled_dataset(
        folder=input_path,
        idx2true_mol_smi=idx2true_mol_smi,
        idx2true_frag_smi=idx2true_frag_smi,
    )

    formatted_output_dir.mkdir(parents=True, exist_ok=True)

    with out_smi_path.open('w') as f:
        for i in range(len(pred_mols_smi)):
            f.write(
                f"{true_frag_smi[i]} {true_mols_smi[i]} "
                f"{pred_mols_smi[i]} {pred_link_smi[i]} {uuids[i]}\n"
            )

    with Chem.SDWriter(str(out_sdf_path)) as writer:
        for mol in pred_mols:
            writer.write(mol)

if __name__ == '__main__':
    disable_rdkit_logging()
    args = parser.parse_args()
    reformat(
        samples=args.samples,
        dataset=args.dataset_type,
        table_path=args.test_table_path,
        checkpoint=args.checkpoint,
        formatted=args.formatted,
        linker_size_model_name=args.linker_size_model_name,
    )
