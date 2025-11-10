import sys
sys.path.append('../../')

import argparse
import os
import pandas as pd

from data.geom.generate_geom_multifrag import check_mmpa_linker, check_mmpa_fragments
from src.utils import disable_rdkit_logging
 
from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem.BRICS import FindBRICSBonds
from tqdm import tqdm

from itertools import combinations

def _count_dummy_atoms(m):
    """统计分子中 dummy 原子个数 AtomicNum == 0"""
    return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 0)

def check_linker_generic(linker_smi, min_size):
    mol = Chem.MolFromSmiles(linker_smi)
    if mol is None:
        return False
    num_exits = _count_dummy_atoms(mol)
    return (mol.GetNumAtoms() - num_exits) >= min_size

def check_fragment_generic(fragment_smi, min_size):
    mol = Chem.MolFromSmiles(fragment_smi)
    if mol is None:
        return False
    num_exits = _count_dummy_atoms(mol)
    return (mol.GetNumAtoms() - num_exits) >= min_size

def check_fragments_generic(fragments_smi, min_size):
    return all(check_fragment_generic(smi, min_size) for smi in fragments_smi.split('.'))

def fragment_by_brics(mol, mol_name, mol_smiles,
                      min_cuts: int, max_cuts: int,
                      min_frag_size: int, min_link_size: int):
    """
    返回结构：[[mol_name, mol_smiles, linker_smiles, fragments_smiles, 'brics'], ...]
    与 fragment_by_mmpa 保持一致。
    """

    # 1) 收集 BRICS 候选断键（原子对 -> bond_idx）
    atom_pairs = list(FindBRICSBonds(mol))  
    bond_indices = []
    for (ai, aj), _ in atom_pairs:
        b = mol.GetBondBetweenAtoms(int(ai), int(aj))
        if b is not None:
            bond_indices.append(b.GetIdx())

    results = []
    seen = set()  # 去重：存储 (linker_smiles, sorted_fragments_smiles_str)

    # 2) 在给定切割数范围内做组合枚举
    for k in range(min_cuts, max_cuts + 1):
        if k <= 0:
            continue
        if k > len(bond_indices):
            break

        for combo in combinations(bond_indices, k):
            # 3) 切割并获取碎片
            fragmol = Chem.FragmentOnBonds(mol, list(combo), addDummies=True)
            frags = Chem.GetMolFrags(fragmol, asMols=True, sanitizeFrags=True)

            # 4) 统计每个碎片的 dummy 数；分类：key = dummy_count
            by_anchor = {}
            for f in frags:
                n_anchor = _count_dummy_atoms(f)
                smi = Chem.MolToSmiles(f, isomericSmiles=True)
                # smi = _normalize_dummy_labels(smi)
                by_anchor.setdefault(n_anchor, []).append((f, smi))

            # 星形拓扑判定：必须恰好有 1 个 'k-连接位' 的 linker，且恰好有 k 个 '1-连接位' 的侧片段
            if (k in by_anchor) and (1 in by_anchor) and len(by_anchor[k]) == 1 and len(by_anchor[1]) == k:
                linker_smiles = by_anchor[k][0][1]
                frag_smiles_list = [s for _, s in by_anchor[1]]
                frag_smiles_list.sort()  # 侧片段排序，保证稳定性与去重友好
                fragments_smiles = '.'.join(frag_smiles_list)

                # 5) 与 MMPA 同口径的尺寸过滤
                if check_linker_generic(linker_smiles, min_link_size) and check_fragments_generic(fragments_smiles, min_frag_size):
                    key = (linker_smiles, fragments_smiles)
                    if key not in seen:
                        seen.add(key)
                        results.append([mol_name, mol_smiles, linker_smiles, fragments_smiles, 'brics'])

    return results

def fragment_by_mmpa(mol, mol_name, mol_smiles, min_cuts, max_cuts, min_frag_size, min_link_size):
    mmpa_results = []
    for i in range(min_cuts, max_cuts + 1):
        mmpa_results += FragmentMol(
            mol,
            minCuts=i,
            maxCuts=i,
            maxCutBonds=100,
            pattern="[#6+0;!$(*=,#[!#6])]!@!=!#[*]",
            resultsAsMols=False
        )

    filtered_mmpa_results = []
    for linker_smiles, fragments_smiles in mmpa_results:
        if check_linker_generic(linker_smiles, min_link_size) and check_fragments_generic(fragments_smiles, min_frag_size):
            filtered_mmpa_results.append([mol_name, mol_smiles, linker_smiles, fragments_smiles, 'mmpa'])
    return filtered_mmpa_results


def run(ligands_dir, output_table, output_conformers):
    # TODO 预处理 考虑到protac的linker一般比较长，我是不是可以让min_link_size大一点？
    min_frag_size = 5
    min_link_size = 3

    mol_results = []
    conformers = []
    for fname in tqdm(os.listdir(ligands_dir)):
        if fname.endswith('.mol') and not fname.startswith('._'):
            mol_name = fname.split('.')[0]
            try:
                mol = Chem.MolFromMolFile(os.path.join(ligands_dir, fname))
                mol = Chem.RemoveAllHs(mol)
                Chem.SanitizeMol(mol)
            except:
                continue
            if mol is None:
                continue

            # TODO 预处理 为什么要加这一步筛选？是怕分子量太大导致fragmentation时间过长吗？我把40改成60，2改成1
            # 加了筛选，只有8万多数据
            if mol.GetNumAtoms() <= 70 and mol.GetRingInfo().NumRings() >= 2:
            # else:
                # try:
                #     res = fragment_by_mmpa(
                #         mol,
                #         mol_smiles=Chem.MolToSmiles(mol),
                #         mol_name=mol_name,
                #         min_cuts=2,
                #         max_cuts=2,
                #         min_link_size=min_link_size,
                #         min_frag_size=min_frag_size,
                #     )
                # except:
                #     continue
                # if len(res) > 0:
                #     mol_results += res
                #     mol.SetProp('_Name', mol_name)
                #     conformers.append(mol)
            
                try:
                    res = fragment_by_brics(
                        mol,
                        mol_smiles=Chem.MolToSmiles(mol),
                        mol_name=mol_name,
                        min_cuts=2,
                        max_cuts=2,
                        min_link_size=min_link_size,
                        min_frag_size=min_frag_size,
                    )                
                    mol_results += res
                except Exception as e:
                    print(f'Error [BRICS] with {mol_name}: {e}')

                if len(res) > 0:
                    mol_results += res
                    mol.SetProp('_Name', mol_name)
                    conformers.append(mol)

    table = pd.DataFrame(mol_results, columns=['molecule_name', 'molecule', 'linker', 'fragments', 'method'])
    table = table.drop_duplicates(['molecule_name', 'molecule', 'linker'])
    table.to_csv(output_table, index=False)

    with Chem.SDWriter(open(output_conformers, 'w')) as writer:
        for mol in conformers:
            writer.write(mol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-ligands', action='store', type=str, required=True)
    parser.add_argument('--out-fragmentations', action='store', type=str, required=True)
    parser.add_argument('--out-conformers', action='store', type=str, required=True)
    args = parser.parse_args()

    disable_rdkit_logging()

    run(
        ligands_dir=args.in_ligands,
        output_table=args.out_fragmentations,
        output_conformers=args.out_conformers,
    )
