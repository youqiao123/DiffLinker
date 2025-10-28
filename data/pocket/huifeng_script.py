from rdkit import Chem
import pandas as pd
from tqdm import tqdm
from rdkit.Chem.rdMMPA import FragmentMol
from collections import defaultdict, Counter
from rdkit.Chem.BRICS import BRICSDecompose,FindBRICSBonds
import os
from rdkit import Chem, Geometry
import numpy as np


def merge_two_mols(mol1, mol2):
    # 创建一个可变的分子对象，开始时将mol1中的原子和键添加到combined_mol中
    combined_mol = Chem.CombineMols(mol1,mol2)
    # atom_count = mol1.GetNumAtoms()
    
    # # 将mol2中的原子和键添加到合并的分子中
    # for atom in mol2.GetAtoms():
    #     combined_mol.AddAtom(atom)
    
    # for bond in mol2.GetBonds():
    #     begin_idx = bond.GetBeginAtomIdx() + atom_count
    #     end_idx = bond.GetEndAtomIdx() + atom_count
    #     combined_mol.AddBond(begin_idx, end_idx, bond.GetBondType())
    
    return combined_mol

def recursive_merge(mol_list):
    # 递归基：如果只有一个分子，返回该分子
    if len(mol_list) == 1:
        return mol_list[0]
    
    # 每次递归合并前两个分子
    mol1 = mol_list[0]
    mol2 = mol_list[1]
    
    # 合并这两个分子
    merged_mol = merge_two_mols(mol1, mol2)
    
    # 对剩余的分子进行递归合并
    return recursive_merge([merged_mol] + mol_list[2:])

def check_fragmentation(mol, bond_idxs, level):
    add_dummies = False if level == 1 else True
    frags = Chem.FragmentOnBonds(mol, bond_idxs, addDummies=add_dummies)
    frags = Chem.GetMolFrags(frags, asMols=True)
    if level == 1:
        frag_matrix = [f.GetNumAtoms() for f in frags]
        return True, {1:frag_matrix}, None, None
    else:
        frag_matrix = defaultdict(list)
        frag_smiles = defaultdict(list)
        frag_mols = defaultdict(list)
        for f in frags:
            f_smiles = Chem.MolToSmiles(f)
            anchor_nums = f_smiles.count('*')
            frag_matrix[anchor_nums].append(f.GetNumAtoms() - anchor_nums)
            frag_smiles[anchor_nums].append(f_smiles)
            frag_mols[anchor_nums].append(f)
        valid = (
            len(frag_matrix) == 2 and
            sorted(frag_matrix.keys()) == [1, level] and
            len(frag_matrix[1]) == level and
            len(frag_matrix[level]) == 1
        )
        return valid, frag_matrix, frag_smiles, frag_mols
    
def check_mmpa_linker(linker_smi, min_size):
    mol = Chem.MolFromSmiles(linker_smi)
    num_exits = linker_smi.count('*:')
    return (mol.GetNumAtoms() - num_exits) >= min_size


def check_mmpa_fragment(fragment_smi, min_size):
    mol = Chem.MolFromSmiles(fragment_smi)
    num_exits = fragment_smi.count('*:')
    return (mol.GetNumAtoms() - num_exits) >= min_size


def check_mmpa_fragments(fragments_smi, min_size):
    for fragment_smi in fragments_smi.split('.'):
        if not check_mmpa_fragment(fragment_smi, min_size):
            return False
    return True
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
    filtered_mmpa_mols = []
    for linker_smiles, fragments_smiles in mmpa_results:
        if check_mmpa_linker(linker_smiles, min_link_size) and check_mmpa_fragments(fragments_smiles, min_frag_size):
            filtered_mmpa_results.append([mol_name, mol_smiles, linker_smiles, fragments_smiles, 'mmpa'])
            filtered_mmpa_mols.append([None])
    return filtered_mmpa_results, filtered_mmpa_mols
    
def fragment_by_brics(mol, mol_name, start, level, current_bond_idxs, min_frag, min_linker, num_frags_min, num_frags_max, bonds):
    bonds_nums = len(bonds)
    filtered_brics_results = []
    filtered_brics_mols = [] # linker,fragments
    for i in range(start, bonds_nums):
        bond_idx = mol.GetBondBetweenAtoms(bonds[i][0], bonds[i][1]).GetIdx()
        current_bond_idxs = current_bond_idxs[:level-1] + [bond_idx]
        valid, frag_matrix, frag_smiles, frag_mols = check_fragmentation(mol, current_bond_idxs, level)
        if valid:
            if (min(frag_matrix[1]) >= min_frag and min(frag_matrix[level]) >= min_linker):
                if level >= num_frags_min - 1:
                    # print('fragment:', frag_smiles[1],'\tlinker:', frag_smiles[level])
                    filtered_brics_results.append([mol_name, Chem.MolToSmiles(mol), frag_smiles[level][0], '.'.join(frag_smiles[1]), 'brics'])
                    filtered_brics_mols.append([frag_mols[level][0], recursive_merge(frag_mols[1])])
                if level < num_frags_max - 1:
                    level += 1
                    filtered_brics_results_, filtered_brics_mols_ = fragment_by_brics(mol, mol_name, i+1,level,current_bond_idxs,min_frag, min_linker, num_frags_min, num_frags_max, bonds)
                    level -= 1
                    filtered_brics_results += filtered_brics_results_
                    filtered_brics_mols += filtered_brics_mols_
    return filtered_brics_results, filtered_brics_mols

def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer

def transfer_conformers(frag, mol):
    """
    Computes coordinates from molecule to fragment (for all matchings)
    """
    matches = mol.GetSubstructMatches(frag)
    if len(matches) < 1:
        raise Exception('Could not find fragment or linker matches')

    match2conf = {}
    for match in matches:
        mol_coords = mol.GetConformer().GetPositions()
        frag_coords = mol_coords[np.array(match)]
        frag_conformer = create_conformer(frag_coords)
        match2conf[match] = frag_conformer

    return match2conf

import itertools

def find_non_intersecting_matches(matches):
    """
    Checks all possible combinations and selects only non-intersecting matches
    """
    combinations = list(itertools.product(*matches))[:10000]
    non_intersecting_matches = set()
    for combination in combinations:
        all_idx = []
        for match in combination:
            all_idx += match
        if len(all_idx) == len(set(all_idx)):
            non_intersecting_matches.add(combination)

    return list(non_intersecting_matches)

def find_matches_with_linker_in_the_middle(non_intersecting_matches, num_frags, mol):
    """
    Selects only matches where linker is between fragments
    I.e. each linker should have at least two atoms with neighbors belonging to fragments
    """
    matches_with_linker_in_the_middle = []
    for m in non_intersecting_matches:
        fragments_m = m[:num_frags]
        linkers_m = m[num_frags:]

        all_frag_atoms = set()
        for frag_match in fragments_m:
            all_frag_atoms |= set(frag_match)

        all_linkers_are_in_the_middle = True
        for linker_match in linkers_m:
            linker_neighbors = set()
            for atom_idx in linker_match:
                atom_neighbors = mol.GetAtomWithIdx(atom_idx).GetNeighbors()
                for neighbor in atom_neighbors:
                    linker_neighbors.add(neighbor.GetIdx())

            number_of_connections = len(linker_neighbors & all_frag_atoms)
            if number_of_connections < 2:
                all_linkers_are_in_the_middle = False
                break

        if all_linkers_are_in_the_middle:
            matches_with_linker_in_the_middle.append(m)

    return matches_with_linker_in_the_middle

def find_correct_match(list_of_match_frag, list_of_match_linker, mol):
    """
    Finds all correct fragments and linker matches
    """
    non_intersecting_matches = find_non_intersecting_matches(list_of_match_frag + list_of_match_linker)
    if len(non_intersecting_matches) == 1:
        frag_match = non_intersecting_matches[0][:len(list_of_match_frag)]
        link_match = non_intersecting_matches[0][len(list_of_match_frag):]
        return frag_match, link_match

    matches_with_linker_in_the_middle = find_matches_with_linker_in_the_middle(
        non_intersecting_matches=non_intersecting_matches,
        num_frags=len(list_of_match_frag),
        mol=mol,
    )
    frag_match = matches_with_linker_in_the_middle[0][:len(list_of_match_frag)]
    link_match = matches_with_linker_in_the_middle[0][len(list_of_match_frag):]
    return frag_match, link_match

def get_exits(mol):
    """
    Returns atoms marked as exits in DeLinker data
    """
    exits = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol == '*':
            exits.append(atom)
    return exits


def update_linker(linker):
    """
    Removes exit atoms with corresponding bonds
    """
    exits = get_exits(linker)

    # Sort exit atoms by id for further correct deletion
    exits = sorted(exits, key=lambda e: e.GetIdx(), reverse=True)
    elinker = Chem.EditableMol(linker)

    # Remove exit bonds
    for exit in exits:
        bonds = exit.GetBonds()
        if len(bonds) > 1:
            raise Exception('Exit atom has more than 1 bond')
        bond = bonds[0]
        source_idx = bond.GetBeginAtomIdx()
        target_idx = bond.GetEndAtomIdx()
        elinker.RemoveBond(source_idx, target_idx)

    # Remove exit atoms
    for exit in exits:
        elinker.RemoveAtom(exit.GetIdx())

    return elinker.GetMol()

def set_anchor_flags(mol, anchor_idx):
    """
    Sets property _Anchor to all atoms in a molecule
    """
    for atom in mol.GetAtoms():
        if atom.GetIdx() == anchor_idx:
            atom.SetProp('_Anchor', '1')
        else:
            atom.SetProp('_Anchor', '0')
            
def update_fragment(frag):
    """
    Removes exit atoms with corresponding bonds and sets _Anchor property
    """
    exits = get_exits(frag)
    if len(exits) > 1:
        raise Exception('Found more than one exits in fragment')
    exit = exits[0]

    bonds = exit.GetBonds()
    if len(bonds) > 1:
        raise Exception('Exit atom has more than 1 bond')
    bond = bonds[0]

    exit_idx = exit.GetIdx()
    source_idx = bond.GetBeginAtomIdx()
    target_idx = bond.GetEndAtomIdx()
    anchor_idx = source_idx if target_idx == exit_idx else target_idx
    set_anchor_flags(frag, anchor_idx)

    efragment = Chem.EditableMol(frag)
    efragment.RemoveBond(source_idx, target_idx)
    efragment.RemoveAtom(exit_idx)

    return efragment.GetMol()

def get_anchors_idx(mol):
    anchors_idx = []
    for atom in mol.GetAtoms():
        if atom.GetProp('_Anchor') == '1':
            anchors_idx.append(atom.GetIdx())

    return anchors_idx

def prepare_fragments_and_linker(frags_smi, linker_smi, mol):
    """
    Given a molecule and SMILES string of fragments from DeLinker data,
    creates fragment and linker conformers according to the molecule conformer,
    removes exit atoms and sets _Anchor property to all fragment atoms
    """
    frags = [Chem.MolFromSmiles(smi) for smi in frags_smi.split('.')]
    linkers = [Chem.MolFromSmiles(smi) for smi in linker_smi.split('.')]
    new_frags = [update_fragment(mol) for mol in frags]
    new_linkers = [update_linker(mol) for mol in linkers]
    
    list_of_match2conf_frag = []
    list_of_match_frag = []
    for frag in new_frags:
        match2conf_frag = transfer_conformers(frag, mol)
        list_of_match2conf_frag.append(match2conf_frag)
        list_of_match_frag.append(list(match2conf_frag.keys()))

    list_of_match2conf_linker = []
    list_of_match_linker = []
    for linker in new_linkers:
        match2conf_linker = transfer_conformers(linker, mol)
        list_of_match2conf_linker.append(match2conf_linker)
        list_of_match_linker.append(list(match2conf_linker.keys()))
        
    frag_matches, link_matches = find_correct_match(list_of_match_frag, list_of_match_linker, mol)
    
    final_frag_mols = []
    for frag, frag_match, match2conf_frag in zip(new_frags, frag_matches, list_of_match2conf_frag):
        conformer = match2conf_frag[frag_match]
        frag.AddConformer(conformer)
        final_frag_mols.append(frag)

    final_link_mols = []
    for link, link_match, match2conf_link in zip(new_linkers, link_matches, list_of_match2conf_linker):
        conformer = match2conf_link[link_match]
        link.AddConformer(conformer)
        final_link_mols.append(link)

    return final_frag_mols, final_link_mols, frag_matches, link_matches

def find_duplicate_indices(lst):
    # 统计元素出现的次数
    counts = Counter(lst)
    
    # 找到重复的元素（出现次数大于1）
    duplicates = [item for item, count in counts.items() if count > 1]
    
    # 查找重复元素的索引
    duplicate_indices = {}
    for item in duplicates:
        duplicate_indices[item] = [i for i, x in enumerate(lst) if x == item]
    
    return duplicate_indices

def get_immediate_subdirectories(dir_path):
    """
    获取指定目录下的直接子文件夹名称列表
    参数:
        dir_path (str): 要扫描的文件夹路径
    返回:
        list: 直接子文件夹名称列表
    """
    try:
        # 获取指定路径下的所有条目（文件和文件夹）
        entries = os.listdir(dir_path)
        # 筛选出其中是文件夹的条目
        subdirectories = [entry for entry in entries 
                         if os.path.isdir(os.path.join(dir_path, entry))]
        return subdirectories
    except FileNotFoundError:
        print(f"错误：目录 '{dir_path}' 不存在。")
        return []
    except PermissionError:
        print(f"错误：没有权限访问目录 '{dir_path}'。")
        return []

def get_immediate_subdirectories(dir_path):
    """
    获取指定目录下的直接子文件夹名称列表
    参数:
        dir_path (str): 要扫描的文件夹路径
    返回:
        list: 直接子文件夹名称列表
    """
    try:
        # 获取指定路径下的所有条目（文件和文件夹）
        entries = os.listdir(dir_path)
        # 筛选出其中是文件夹的条目
        subdirectories = [entry for entry in entries 
                         if os.path.isdir(os.path.join(dir_path, entry))]
        return subdirectories
    except FileNotFoundError:
        print(f"错误：目录 '{dir_path}' 不存在。")
        return []
    except PermissionError:
        print(f"错误：没有权限访问目录 '{dir_path}'。")
        return []

output_table = os.path.join('/home/qianyouqiao/pdbbind_processed','generated_splits_hf.csv')
output_table_all = os.path.join('/home/qianyouqiao/pdbbind_processed','train_tables_hf.csv')
output_conformers = os.path.join('/home/qianyouqiao/pdbbind_processed','generated_conformers_hf.sdf')
mol_results = []
mol_conformers = []
out_table_flitered_all = []

opted = get_immediate_subdirectories('/home/qianyouqiao/sc_complexes')
print(len(opted))

for pdb in tqdm(opted):
    res = []
    cons = []
    if pdb+'_pocket_ligH12A.pdb' in os.listdir(os.path.join('/home/qianyouqiao/sc_complexes',pdb)):
        try:
            pdb_sdf = os.path.join('/home/qianyouqiao/sc_complexes',pdb, pdb+'_ligand.sdf')
            mol = Chem.MolFromMolFile(pdb_sdf)
            mol = Chem.RemoveAllHs(mol)
            Chem.SanitizeMol(mol)
        except:
            print(pdb)
            continue
        if mol is None:
            print(pdb)
            continue
        if mol.GetNumAtoms() <= 60 and mol.GetRingInfo().NumRings() >= 2:
            try:
                res_,cons_ = fragment_by_mmpa(mol, pdb, Chem.MolToSmiles(mol), min_cuts=2, max_cuts=4, min_link_size=3, min_frag_size=5)
                res+= res_
                cons+= cons_
            except Exception as e:
                print(f'Error [MMPA] with {pdb}: {e}')
            
        try:
            bonds = [bond[0] for bond in FindBRICSBonds(mol)]
            res_, cons_ = fragment_by_brics(mol, pdb, 0,1,[],min_frag=5,min_linker=3,num_frags_min=3,num_frags_max=5,bonds=bonds)
            res+= res_
            cons+= cons_
        except Exception as e:
            print(f'Error [BRICS] with {pdb}: {e}')
        
        if len(res) > 0:
            mol_results += res
            mol_conformers += cons
            # mol.SetProp('_Name', pdb)
            # conformers.append(mol)
            fragments = []
            linkers = []
            out_table = []
            mols_smi = []
            for idx, r in enumerate(res):
                molecule_name, molecule_smi, linker_smi, fragments_smi, method = r
                try:
                    frags, linker, frag_matches, linker_matches = prepare_fragments_and_linker(fragments_smi, linker_smi, mol)
                except Exception as e:
                    print(f'{molecule_name} | {linker_smi} | {fragments_smi} : {e}')
                    continue
                combined_frag = recursive_merge(frags)
                anchors_idx = get_anchors_idx(combined_frag)
                combined_linker = recursive_merge(linker)
                frag_ids = [str(j) for i in frag_matches for j in i ]
                linker_ids = [str(j) for i in linker_matches for j in i ]
                fragments.append(combined_frag)
                linkers.append(combined_linker)
                mols_smi.append('.'.join([Chem.MolToSmiles(combined_linker,canonical=True),Chem.MolToSmiles(combined_frag,canonical=True)]))
                out_table.append({
                    'mol_name': pdb,
                    'molecule': Chem.MolToSmiles(mol),
                    'fragments': Chem.MolToSmiles(combined_frag),
                    'linker': Chem.MolToSmiles(combined_linker),
                    'anchors': '-'.join(map(str, anchors_idx)),
                    'frag_ids':'-'.join(frag_ids),
                    'linker_ids':'-'.join(linker_ids),
                })
            drop_index = [j for i in find_duplicate_indices(mols_smi).values() for j in i[1:]]
            fragments_flitered = [i for idx,i in enumerate(fragments) if idx not in drop_index]
            linkers_flitered = [i for idx,i in enumerate(linkers) if idx not in drop_index]
            out_table_flitered = [i for idx,i in enumerate(out_table) if idx not in drop_index]
            out_table_flitered_all += out_table_flitered
            # with Chem.SDWriter(open('f.sdf', 'w')) as writer:
            #     writer.SetKekulize(False)
            #     for frags in fragments:
            #         writer.write(frags)
    
table = pd.DataFrame(mol_results, columns=['molecule_name', 'molecule', 'linker', 'fragments', 'method'])
table = table.drop_duplicates(['molecule_name', 'molecule', 'linker'])
table.to_csv(output_table, index=False)
pd.DataFrame(out_table_flitered_all).to_csv(output_table_all, index=False)
# with Chem.SDWriter(open(output_conformers, 'w')) as writer:
#     for mol in conformers:
#         writer.write(mol)