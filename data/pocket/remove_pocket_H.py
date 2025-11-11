from pathlib import Path
from tqdm import tqdm
import subprocess
import argparse

def process_pdbbind_protein(root_dir):
    root = Path(root_dir)

    for pocket_path in tqdm(root.rglob('*_pocket.pdb')):
        pdb_code = pocket_path.stem[:4]

        pocket_removeH_path = pocket_path.parent / f'{pdb_code}_pocket_removeH.pdb'
        temp_path_0 = pocket_path.parent / f'{pdb_code}_temp_0.pdb'
        # temp_path_1 = pocket_path.parent / f'{pdb_code}_temp_1.pdb'


        # subprocess.run(f'pdb_selmodel -1 {pocket_path} > {temp_path_0}', shell=True)
        subprocess.run(f'pdb_delelem -H {pocket_path} > {temp_path_0}', shell=True)
        subprocess.run(f'pdb_delhetatm {temp_path_0} > {pocket_removeH_path}', shell=True)

        temp_path_0.unlink()
        # temp_path_1.unlink()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default="/home/qianyouqiao/sc_complexes/", type=str)
    args = parser.parse_args()

    process_pdbbind_protein(args.root_dir)
