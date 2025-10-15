import argparse
import os
import sys
from rdkit import Chem
import pandas as pd


def make_test_smi(table_path, out_path, mol_col='molecule', frag_col='fragments'):
    if not os.path.exists(table_path):
        raise FileNotFoundError(f"Table file not found: {table_path}")
    df = pd.read_csv(table_path)
    if mol_col not in df.columns or frag_col not in df.columns:
        raise KeyError(f"Expected columns '{mol_col}' and '{frag_col}' in {table_path}. Found: {list(df.columns)}")
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as fw:
        for _, r in df.iterrows():
            fw.write(f"{r[mol_col]} {r[frag_col]}\n")
    print(f"Wrote test smiles to {out_path}")


def make_train_linkers(link_sdf_path, out_path):
    if not os.path.exists(link_sdf_path):
        raise FileNotFoundError(f"Link SDF not found: {link_sdf_path}")
    suppl = Chem.SDMolSupplier(link_sdf_path, sanitize=False)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as fw:
        for m in suppl:
            if m is None:
                continue
            try:
                s = Chem.MolToSmiles(m)
            except Exception:
                s = Chem.MolToSmiles(m, canonical=False)
            fw.write(s + '\n')
    print(f"Wrote train linkers SMILES to {out_path}")


def make_uuids(samples_dir, dataset_prefix, out_dir):
    # samples_dir should be e.g. samples/pdbbind_test.full/pdbbind_1
    if not os.path.exists(samples_dir):
        raise FileNotFoundError(f"samples_dir not found: {samples_dir}")
    names = [n for n in os.listdir(samples_dir) if n.isdigit()]
    if len(names) == 0:
        # try to be more permissive: pick dirs that start with digits
        cand = [n for n in os.listdir(samples_dir) if os.path.isdir(os.path.join(samples_dir, n))]
        names = [n for n in cand if n.split('_')[0].isdigit()]
    ids = sorted([int(n.split('_')[0]) if '_' in n else int(n) for n in names])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, dataset_prefix.split('.')[0], 'uuids.txt') if '.' in dataset_prefix else os.path.join(out_dir, dataset_prefix, 'uuids.txt')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as fw:
        for i in ids:
            fw.write(f"{i}\n")
    print(f"Wrote uuids to {out_path} (count={len(ids)})")


def main():
    parser = argparse.ArgumentParser(description='Prepare .smi and uuids.txt files')
    sub = parser.add_subparsers(dest='cmd')

    p1 = sub.add_parser('make_smi')
    p1.add_argument('--table', required=True)
    p1.add_argument('--out', required=True)
    p1.add_argument('--mol_col', default='molecule')
    p1.add_argument('--frag_col', default='fragments')

    p2 = sub.add_parser('make_train_linkers')
    p2.add_argument('--link_sdf', required=True)
    p2.add_argument('--out', required=True)

    p3 = sub.add_parser('make_uuids')
    p3.add_argument('--samples_dir', required=True, help='Path to samples/<dataset>/<checkpoint> directory')
    p3.add_argument('--dataset_prefix', required=True, help='Dataset prefix used in sampling e.g. pdbbind_test.full')
    p3.add_argument('--out_dir', required=True, help='Formatted output dir, e.g. formatted')

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        sys.exit(1)

    if args.cmd == 'make_smi':
        make_test_smi(args.table, args.out, mol_col=args.mol_col, frag_col=args.frag_col)
    elif args.cmd == 'make_train_linkers':
        make_train_linkers(args.link_sdf, args.out)
    elif args.cmd == 'make_uuids':
        make_uuids(args.samples_dir, args.dataset_prefix, args.out_dir)


if __name__ == '__main__':
    main()
