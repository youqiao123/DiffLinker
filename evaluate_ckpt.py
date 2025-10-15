"""High-level orchestration script for evaluating a DiffLinker checkpoint.

This script automates the prerequisite steps needed before running
``compute_metrics.py``:

* Preparing ``*.smi`` files from CSV tables of the evaluation set.
* (Optionally) preparing the training linker SMILES from an SDF file.
* Sampling molecules from a trained checkpoint via ``sample.py``.
* Creating the ``uuids.txt`` file required for MOAD / PDBBind style datasets.
* Converting sampled XYZ files into SMILES/SDF pairs with OpenBabel
  through ``reformat_data_obabel.py``.
* Finally, executing ``compute_metrics.py`` to obtain the evaluation metrics.

The goal is to provide a single entry-point for reproducing the steps described
in the README when evaluating a checkpoint.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

# TODO: 直接在这个文件里实现这三个函数，删除prepare_smi_or_uuid这个文件
from prepare_smi_or_uuid import make_test_smi, make_train_linkers, make_uuids
from reformat_data_obabel import reformat as reformat_samples
from src.utils import disable_rdkit_logging


def _run_subprocess(cmd: list[str], cwd: Optional[str] = None) -> None:
    """Run a subprocess command while streaming its output."""
    print("[evaluate] Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed with exit code {exc.returncode}: {' '.join(cmd)}") from exc


def _ensure_true_smiles(table_path: Optional[str], out_path: str, mol_col: str, frag_col: str) -> str:
    """Ensure that the evaluation SMILES file exists, creating it from a CSV if needed."""
    out_file = Path(out_path)
    if out_file.exists():
        print(f"[evaluate] Found existing evaluation SMILES: {out_file}")
        return str(out_file)

    if table_path is None:
        raise FileNotFoundError(
            f"Evaluation SMILES file not found at {out_file}. Provide --test-table to generate it or "
            "point --true-smiles-path to an existing file."
        )

    print(f"[evaluate] Preparing evaluation SMILES at {out_file} from table {table_path}")
    make_test_smi(table_path, str(out_file), mol_col=mol_col, frag_col=frag_col)
    return str(out_file)


def _ensure_train_linkers(train_linkers: Optional[str], train_linkers_sdf: Optional[str], out_path: str) -> str:
    """Ensure the training linkers SMILES is available."""
    if train_linkers and Path(train_linkers).exists():
        print(f"[evaluate] Using provided training linkers SMILES: {train_linkers}")
        return train_linkers

    if train_linkers_sdf is None:
        raise FileNotFoundError(
            "Training linker SMILES not found. Provide --train-linkers pointing to an existing file or "
            "specify --train-linkers-sdf to build it from an SDF archive."
        )

    out_file = Path(out_path)
    if out_file.exists():
        print(f"[evaluate] Found existing converted training linkers: {out_file}")
        return str(out_file)

    print(f"[evaluate] Converting training linkers SDF {train_linkers_sdf} -> {out_file}")
    make_train_linkers(train_linkers_sdf, str(out_file))
    return str(out_file)


# TODO: 这里采样的设置不是很自由，想重新采样只能把原本的采样文件夹删掉。因为没有单独指定experiment name
def _maybe_sample(args: argparse.Namespace, experiment_name: str) -> None:
    """Run sampling if requested or if outputs are missing."""
    sample_root = Path(args.samples_dir) / args.dataset_prefix / experiment_name
    if sample_root.exists() and any(sample_root.iterdir()) and not args.force_resample:
        print(f"[evaluate] Sampling outputs already present at {sample_root}. Skipping sampling.")
        return

    sample_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-W",
        "ignore",
        "sample.py",
        "--checkpoint",
        args.checkpoint,
        "--samples",
        args.samples_dir,
        "--prefix",
        args.dataset_prefix,
        "--n_samples",
        str(args.n_samples),
        "--device",
        args.device,
    ]
    if args.data_path:
        cmd += ["--data", args.data_path]
    if args.n_steps is not None:
        cmd += ["--n_steps", str(args.n_steps)]
    if args.linker_size_model:
        cmd += ["--linker_size_model", args.linker_size_model]

    _run_subprocess(cmd)


def _ensure_uuids(samples_dir: Path, dataset_prefix: str, formatted_dir: str) -> None:
    """Create the uuids.txt file required for certain datasets."""
    formatted_base = Path(formatted_dir) / (dataset_prefix.split(".")[0])
    uuid_file = formatted_base / "uuids.txt"
    if uuid_file.exists():
        print(f"[evaluate] UUID list already exists at {uuid_file}")
        return

    print(f"[evaluate] Generating UUID list at {uuid_file}")
    make_uuids(str(samples_dir), dataset_prefix, formatted_dir)


def _run_reformat(args: argparse.Namespace, experiment_name: str, true_smiles_path: str) -> str:
    """Reformat sampled molecules via OpenBabel and return the generated .smi path."""
    reformat_samples(
        samples=args.samples_dir,
        dataset=args.dataset_prefix,
        true_smiles_path=true_smiles_path,
        checkpoint=experiment_name,
        formatted=args.formatted_dir,
        linker_size_model_name=args.linker_size_model_name,
    )
    if args.linker_size_model_name is None:
        formatted_dir = Path(args.formatted_dir) / experiment_name
    else:
        formatted_dir = Path(args.formatted_dir) / experiment_name / "sampled_size" / args.linker_size_model_name
    smi_path = formatted_dir / f"{args.dataset_prefix}.smi"
    if not smi_path.exists():
        raise FileNotFoundError(f"Formatted SMILES not found at {smi_path}")
    return str(smi_path)


def _run_compute_metrics(args: argparse.Namespace, formatted_smi: str, train_linkers_path: str) -> None:
    restrict_value = str(args.restrict) if args.restrict is not None else "None"
    cmd = [
        sys.executable,
        "-W",
        "ignore",
        "compute_metrics.py",
        args.dataset_name,
        formatted_smi,
        train_linkers_path,
        str(args.n_cores),
        "True" if args.verbose else "",
        restrict_value,
        args.pains_path,
        args.method,
    ]
    _run_subprocess(cmd)


def main() -> None:
    disable_rdkit_logging()

    parser = argparse.ArgumentParser(description="Evaluate a DiffLinker checkpoint end-to-end")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained .ckpt file")
    parser.add_argument("--dataset-name", required=True, help="Name passed to compute_metrics (e.g. ZINC, CASF, pdbbind)")
    parser.add_argument("--dataset-prefix", required=True, help="Prefix used during sampling (e.g. pdbbind_test.full)")
    
    parser.add_argument("--experiment-name", default=None, help="Optional name for this evaluation run. Defaults to the checkpoint name")
    
    parser.add_argument("--samples-dir", default="samples", help="Directory to store sampling outputs")
    parser.add_argument("--formatted-dir", default="formatted", help="Directory for formatted evaluation files")
    parser.add_argument("--data-path", default=None, help="Optional override for the dataset root passed to sample.py")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of samples per molecule")
    parser.add_argument("--device", default="cuda:0", help="Torch device used during sampling")
    parser.add_argument("--n-steps", type=int, default=None, help="Optional diffusion steps override for sampling")

    parser.add_argument("--linker-size-model", default=None, help="Optional linker size classifier checkpoint")
    parser.add_argument("--linker-size-model-name", default=None, help="Name used for linker-size-aware sampling outputs")

    parser.add_argument("--test-table", default=None, help="CSV table containing columns for molecules and fragments")
    parser.add_argument("--test-table-mol-col", default="molecule", help="Column with full molecules in the test CSV")
    parser.add_argument("--test-table-frag-col", default="fragments", help="Column with fragment SMILES in the test CSV")
    parser.add_argument("--true-smiles-path", default=None, help="Existing SMILES file with true molecules/fragments")
    parser.add_argument("--train-linkers", default=None, help="Path to training linker SMILES")
    parser.add_argument("--train-linkers-sdf", default=None, help="Path to training linker SDF archive")
    parser.add_argument("--train-linkers-out", default="datasets/train_linkers_auto.smi", help="Output SMILES path when converting from SDF")
    parser.add_argument("--pains-path", default="resources/wehi_pains.csv", help="CSV with PAINS SMARTS definitions")
    parser.add_argument("--n-cores", type=int, default=8, help="Number of CPU cores for compute_metrics")
    parser.add_argument("--restrict", type=int, default=None, help="Optional restriction on the number of evaluated samples")
    parser.add_argument("--method", default="diffusion", choices=["diffusion", "3dlinker", "delinker"], help="Method argument for compute_metrics")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging in compute_metrics")
    parser.add_argument("--force-resample", action="store_true", help="Re-run sampling even if outputs already exist")

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not args.experiment_name:
        experiment_name = checkpoint_path.stem
    else:
        experiment_name = args.experiment_name

    print(f"[evaluate] Evaluating checkpoint '{experiment_name}'")

    true_smiles_path = args.true_smiles_path
    if true_smiles_path is None:
        default_true_path = Path("datasets") / f"{args.dataset_prefix.split('.')[0]}_test_smiles.smi"
        true_smiles_path = _ensure_true_smiles(args.test_table, str(default_true_path), args.test_table_mol_col, args.test_table_frag_col)
    else:
        true_smiles_path = _ensure_true_smiles(None, true_smiles_path, args.test_table_mol_col, args.test_table_frag_col)

    train_linkers_path = _ensure_train_linkers(args.train_linkers, args.train_linkers_sdf, args.train_linkers_out)

    _maybe_sample(args, experiment_name)

    samples_dir = Path(args.samples_dir) / args.dataset_prefix / experiment_name
    _ensure_uuids(samples_dir, args.dataset_prefix, args.formatted_dir)

    formatted_smi = _run_reformat(args, experiment_name, true_smiles_path)

    _run_compute_metrics(args, formatted_smi, train_linkers_path)


if __name__ == "__main__":
    main()