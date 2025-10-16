#!/usr/bin/env python3
"""
Simple launcher for Task 3 (Cell Type Classification) using the RRT model.

This script prepares a clean command for Tianyu's main.py and runs it, with
three easy presets: quick, default, and full. It keeps paths explicit and
shows exactly what will run before asking for confirmation.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


# Fixed paths on the cluster (edit if your environment differs)
TIANYU_PROJECT_PATH = Path("/cbica/home/tianyu/projects/dna-rrt-patch-only")
DATA_ROOT = Path("/gpfs/fs001/cbica/projects/Path_UPenn_GBM/tianyu_files/dataset/Penn_NIH_Combine_Features")


def build_main_cli(results_dir: Path, exp_code: str, mode: str) -> list[str]:
    """Build the argv list for Tianyu's main.py based on mode."""
    base_args = [
        "--task", "task_3_cell_type_classification",
        "--data_root_dir", str(DATA_ROOT),
        "--results_dir", str(results_dir),
        "--model_type", "rrt",
        "--subtyping",
        "--embed_dim", "1024",
        "--exp_code", exp_code,
        "--k", "5",
    ]

    if mode == "quick":
        # 1 fold, 5 epochs (no --testing to avoid upstream bug)
        base_args += [
            "--k_start", "0",
            "--k_end", "1",
            "--early_stopping",
            "--max_epochs", "5",
        ]
    elif mode == "full":
        # all 5 folds, 200 epochs
        base_args += [
            "--k_start", "0",
            "--k_end", "5",
            "--early_stopping",
            "--max_epochs", "200",
            "--lr", "2e-4",
            "--reg", "1e-5",
        ]
    else:  # default
        # first 2 folds, 50 epochs
        base_args += [
            "--k_start", "0",
            "--k_end", "2",
            "--early_stopping",
            "--max_epochs", "50",
        ]

    return base_args


def print_overview(exp_code: str, results_dir: Path, data_root: Path, mode: str, main_cli: list[str]) -> None:
    print(f"Results will be saved to: {results_dir}")
    print("\n" + "=" * 60)
    print(f"MODE: {mode.upper()}")
    print("COMMAND TO BE EXECUTED:")
    print(f"{TIANYU_PROJECT_PATH / 'main.py'} {' '.join(main_cli)}")
    print("=" * 60)
    print("\nTraining Configuration:")
    print(f"  - Experiment code: {exp_code}")
    print(f"  - Results directory: {results_dir}")
    print(f"  - Data directory: {data_root}")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  - GPU available: Yes (Device: {torch.cuda.get_device_name(0)})")
        else:
            print("  - GPU available: No (Using CPU)")
    except Exception:
        print("  - GPU check skipped (torch not available)")


def run_tianyu_main(main_cli: list[str], gpu_id: int) -> int:
    """Run Tianyu's main.py ensuring our repo takes precedence on sys.path."""
    current_dir = str(Path.cwd())
    main_py = str(TIANYU_PROJECT_PATH / "main.py")

    # Insert our repo path at the front of sys.path, set argv, then run main.py
    code = (
        "import sys, runpy; "
        f"sys.path.insert(0, r'{current_dir}'); "
        f"sys.argv = [r'{main_py}'] + {repr(main_cli)}; "
        f"runpy.run_path(r'{main_py}', run_name='__main__')"
    )

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_id))

    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )

    # Stream output live
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")

    return process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 3 Cell Type Classification Training")
    parser.add_argument("--quick", action="store_true", help="Quick test: 1 fold, 5 epochs, testing subset")
    parser.add_argument("--full", action="store_true", help="Full training: 5 folds, 200 epochs")
    parser.add_argument("--exp-code", type=str, default="task3_experiment", help="Experiment code for results")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use")
    args = parser.parse_args()

    mode = "quick" if args.quick else ("full" if args.full else "default")
    results_dir = Path.cwd() / "results"
    results_dir.mkdir(exist_ok=True)

    main_cli = build_main_cli(results_dir, args.exp_code, mode)
    print_overview(args.exp_code, results_dir, DATA_ROOT, mode, main_cli)

    resp = input("\nProceed with training? (y/n): ").strip().lower()
    if resp != "y":
        print("Training cancelled.")
        return

    print("\n" + "=" * 60)
    print("STARTING TRAINING...")
    print("=" * 60 + "\n")
    start_time = time.time()

    try:
        return_code = run_tianyu_main(main_cli, args.gpu)
        elapsed_min = (time.time() - start_time) / 60

        if return_code == 0:
            print("\n" + "=" * 60)
            print("✅ TRAINING COMPLETED SUCCESSFULLY!")
            print(f"Total time: {elapsed_min:.1f} minutes")
            print(f"Results saved in: {results_dir / args.exp_code}*")
            print("=" * 60)
            print("\nTo check results:")
            print(f"  - Summary: cat {results_dir}/{args.exp_code}*/summary.csv")
            print(f"  - Checkpoints: ls {results_dir}/{args.exp_code}*/*.pt")
            print(f"  - Logs: ls {results_dir}/{args.exp_code}*/*.pkl")
        else:
            print(f"\n❌ Training failed with return code: {return_code}")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")


if __name__ == "__main__":
    main()
