# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: Argparse entry for training/evaluation.
- `dataset_modules/`: Dataset classes and split utilities.
- `models/`: Model implementations (CLAM, RRT variants).
- `utils/`: Training loop and helpers (`core_utils.py`, `file_utils.py`).
- `wsi_core/`: Whole-slide image I/O, patching, preprocessing.
- `visualization/`, `vis_utils/`: Plotting and heatmap helpers.
- `presets/`: CSV presets for patch extraction/thresholds.
- `dataset_csv/`: Slide/split CSVs (no PHI).
- `splits/`: Generated or curated folds for `--split_dir`.
- `eval.py`, `eval_*/*.sh`: Evaluation scripts and runners.

## Build, Test, and Development Commands
- Create env: `conda env create -f env.yml && conda activate clam_latest`.
- Train (example): `python main.py --task task_2_tumor_subtyping --data_root_dir /path/to/features --subtyping --model_type clam_sb --exp_code exp01`.
- Evaluate: `python eval.py --results_dir ./results/exp01_s1`.
- Common runners: `bash run_extract_patches.sh`, `bash run_rrt.sh`.
- Tip: use `--testing` for quick sanity checks; store artifacts under `./results/`.

## Coding Style & Naming Conventions
- Python 3.10; PEP8, 4-space indentation.
- Modules/functions: snake_case; classes: CamelCase.
- Scripts must be idempotent and CLI-driven (`argparse`).
- Prefer explicit flags for paths (e.g., `--data_root_dir`, `--split_dir`).

## Testing Guidelines
- No formal unit tests yet; validate with small runs.
- Fast checks: append `--testing` to train/eval commands.
- Single-fold debugging: `--k 1 --k_start 0 --k_end 1`.
- If adding tests, place near modules or under `tests/` with `pytest`-style names `test_*.py`.

## Commit & Pull Request Guidelines
- Commits: concise, imperative; optional scope, e.g., `feat(models): add RRT pos embedding option`.
- PRs should include: purpose/rationale, example commands, dataset assumptions, and before/after metrics/plots (AUC/ACC).
- Link related issues and attach small logs (`experiment_*.txt`, `summary.csv`).

## Security & Configuration Tips
- Do not commit datasets or PHI; keep data local via `--data_root_dir`.
- Save run configs and outputs (`experiment_*.txt`, `summary.csv`) for reproducibility.
- Keep large artifacts in `./results/` (consider `.gitignore`).
- Avoid hard-coded absolute paths; prefer project-relative paths and flags.
