# Repository Guidelines

This guide helps contributors work productively on HistoMethNet. It summarizes layout, key commands, conventions, and review expectations.

## Project Structure & Module Organization
- `main.py`: Argparse entry for training/evaluation.
- `dataset_modules/`: Dataset classes and split utilities.
- `models/`: Model implementations (e.g., CLAM, RRT variants).
- `utils/`: Training loop and helpers (`core_utils.py`, `file_utils.py`).
- `wsi_core/`: Whole-slide image I/O, patching, preprocessing.
- `visualization/`, `vis_utils/`: Plotting and heatmap helpers.
- `presets/`: CSV presets for patch extraction/thresholds.
- `dataset_csv/`: Slide/split CSVs (no PHI).
- `splits/`: Generated/curated folds used by `--split_dir`.
- `eval.py`, `eval_*/*.sh`: Evaluation scripts.

## Build, Test, and Development Commands
- Create env: `conda env create -f env.yml && conda activate clam_latest`.
- Train (example): `python main.py --task task_2_tumor_subtyping --data_root_dir /path/to/features --subtyping --model_type clam_sb --exp_code exp01`.
- Evaluate: `python eval.py --results_dir ./results/exp01_s1`.
- Common runners: `bash run_extract_patches.sh`, `bash run_rrt.sh`.

## Coding Style & Naming Conventions
- Python 3.10; follow PEP8, 4-space indentation.
- Modules/functions: snake_case; classes: CamelCase.
- Keep scripts idempotent and CLI-driven (use `argparse`).
- Prefer explicit paths via flags (e.g., `--data_root_dir`, `--split_dir`).

## Testing Guidelines
- No formal unit tests yet; validate via small runs.
- Quick sanity: add `--testing` to training/eval commands.
- Single-fold checks: `--k 1 --k_start 0 --k_end 1`.
- If adding tests, place near modules or under `tests/` with `pytest`-style names `test_*.py`.

## Commit & Pull Request Guidelines
- Commits: concise, imperative; optional scope, e.g., `feat(models): add RRT pos embedding option`.
- PRs: include description, rationale, example command lines, dataset assumptions, and before/after metrics/plots (AUC/ACC).
- Link related issues; attach small logs (e.g., `experiment_*.txt`, `summary.csv`).

## Security & Configuration Tips
- Never commit datasets or PHI; keep data local via `--data_root_dir`.
- Store large artifacts under `./results/` (gitignored recommended) to keep the repo lean.
- Preserve reproducibility by saving run configs, `experiment_*.txt`, and `summary.csv`; avoid hard-coded absolute paths.

