HistoMethNet: Logistic Heads for Cell-Type Proportions

This repository extends the original RRT pipeline with two lightweight logistic heads for cell-type proportion prediction:
- `rrt_logit`: keeps the RRT encoder and replaces the classifier with a multinomial logistic regression head.
- `logit_only`: removes the RRT encoder entirely and uses only a multinomial logistic regression head over patch features.

What’s New
- `models/logit_regression.py`: Minimal multinomial logistic head (`nn.Linear(in_dim → num_classes)`).
- `models/rrt_logit.py`: `RRTMILLogit` model that reuses the `RRTEncoder` and replaces the classifier with the logistic head.
- `models/logit_only.py`: `LogitOnly` model that uses only multinomial logistic regression (no RRT encoder).
- `main.py`, `eval.py`: `--model_type rrt_logit` and `--model_type logit_only` added to CLI choices.
- `utils/core_utils.py`, `utils/eval_utils.py`: Training/validation/summary updated to support `logit_only` via the existing RRT-style code paths.

When To Use
- `rrt_logit`: simplest classifier on top of RRT-encoded patch features. Baseline/ablation vs. TransformerBackbone.
- `logit_only`: no encoder; fastest baseline that treats patch features as linearly separable. Useful for quick sanity checks and lower compute.

Train (RRT + Logistic Head)
```
python main.py \
  --task task_3_cell_type_classification \
  --data_root_dir /path/to/features \
  --subtyping \
  --model_type rrt_logit \
  --exp_code exp_rrt_logit \
  --embed_dim 1024 \
  --drop_out 0.25 \
  --lr 2e-4 \
  --k 5
```
- `--model_type rrt_logit`: selects the logistic regression head on top of the RRT encoder.
- `--cell_property` controls class count (default `cell_type` → 8 classes; otherwise 5).

Train (Logit-Only)
```
python main.py \
  --task task_3_cell_type_classification \
  --data_root_dir /path/to/features \
  --subtyping \
  --model_type logit_only \
  --exp_code exp_logit_only \
  --embed_dim 1536 \
  --drop_out 0.25 \
  --lr 2e-4 \
  --k 5
```
- `--model_type logit_only`: uses only a multinomial logistic regression head (no RRT encoder).
- `--embed_dim` must match your patch feature dimension (e.g., 1024, 1536).

Evaluate (either model)
```
python eval.py \
  --model_type rrt_logit \
  --data_root_dir /path/to/features \
  --models_exp_code exp_rrt_logit_s1 \
  --save_exp_code eval_rrt_logit \
  --task task_3_cell_type_classification \
  --data_csv /path/to/dataset.csv
```
For `logit_only`, replace the model flags accordingly, e.g. `--model_type logit_only`, `--models_exp_code exp_logit_only_s1`, `--save_exp_code eval_logit_only`.

Runner Scripts
- `run_rrt_logit.sh`: RRT encoder + logistic head. Update paths/flags, then run:
```
bash run_rrt_logit.sh
```
- `run_logit_only.sh`: Logit-only baseline (no encoder). Update paths/flags, then run:
```
bash run_logit_only.sh
```

Original RRT
- To use the original Transformer-based classifier, keep `--model_type rrt`.
