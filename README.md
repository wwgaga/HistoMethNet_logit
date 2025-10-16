HistoMethNet: RRT with Logistic Regression Head

This repository includes an additional RRT variant that swaps the original Transformer-based classifier for a simple multinomial logistic regression head. It is added as a new model type so the original `models/rrt.py` remains unchanged.

What’s New
- `models/logit_regression.py`: Minimal multinomial logistic head (`nn.Linear(in_dim → num_classes)`).
- `models/rrt_logit.py`: `RRTMILLogit` model that reuses the `RRTEncoder` and replaces the classifier with the logistic head.
- `main.py` and `eval.py`: `--model_type rrt_logit` added to CLI choices.
- `utils/core_utils.py`: Training/validation/summary logic extended to support `rrt_logit` using the existing RRT code paths.

When To Use
- Use `rrt_logit` when you want the simplest classifier (logistic regression) on top of RRT-encoded patch features. This can serve as a baseline or ablation relative to the TransformerBackbone classifier.

Train
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
- `--model_type rrt_logit`: selects the logistic regression head.
- `--cell_property` controls class count (default `cell_type` → 8 classes; otherwise 5).

Evaluate
```
python eval.py \
  --model_type rrt_logit \
  --data_root_dir /path/to/features \
  --models_exp_code exp_rrt_logit_s1 \
  --save_exp_code eval_rrt_logit \
  --task task_3_cell_type_classification \
  --data_csv /path/to/dataset.csv
```

Runner Script
- A convenience script `run_rrt_logit.sh` is provided. Update `--data_root_dir`, `--exp_code`, and other flags as needed, then run:
```
bash run_rrt_logit.sh
```

Original RRT
- To use the original Transformer-based classifier, keep `--model_type rrt` (default behavior from the original codebase).

