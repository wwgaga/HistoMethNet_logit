import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitOnly(nn.Module):
    """
    Minimal multinomial logistic regression over patch features.

    - Input: tensor of shape [num_patches, feature_dim]
    - Per-patch head: Linear(feature_dim -> n_classes)
    - Slide-level proportion: mean of per-patch softmax across patches

    Returns an instance_dict aligned with RRT-style training utilities:
      - 'cell_type_logits': [num_patches, n_classes]
      - 'cell_type_prob':   [n_classes] (mean-softmax over patches)
      - 'features':         [num_patches, feature_dim] (identity passthrough)
    """

    def __init__(self, input_dim: int, n_classes: int | None = None, dropout: float = 0.0, cell_property: str = 'cell_type'):
        super().__init__()
        # Determine output dimension when not specified
        if n_classes is None:
            n_classes = 8 if cell_property == 'cell_type' else 5
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else None
        self.classifier = nn.Linear(input_dim, n_classes, bias=True)

    def forward(self, x: torch.Tensor):
        # x: [num_patches, feat_dim]
        z = self.dropout(x) if self.dropout is not None else x
        patch_logits = self.classifier(z)
        # Mean of per-patch probabilities as slide-level proportion
        cell_type_prob = F.softmax(patch_logits, dim=1).mean(dim=0)

        return {
            'cell_type_logits': patch_logits,
            'cell_type_prob': cell_type_prob,
            'features': x,
        }
