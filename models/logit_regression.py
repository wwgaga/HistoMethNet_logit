from torch import nn


class LogitRegression(nn.Module):
    """
    Minimal multinomial logistic regression head.

    Maps per-patch feature vectors to class logits using a single
    linear layer. Use with CrossEntropyLoss on raw logits.
    """

    def __init__(self, in_dim: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes, bias=bias)

    def forward(self, x):
        # x: [num_patches, in_dim]
        return self.classifier(x)

