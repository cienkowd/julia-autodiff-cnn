"""Sieć referencyjna w PyTorch — odpowiednik sieci z KM1 (flex.jl / notebook).

Struktura i parametry MUSZĄ być takie same jak w wersji Julia:
  Conv 1->6 (3x3, pad=1, bez bias) -> MaxPool 2 ->
  Conv 6->16 (3x3, pad=1, bez bias) -> MaxPool 2 ->
  Flatten (16*7*7 = 784) -> Dense 784->84 -> ReLU -> Dropout(0.4) -> Dense 84->10
  strata: cross-entropy na logitach; optymalizator: SGD, lr=1e-2, batch=10, 3 epoki.
"""

import torch.nn as nn


class FlexNet(nn.Module):
    def __init__(self, p_drop: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 84),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def he_init(m):
    """Inicjalizacja He (jak w flex.jl: skalowanie sqrt(2/fan_in))."""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
