"""The Eval Network. Change the layers as you like: export.py writes them to the
weights file, and the C++ bot reads them from there.

The ends must stay the same: INPUTS inputs, a value head that ends in Tanh with
1 output, and a policy head with CELLS outputs. export.py supports the layers
Linear, ReLU and Tanh.
"""

import torch
from torch import nn

from encode import CELLS, INPUTS


class HexNet(nn.Module):
    def __init__(self, hidden: tuple[int, ...] = (64,)):
        """`hidden` is the size of each hidden layer."""
        super().__init__()
        self.hidden = tuple(hidden)
        layers = []
        size = INPUTS
        for width in self.hidden:
            layers += [nn.Linear(size, width), nn.ReLU()]
            size = width
        self.trunk = nn.Sequential(*layers)
        self.value_head = nn.Sequential(nn.Linear(size, 1), nn.Tanh())
        self.policy_head = nn.Sequential(nn.Linear(size, CELLS))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the values (batch) and the policy logits (batch, CELLS)."""
        features = self.trunk(inputs)
        return self.value_head(features).squeeze(-1), self.policy_head(features)


def save_checkpoint(model: HexNet, path) -> None:
    torch.save({"hidden": model.hidden, "model": model.state_dict()}, path)


def load_checkpoint(path) -> HexNet:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = HexNet(checkpoint["hidden"])
    model.load_state_dict(checkpoint["model"])
    return model
