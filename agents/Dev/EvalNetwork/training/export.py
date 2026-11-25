"""Writes a network to a weights file for the C++ bot.

    python3 export.py OUT.bin --checkpoint MODEL.pt     a trained network
    python3 export.py OUT.bin --hidden 64 --seed 0      an untrained network

The file describes the network's layers, so the C++ needs no change when they
change. All numbers are little-endian:

    "HEXN", version
    trunk, value head, policy head, each as: number of layers, then the layers

    Linear: 1, inputs, outputs, weights [outputs][inputs], bias [outputs]
    ReLU:   2
    Tanh:   3

Integers are unsigned 32-bit and weights are 32-bit floats. The reader is
Network::load in TreeSearch/bot/src/evaluation/Network.cpp.
"""

import argparse
import os
import struct

import numpy as np
import torch
from torch import nn

from model import HexNet, load_checkpoint

MAGIC = b"HEXN"
VERSION = 1
LINEAR, RELU, TANH = 1, 2, 3


def _floats(tensor: torch.Tensor) -> bytes:
    values = tensor.detach().cpu().numpy().astype("<f4")
    if not np.isfinite(values).all():
        raise ValueError("the network has weights that are not numbers (did training diverge?)")
    return values.tobytes()


def _layers(layers: nn.Sequential) -> bytes:
    data = struct.pack("<I", len(layers))
    for layer in layers:
        if type(layer) is nn.Linear:
            if layer.bias is None:
                raise ValueError("Linear layers must have a bias")
            data += struct.pack("<III", LINEAR, layer.in_features, layer.out_features)
            data += _floats(layer.weight) + _floats(layer.bias)
        elif type(layer) is nn.ReLU:
            data += struct.pack("<I", RELU)
        elif type(layer) is nn.Tanh:
            data += struct.pack("<I", TANH)
        else:
            raise TypeError(f"{type(layer).__name__} layers are not supported")
    return data


def export_network(model: HexNet, path) -> None:
    data = MAGIC + struct.pack("<I", VERSION)
    for part in (model.trunk, model.value_head, model.policy_head):
        data += _layers(part)
    # written to another file first, so that a bot never reads half a network
    temporary = f"{path}.tmp"
    with open(temporary, "wb") as f:
        f.write(data)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description="Write a network for the C++ bot.")
    parser.add_argument("out", help="the weights file to write")
    parser.add_argument("--checkpoint", help="a checkpoint from training")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64],
                        help="hidden layer sizes of an untrained network (default: 64)")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed of an untrained network (default: 0)")
    args = parser.parse_args()

    if args.checkpoint:
        model = load_checkpoint(args.checkpoint)
    else:
        torch.manual_seed(args.seed)
        model = HexNet(tuple(args.hidden))
    export_network(model, args.out)
    count = sum(p.numel() for p in model.parameters())
    print(f"wrote {args.out}: hidden layers {list(model.hidden)}, {count} weights")


if __name__ == "__main__":
    main()
