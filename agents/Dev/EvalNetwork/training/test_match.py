"""Checks that the C++ and PyTorch networks give the same outputs.

    python3 test_match.py

Compiles TreeSearch/tests/print_evaluation.cpp and compares its answers with
PyTorch, for networks of several shapes and random positions of both colours.
"""

import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from encode import CELLS, SIZE, encode, network_cell, side_to_move
from export import export_network
from model import HexNet, load_checkpoint, save_checkpoint

TREE_SEARCH = Path(__file__).resolve().parents[2] / "TreeSearch"
SHAPES = [(64,), (128, 64), (32, 32, 32), ()]
TOLERANCE = 1e-5


def build(program: Path) -> None:
    sources = sorted((TREE_SEARCH / "bot" / "src").rglob("*.cpp"))
    subprocess.run(
        ["g++", "-std=c++20", "-O2", "-I", TREE_SEARCH / "bot" / "include", "-o", program,
         TREE_SEARCH / "tests" / "print_evaluation.cpp", *sources],
        check=True,
    )


def random_board(rng: random.Random) -> str:
    stones = rng.randrange(0, CELLS // 2)
    cells = ["0"] * CELLS
    for i, cell in enumerate(rng.sample(range(CELLS), stones)):
        cells[cell] = "RB"[i % 2]  # Red first, so both colours get to move
    return ",".join("".join(cells[x * SIZE:(x + 1) * SIZE]) for x in range(SIZE))


def python_outputs(model: HexNet, board: str) -> np.ndarray:
    with torch.no_grad():
        value, logits = model(torch.from_numpy(encode(board)).unsqueeze(0))
    policy = torch.softmax(logits[0], dim=0).numpy()
    colour = side_to_move(board)
    cells = [policy[network_cell(cell, colour)] for cell in range(CELLS)]
    return np.array([value.item(), *cells])


def main() -> int:
    rng = random.Random(1)
    boards = [random_board(rng) for _ in range(200)]
    colours = [side_to_move(board) for board in boards]
    print(f"{colours.count('R')} positions with Red to move, {colours.count('B')} with Blue")

    failed = False
    with tempfile.TemporaryDirectory() as folder:
        program = Path(folder) / "print_evaluation"
        build(program)
        for seed, hidden in enumerate(SHAPES):
            torch.manual_seed(seed)
            # through a checkpoint as well, as in training
            save_checkpoint(HexNet(hidden), Path(folder) / "model.pt")
            model = load_checkpoint(Path(folder) / "model.pt")
            weights = Path(folder) / "weights.bin"
            export_network(model, weights)

            result = subprocess.run([program, weights], input="\n".join(boards) + "\n",
                                    capture_output=True, text=True, check=True)
            cpp = np.array([line.split() for line in result.stdout.splitlines()], dtype=float)
            python = np.array([python_outputs(model, board) for board in boards])
            difference = np.abs(cpp - python).max() if cpp.shape == python.shape else np.inf
            ok = difference < TOLERANCE
            failed |= not ok
            print(f"{'ok  ' if ok else 'FAIL'}  hidden {hidden}: largest difference {difference:.1e}")
    print("FAILED" if failed else "all match")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
