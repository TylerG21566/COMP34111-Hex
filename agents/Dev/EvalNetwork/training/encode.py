"""Turns a board into network inputs, and cells into network cells.

Must match TreeSearch/bot/src/evaluation/NetworkEvaluator.cpp: test_match.py
checks this. The network always sees the board as the player who connects top
to bottom, so for Blue the board is mirrored along its diagonal.
"""

import numpy as np

SIZE = 11
CELLS = SIZE * SIZE
INPUTS = 2 * CELLS  # plane 0: the side to move's stones, plane 1: the opponent's


def side_to_move(board: str) -> str:
    """The colour to move ("R" or "B") on a board in the protocol format."""
    return "B" if board.count("R") > board.count("B") else "R"


def network_cell(cell: int, colour: str) -> int:
    """Where the network sees a cell (x * SIZE + y) when `colour` is to move.
    Mirroring twice changes nothing, so this also converts back."""
    if colour == "R":
        return cell
    return (cell % SIZE) * SIZE + cell // SIZE


def encode(board: str) -> np.ndarray:
    """The network inputs for a board in the protocol format (rows of R/B/0
    joined by commas)."""
    colour = side_to_move(board)
    inputs = np.zeros(INPUTS, dtype=np.float32)
    for cell, stone in enumerate(board.replace(",", "")):
        if stone != "0":
            plane = 0 if stone == colour else 1
            inputs[plane * CELLS + network_cell(cell, colour)] = 1.0
    return inputs
