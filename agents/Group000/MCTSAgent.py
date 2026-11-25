"""Monte Carlo tree search Hex agent with RAVE (all-moves-as-first).

How it works
------------
* The tree is searched with UCT, with each child's win rate blended with its
  RAVE (AMAF) win rate. RAVE matters in Hex: a move that is good later in a
  playout is usually good now, so the search learns much faster.
* Playouts fill every empty cell at random, alternating colours, then check
  the winner once. Hex has no draws, and a full board has exactly one winner,
  so no per-move win check is needed. Once a side has connected, filling the
  rest of the board can't undo it, so terminal nodes need no special case.
* Before searching, the agent plays any move that wins at once and blocks any
  single cell where the opponent would win at once.
* The subtree under the move actually played is kept between turns.
* Time: a per-move budget drawn from the agent's own clock (5 minutes for the
  game), capped at MAX_MOVE_TIME. Single-threaded, so nothing runs while the
  agent is paused during the opponent's turn.

Coordinates: cell (x, y) is board.tiles[x][y], stored as index x * size + y.
Red (1) connects row 0 to row size-1; Blue (2) connects column 0 to size-1.
"""

import math
import time

import numpy as np

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

EMPTY, RED, BLUE = 0, 1, 2

# --- tuning -----------------------------------------------------------------
MAX_MOVE_TIME = 1.5        # seconds per move at most
GAME_TIME = 300.0          # the chess clock for the whole game
SAFETY_MARGIN = 25.0       # seconds kept in reserve
RAVE_K = 600.0             # RAVE weight: beta = sqrt(K / (3n + K))
EXPLORATION = 0.25         # UCT exploration constant
FPU = 1.1                  # first-play urgency for unvisited children
# Red's first move. A centre move is usually swapped, an edge move is weak;
# this one is a compromise that is fine either way.
OPENING_MOVE = (1, 2)


def _other(player):
    return RED if player == BLUE else BLUE


class _Node:
    """A position in the tree. Per-child statistics are stored in arrays
    indexed by cell, from the point of view of `to_move` at this node."""

    __slots__ = ("to_move", "legal", "n", "w", "rn", "rw", "visits", "children")

    def __init__(self, to_move, legal):
        size = legal.shape[0]
        self.to_move = to_move
        self.legal = legal                     # bool array: empty cells
        self.n = np.zeros(size)                # visits of each child
        self.w = np.zeros(size)                # wins of each child
        self.rn = np.zeros(size)               # RAVE visits
        self.rw = np.zeros(size)               # RAVE wins
        self.visits = 0
        self.children = {}                     # cell -> _Node (created lazily)

    def select(self):
        n = self.n
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(n > 0, self.w / n, FPU)
            rq = np.where(self.rn > 0, self.rw / self.rn, 0.5)
        beta = np.sqrt(RAVE_K / (3.0 * n + RAVE_K))
        explore = EXPLORATION * np.sqrt(math.log(self.visits + 1) / (n + 1))
        score = (1.0 - beta) * q + beta * rq + explore
        score[~self.legal] = -np.inf
        return int(np.argmax(score))


class _Geometry:
    """Neighbour lists and edge cells for one board size."""

    def __init__(self, size):
        self.size = size
        self.cells = size * size
        offsets = ((0, -1), (0, 1), (-1, 0), (-1, 1), (1, -1), (1, 0))
        self.neighbours = []
        for x in range(size):
            for y in range(size):
                nb = []
                for dx, dy in offsets:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        nb.append(nx * size + ny)
                self.neighbours.append(tuple(nb))
        # start cells and goal test for each colour
        self.start = {
            RED: [y for y in range(size)],                          # row 0
            BLUE: [x * size for x in range(size)],                  # column 0
        }
        self.goal = {
            RED: set((size - 1) * size + y for y in range(size)),   # last row
            BLUE: set(x * size + size - 1 for x in range(size)),    # last column
        }

    def connected(self, board, player):
        """True if `player` connects their two sides on `board` (a list)."""
        neighbours = self.neighbours
        goal = self.goal[player]
        stack = [c for c in self.start[player] if board[c] == player]
        seen = set(stack)
        while stack:
            c = stack.pop()
            if c in goal:
                return True
            for nb in neighbours[c]:
                if nb not in seen and board[nb] == player:
                    seen.add(nb)
                    stack.append(nb)
        return False


class MCTSAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._geo = None
        self._root = None
        self._time_used = 0.0

    # --- engine interface ---------------------------------------------------
    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        start = time.perf_counter()
        try:
            move = self._choose(turn, board, opp_move, start)
        except Exception:
            # never lose to a crash: fall back to any legal move
            move = self._fallback(board)
        self._time_used += time.perf_counter() - start
        return move

    # --- move choice --------------------------------------------------------
    def _choose(self, turn, board, opp_move, start):
        size = board.size
        if self._geo is None or self._geo.size != size:
            self._geo = _Geometry(size)
            self._root = None
        geo = self._geo
        me = RED if self.colour == Colour.RED else BLUE

        cells = self._read_board(board)
        empties = [i for i, v in enumerate(cells) if v == EMPTY]

        # pie rule: swap a strong opening stone (anything off the outer ring)
        if turn == 2 and opp_move is not None and not opp_move.is_swap():
            if 1 <= opp_move.x <= size - 2 and 1 <= opp_move.y <= size - 2:
                self._root = None
                return Move(-1, -1)

        if turn == 1:
            x, y = OPENING_MOVE
            if x < size and y < size:
                self._root = None
                return Move(x, y)

        # immediate win, then forced block
        forced = self._forced_move(cells, empties, me)
        if forced is not None:
            self._root = None
            return self._to_move(forced, size)

        root = self._reuse_tree(cells, opp_move, me)
        deadline = start + self._budget(len(empties))
        self._search(root, cells, deadline)

        best = int(np.argmax(np.where(root.legal, root.n, -1)))
        self._root = root.children.get(best)
        return self._to_move(best, size)

    def _search(self, root, cells, deadline):
        geo = self._geo
        cell_count = geo.cells
        permutation = np.random.permutation
        iterations = 0
        while True:
            # check the clock every 16 iterations to keep overhead low
            if iterations & 15 == 0 and time.perf_counter() >= deadline:
                break
            iterations += 1

            board = cells.copy()
            node = root
            path = []
            # selection and expansion
            while True:
                cell = node.select()
                path.append((node, cell))
                board[cell] = node.to_move
                child = node.children.get(cell)
                if child is None:
                    legal = node.legal.copy()
                    legal[cell] = False
                    if legal.any():
                        node.children[cell] = _Node(_other(node.to_move), legal)
                    break
                node = child

            # playout: fill the rest of the board at random, alternating
            final = np.array(board, dtype=np.int8)
            empty_idx = np.flatnonzero(final == EMPTY)
            if empty_idx.size:
                order = permutation(empty_idx)
                mover = _other(path[-1][0].to_move)
                final[order[0::2]] = mover
                final[order[1::2]] = _other(mover)
            winner = RED if geo.connected(final.tolist(), RED) else BLUE

            # backpropagation, with RAVE updates for every cell each side owns
            red_mask = final == RED
            blue_mask = ~red_mask
            for node, cell in path:
                won = 1.0 if winner == node.to_move else 0.0
                node.visits += 1
                node.n[cell] += 1.0
                node.w[cell] += won
                owned = (red_mask if node.to_move == RED else blue_mask) & node.legal
                node.rn[owned] += 1.0
                if won:
                    node.rw[owned] += 1.0
        self._last_iterations = iterations
        return iterations

    # --- helpers ------------------------------------------------------------
    def _budget(self, empty_count):
        remaining = GAME_TIME - SAFETY_MARGIN - self._time_used
        # assume we still have to make about half the remaining moves
        moves_left = max(8, empty_count // 2)
        return max(0.05, min(MAX_MOVE_TIME, remaining / moves_left))

    def _reuse_tree(self, cells, opp_move, me):
        """Keep the subtree for the opponent's reply if we have it."""
        root = self._root
        self._root = None
        if root is not None and opp_move is not None and not opp_move.is_swap():
            cell = opp_move.x * self._geo.size + opp_move.y
            child = root.children.get(cell)
            if child is not None and child.to_move == me:
                legal = np.array([v == EMPTY for v in cells])
                if np.array_equal(child.legal, legal):
                    return child
        legal = np.array([v == EMPTY for v in cells])
        return _Node(me, legal)

    def _forced_move(self, cells, empties, me):
        geo = self._geo
        opp = _other(me)
        board = cells.copy()
        for player in (me, opp):
            for c in empties:
                board[c] = player
                if geo.connected(board, player):
                    return c
                board[c] = EMPTY
        return None

    @staticmethod
    def _read_board(board):
        cells = []
        for row in board.tiles:
            for tile in row:
                if tile.colour == Colour.RED:
                    cells.append(RED)
                elif tile.colour == Colour.BLUE:
                    cells.append(BLUE)
                else:
                    cells.append(EMPTY)
        return cells

    @staticmethod
    def _to_move(cell, size):
        return Move(cell // size, cell % size)

    @staticmethod
    def _fallback(board):
        for row in board.tiles:
            for tile in row:
                if tile.colour is None:
                    return Move(tile.x, tile.y)
        return Move(-1, -1)
