from copy import deepcopy
from pathlib import Path
from subprocess import PIPE, Popen

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

# The compiled C++ agent. Change this to launch a different program.
CPP_AGENT = Path(__file__).resolve().parent / "cpp_agent"
BOARD_SIZE = 11


class DevAgent(AgentBase):
    """Runs a compiled C++ agent and passes messages between it and the engine.

    The C++ program is started once per game as `cpp_agent <R|B> <board size>`.
    Each turn it reads one line from stdin and writes one line to stdout:

        in:   COMMAND;MOVE;BOARD;TURN;    COMMAND is START, CHANGE or SWAP
        out:  x,y                         or -1,-1 to swap (turn 2 only)

    The C++ program must:
      - flush stdout after every move (std::endl does this)
      - write debug text to std::cerr only
      - change its own colour when it receives SWAP, and also when it sends
        -1,-1 itself (no message is sent for its own swap)
      - exit when stdin closes

    Full protocol: "Message format" in docs/GAME.md.
    """

    def __init__(self, colour: Colour, cpp_agent: Path | str = CPP_AGENT):
        super().__init__(colour)

        if not Path(cpp_agent).is_file():
            raise FileNotFoundError(f"C++ agent not found at {cpp_agent}: compile it first.")

        # stderr is not piped, so C++ debug output goes to the terminal
        self.agent_process = Popen(
            [str(cpp_agent), colour.to_protocol_char(), str(BOARD_SIZE)],
            stdin=PIPE,
            stdout=PIPE,
            text=True,
        )

    def __deepcopy__(self, memo):
        """Copies everything except the C++ process, which cannot be copied."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__ = {
            k: deepcopy(v, memo)
            for k, v in self.__dict__.items()
            if k != "agent_process"
        }
        result.agent_process = None
        return result

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Sends the position to the C++ agent and returns the move it chooses."""
        self._send(self._build_message(turn, board, opp_move))
        return self._parse_move(self._receive())

    @staticmethod
    def _build_message(turn: int, board: Board, opp_move: Move | None) -> str:
        # rows joined by commas, R/B/0 per tile
        board_string = board.to_protocol_string()

        if opp_move is None:
            return f"START;;{board_string};{turn};"
        if opp_move.is_swap():
            return f"SWAP;;{board_string};{turn};"
        return f"CHANGE;{opp_move.x},{opp_move.y};{board_string};{turn};"

    def _send(self, message: str) -> None:
        self.agent_process.stdin.write(message + "\n")
        self.agent_process.stdin.flush()

    def _receive(self) -> str:
        response = self.agent_process.stdout.readline().strip()
        if not response:
            raise RuntimeError(
                f"C++ agent sent no move (exit code {self.agent_process.poll()}): "
                "it may have crashed, or sent an empty line."
            )
        return response

    @staticmethod
    def _parse_move(response: str) -> Move:
        try:
            x, y = response.split(",")
            return Move(int(x), int(y))
        except ValueError:
            raise ValueError(
                f"C++ agent sent {response!r}, expected 'x,y'. "
                "Debug text must go to std::cerr, not std::cout."
            ) from None
