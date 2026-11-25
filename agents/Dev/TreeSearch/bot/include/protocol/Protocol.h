#pragma once

#include <string>

#include "game/Board.h"

namespace hex {

// One message from the engine: COMMAND;MOVE;BOARD;TURN;
// See "Message format" in docs/GAME.md.
struct Request {
    std::string command;  // START, CHANGE or SWAP
    Board board;
    int turn = 0;
};

// Throws std::invalid_argument if the line is not a valid message.
Request parseRequest(const std::string& line);

std::string formatMove(int cell);
inline std::string formatSwap() { return "-1,-1"; }

}  // namespace hex
