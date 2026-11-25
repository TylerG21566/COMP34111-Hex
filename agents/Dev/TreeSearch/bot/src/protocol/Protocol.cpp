#include "protocol/Protocol.h"

#include <sstream>
#include <stdexcept>

namespace hex {

Request parseRequest(const std::string& line) {
    std::istringstream stream(line);
    std::string move, board, turn;
    Request request;
    if (!std::getline(stream, request.command, ';') || !std::getline(stream, move, ';') ||
        !std::getline(stream, board, ';') || !std::getline(stream, turn, ';')) {
        throw std::invalid_argument("message has too few fields");
    }
    if (request.command != "START" && request.command != "CHANGE" && request.command != "SWAP") {
        throw std::invalid_argument("unknown command");
    }
    request.board = Board::fromProtocol(board);
    request.turn = std::stoi(turn);
    return request;
}

std::string formatMove(int cell) {
    return std::to_string(cell / kSize) + "," + std::to_string(cell % kSize);
}

}  // namespace hex
