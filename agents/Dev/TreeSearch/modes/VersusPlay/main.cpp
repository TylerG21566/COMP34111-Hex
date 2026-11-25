// The program DevAgent.py starts: cpp_agent <R|B> <board size> <weights file>
// The colour is not used: the board in each message gives the side to move,
// which stays correct when either player swaps.

#include <exception>
#include <iostream>
#include <optional>
#include <string>

#include "evaluation/NetworkEvaluator.h"
#include "mcts/Search.h"
#include "protocol/Protocol.h"

namespace {

// Red's first move. A fixed move off the centre, so that it is not an obvious
// swap; a trained network could choose it instead.
constexpr int kOpeningCell = 1 * hex::kSize + 2;

std::string chooseReply(hex::Search& search, const hex::Request& request) {
    if (request.turn == 1) return hex::formatMove(kOpeningCell);

    const hex::SearchResult result = search.run(request.board);
    std::cerr << "turn " << request.turn << ": value " << result.value << ", "
              << result.simulations << " simulations\n";

    // After a swap we have the opponent's side of the same position, which is
    // worth -value to us.
    if (request.turn == 2 && result.value < 0.0f) return hex::formatSwap();
    return hex::formatMove(result.bestCell);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: cpp_agent <R|B> <board size> <weights file>\n";
        return 1;
    }
    if (std::string(argv[2]) != std::to_string(hex::kSize)) {
        std::cerr << "only " << hex::kSize << "x" << hex::kSize << " boards are supported\n";
        return 1;
    }

    std::optional<hex::NetworkEvaluator> evaluator;
    try {
        evaluator.emplace(hex::Network::load(argv[3]));
    } catch (const std::exception& e) {
        std::cerr << "cpp_agent: " << e.what() << "\n";
        return 1;
    }
    hex::Search search(*evaluator, hex::SearchConfig{});

    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            std::cout << chooseReply(search, hex::parseRequest(line)) << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "cpp_agent: " << e.what() << " in: " << line << "\n";
            return 1;
        }
    }
    return 0;
}
