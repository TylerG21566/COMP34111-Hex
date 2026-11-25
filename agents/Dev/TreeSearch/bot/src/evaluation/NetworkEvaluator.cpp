#include "evaluation/NetworkEvaluator.h"

namespace hex {

namespace {

// Where the network sees `cell`. Mirroring swaps row and column.
int networkCell(int cell, Colour toMove) {
    if (toMove == Colour::Red) return cell;
    return (cell % kSize) * kSize + cell / kSize;
}

}  // namespace

Evaluation NetworkEvaluator::evaluate(const Board& board) {
    const Colour me = board.toMove();

    std::array<float, Network::kInputs> input{};
    for (int cell = 0; cell < kCells; ++cell) {
        if (board.isEmpty(cell)) continue;
        const int plane = board.at(cell) == me ? 0 : 1;
        input[plane * kCells + networkCell(cell, me)] = 1.0f;
    }

    const Network::Output output = network_.forward(input);

    Evaluation result;
    result.value = output.value;
    for (int cell = 0; cell < kCells; ++cell) {
        result.policy[cell] = output.policy[networkCell(cell, me)];
    }
    return result;
}

}  // namespace hex
