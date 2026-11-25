#pragma once

#include <array>

#include "game/Board.h"

namespace hex {

struct Evaluation {
    // Expected result for the side to move: -1 (loss) to 1 (win).
    float value = 0.0f;
    // How promising each cell is, all >= 0. The search ignores occupied cells
    // and normalises the rest, so this need not sum to 1.
    std::array<float, kCells> policy{};
};

// What the search needs from the Eval Network. The trained network will be
// another implementation of this class.
class Evaluator {
public:
    virtual ~Evaluator() = default;
    virtual Evaluation evaluate(const Board& board) = 0;
};

}  // namespace hex
