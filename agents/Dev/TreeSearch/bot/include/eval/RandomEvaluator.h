#pragma once

#include <cstdint>

#include "eval/Evaluator.h"

namespace hex {

// Placeholder for the Eval Network: random value and policy, with no knowledge
// of Hex. Like a network, it gives the same output for the same position.
class RandomEvaluator : public Evaluator {
public:
    explicit RandomEvaluator(std::uint64_t seed) : seed_(seed) {}
    Evaluation evaluate(const Board& board) override;

private:
    std::uint64_t seed_;
};

}  // namespace hex
