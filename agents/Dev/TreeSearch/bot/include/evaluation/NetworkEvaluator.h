#pragma once

#include <utility>

#include "evaluation/Evaluator.h"
#include "evaluation/Network.h"

namespace hex {

// Scores positions with a neural network. The network always sees the board as
// the player who connects top to bottom: for Blue the board is mirrored along
// its diagonal, so the network only has to learn one side.
class NetworkEvaluator : public Evaluator {
public:
    explicit NetworkEvaluator(Network network) : network_(std::move(network)) {}
    Evaluation evaluate(const Board& board) override;

private:
    Network network_;
};

}  // namespace hex
