#pragma once

#include <array>
#include <cstdint>

#include "game/Board.h"

namespace hex {

// A small fully connected network with one hidden layer:
//
//   input (242) -> hidden (64, ReLU) -> value  (1, tanh)
//                                    -> policy (121, softmax)
class Network {
public:
    // Plane 0: the side to move's stones. Plane 1: the opponent's stones.
    static constexpr int kInputs = 2 * kCells;
    static constexpr int kHidden = 64;

    struct Output {
        float value = 0.0f;                  // -1 to 1, for the side to move
        std::array<float, kCells> policy{};  // sums to 1
    };

    // Untrained: random weights, and zero biases. The same seed always gives
    // the same network.
    static Network withRandomWeights(std::uint64_t seed);

    Output forward(const std::array<float, kInputs>& input) const;

private:
    Network() = default;

    // Weights are stored as [from][to]
    std::array<std::array<float, kHidden>, kInputs> hiddenWeights_{};
    std::array<float, kHidden> hiddenBias_{};
    std::array<float, kHidden> valueWeights_{};
    float valueBias_ = 0.0f;
    std::array<std::array<float, kCells>, kHidden> policyWeights_{};
    std::array<float, kCells> policyBias_{};
};

}  // namespace hex
