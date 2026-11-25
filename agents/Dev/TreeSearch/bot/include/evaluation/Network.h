#pragma once

#include <string>
#include <variant>
#include <vector>

#include "game/Board.h"

namespace hex {

// One layer of the network. A new kind of layer needs a struct here, a case in
// Network.cpp (reading and running it), and a case in export.py.
struct Linear {
    int inputs = 0;
    int outputs = 0;
    std::vector<float> weights;  // [input * outputs + output]
    std::vector<float> bias;     // [output]
};
struct Relu {};
struct Tanh {};

using Layer = std::variant<Linear, Relu, Tanh>;
using Layers = std::vector<Layer>;

// A neural network whose layers come from a file written by export.py, so a
// different network needs no change here. Only its ends are fixed:
//
//   input (242) -> trunk -> value head  -> value (1), ends in tanh
//                        -> policy head -> one logit per cell (121)
class Network {
public:
    // Plane 0: the side to move's stones. Plane 1: the opponent's stones.
    static constexpr int kInputs = 2 * kCells;

    struct Output {
        float value = 0.0f;         // -1 to 1, for the side to move
        std::vector<float> policy;  // one per cell, softmax, sums to 1
    };

    // Throws std::invalid_argument if the layers do not fit together, or do not
    // have the inputs and outputs above.
    Network(Layers trunk, Layers valueHead, Layers policyHead);

    // Throws std::runtime_error if the file cannot be read or is not valid.
    static Network load(const std::string& path);

    // `input` must have kInputs values.
    Output forward(const std::vector<float>& input) const;

private:
    Layers trunk_;
    Layers valueHead_;
    Layers policyHead_;
};

}  // namespace hex
