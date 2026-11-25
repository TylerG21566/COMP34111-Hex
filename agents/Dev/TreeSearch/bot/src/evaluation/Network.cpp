#include "evaluation/Network.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace hex {

Network Network::withRandomWeights(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    // Xavier initialisation: uniform, with a range set by the layer's size.
    // std::uniform_real_distribution is not used, because its results differ
    // between compilers.
    auto weight = [&rng](int inputs, int outputs) {
        const float uniform = static_cast<float>(rng() >> 40) / static_cast<float>(1 << 24);
        return (2.0f * uniform - 1.0f) * std::sqrt(6.0f / static_cast<float>(inputs + outputs));
    };

    Network network;
    for (auto& row : network.hiddenWeights_) {
        for (float& w : row) w = weight(kInputs, kHidden);
    }
    for (float& w : network.valueWeights_) w = weight(kHidden, 1);
    for (auto& row : network.policyWeights_) {
        for (float& w : row) w = weight(kHidden, kCells);
    }
    return network;
}

Network::Output Network::forward(const std::array<float, kInputs>& input) const {
    // Most inputs are 0, so they are skipped
    std::array<float, kHidden> hidden = hiddenBias_;
    for (int i = 0; i < kInputs; ++i) {
        if (input[i] == 0.0f) continue;
        for (int h = 0; h < kHidden; ++h) hidden[h] += input[i] * hiddenWeights_[i][h];
    }

    float value = valueBias_;
    std::array<float, kCells> logits = policyBias_;
    for (int h = 0; h < kHidden; ++h) {
        if (hidden[h] <= 0.0f) continue;  // ReLU
        value += hidden[h] * valueWeights_[h];
        for (int c = 0; c < kCells; ++c) logits[c] += hidden[h] * policyWeights_[h][c];
    }

    Output output;
    output.value = std::tanh(value);

    // Softmax. Subtracting the largest logit keeps exp() from overflowing.
    const float largest = *std::max_element(logits.begin(), logits.end());
    float total = 0.0f;
    for (int c = 0; c < kCells; ++c) {
        output.policy[c] = std::exp(logits[c] - largest);
        total += output.policy[c];
    }
    for (float& p : output.policy) p /= total;
    return output;
}

}  // namespace hex
