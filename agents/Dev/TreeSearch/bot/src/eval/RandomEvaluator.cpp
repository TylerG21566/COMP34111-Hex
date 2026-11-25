#include "eval/RandomEvaluator.h"

namespace hex {

namespace {

// SplitMix64: small, fast random generator
std::uint64_t nextRandom(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Uniform in [0, 1)
float nextFloat(std::uint64_t& state) {
    return static_cast<float>(nextRandom(state) >> 40) / static_cast<float>(1 << 24);
}

}  // namespace

Evaluation RandomEvaluator::evaluate(const Board& board) {
    // FNV-1a hash of the position
    std::uint64_t state = 0xCBF29CE484222325ULL ^ seed_;
    for (int cell = 0; cell < kCells; ++cell) {
        state = (state ^ static_cast<std::uint64_t>(board.at(cell))) * 0x100000001B3ULL;
    }

    Evaluation result;
    result.value = 2.0f * nextFloat(state) - 1.0f;
    for (float& p : result.policy) p = nextFloat(state);
    return result;
}

}  // namespace hex
