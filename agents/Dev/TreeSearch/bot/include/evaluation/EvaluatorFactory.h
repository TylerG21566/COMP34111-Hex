#pragma once

#include <cstdint>
#include <memory>

#include "evaluation/Evaluator.h"

namespace hex {

// Makes the evaluator the bot plays with: a neural network, which has random
// weights made from the seed until trained weights are loaded.
std::unique_ptr<Evaluator> makeEvaluator(std::uint64_t seed);

}  // namespace hex
