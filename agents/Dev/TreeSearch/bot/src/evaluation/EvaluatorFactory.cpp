#include "evaluation/EvaluatorFactory.h"

#include "evaluation/NetworkEvaluator.h"

namespace hex {

std::unique_ptr<Evaluator> makeEvaluator(std::uint64_t seed) {
    // Untrained: replace with weights loaded from EvalNetwork/weights
    return std::make_unique<NetworkEvaluator>(Network::withRandomWeights(seed));
}

}  // namespace hex
