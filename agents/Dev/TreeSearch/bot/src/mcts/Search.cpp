#include "mcts/Search.h"

#include <cmath>
#include <limits>

namespace hex {

SearchResult Search::run(const Board& root) {
    const auto deadline = std::chrono::steady_clock::now() + config_.timeLimit;

    SearchResult result;
    result.bestCell = root.findWinningMove();
    if (result.bestCell >= 0) {
        result.value = 1.0f;
        return result;
    }

    nodes_.clear();
    edges_.clear();
    int rootIndex = -1;
    expand(root, rootIndex);

    // Stop early when a move is proven to win, or at most one does not lose
    bool decided = false;
    while (!decided && result.simulations < config_.simulations &&
           std::chrono::steady_clock::now() < deadline) {
        simulate(root);
        ++result.simulations;
        decided = hasProvenWin(nodes_[rootIndex]) || openMoves(nodes_[rootIndex]) <= 1;
    }

    // A proven win is best and a proven loss is worst. Otherwise play the most
    // visited move; the prior breaks ties.
    const Node node = nodes_[rootIndex];
    auto rank = [](Proof proof) {
        return proof == Proof::Win ? 2 : proof == Proof::None ? 1 : 0;
    };
    const Edge* best = nullptr;
    float valueSum = 0.0f;
    int visits = 0;
    for (int i = 0; i < node.edgeCount; ++i) {
        const Edge& edge = edges_[node.firstEdge + i];
        if (edge.proof != Proof::Loss) {
            valueSum += edge.valueSum;
            visits += edge.visits;
        }
        if (best == nullptr || rank(edge.proof) > rank(best->proof) ||
            (edge.proof == best->proof &&
             (edge.visits > best->visits ||
              (edge.visits == best->visits && edge.prior > best->prior)))) {
            best = &edge;
        }
    }
    if (best == nullptr) return result;

    result.bestCell = best->cell;
    if (best->proof == Proof::Win) {
        result.value = 1.0f;
    } else if (best->proof == Proof::Loss) {
        result.value = -1.0f;
    } else if (visits > 0) {
        result.value = valueSum / visits;
    }
    return result;
}

float Search::expand(const Board& board, int& nodeIndex) {
    const Evaluation evaluation = evaluator_.evaluate(board);

    Node node;
    node.firstEdge = static_cast<int>(edges_.size());
    float total = 0.0f;
    for (int cell = 0; cell < kCells; ++cell) {
        if (!board.isEmpty(cell)) continue;
        Edge edge;
        edge.cell = static_cast<std::uint8_t>(cell);
        edge.prior = evaluation.policy[cell];
        total += edge.prior;
        edges_.push_back(edge);
        ++node.edgeCount;
    }
    for (int i = 0; i < node.edgeCount; ++i) {
        Edge& edge = edges_[node.firstEdge + i];
        edge.prior = total > 0.0f ? edge.prior / total : 1.0f / node.edgeCount;
    }

    nodeIndex = static_cast<int>(nodes_.size());
    nodes_.push_back(node);
    return evaluation.value;
}

int Search::openMoves(const Node& node) const {
    int count = 0;
    for (int i = 0; i < node.edgeCount; ++i) {
        if (edges_[node.firstEdge + i].proof != Proof::Loss) ++count;
    }
    return count;
}

bool Search::hasProvenWin(const Node& node) const {
    for (int i = 0; i < node.edgeCount; ++i) {
        if (edges_[node.firstEdge + i].proof == Proof::Win) return true;
    }
    return false;
}

// The node is never proven: simulate() does not enter a proven node and run()
// stops when the root is proven. So there is always a move that does not lose.
int Search::selectEdge(const Node& node) const {
    float valueSum = 0.0f;
    int visits = 0;
    for (int i = 0; i < node.edgeCount; ++i) {
        const Edge& edge = edges_[node.firstEdge + i];
        if (edge.proof == Proof::Loss) continue;
        valueSum += edge.valueSum;
        visits += edge.visits;
    }
    // Unvisited moves are assumed to be as good as the node's average
    const float unvisitedValue = visits > 0 ? valueSum / visits : 0.0f;
    const float explore = config_.cPuct * std::sqrt(static_cast<float>(visits + 1));

    int best = -1;
    float bestScore = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < node.edgeCount; ++i) {
        const Edge& edge = edges_[node.firstEdge + i];
        if (edge.proof == Proof::Loss) continue;
        const float q = edge.visits > 0 ? edge.valueSum / edge.visits : unvisitedValue;
        const float score = q + explore * edge.prior / (1 + edge.visits);
        if (score > bestScore) {
            bestScore = score;
            best = node.firstEdge + i;
        }
    }
    return best;
}

void Search::simulate(const Board& root) {
    Board board = root;
    path_.clear();
    int nodeIndex = 0;
    float value;  // for the player who made the last move on the path

    // Go down the tree to a new position. Every node in the tree has no winning
    // move, so the moves on the way never end the game.
    while (true) {
        const int edgeIndex = selectEdge(nodes_[nodeIndex]);
        path_.push_back({nodeIndex, edgeIndex});
        board.play(edges_[edgeIndex].cell);

        if (edges_[edgeIndex].child >= 0) {
            nodeIndex = edges_[edgeIndex].child;
            continue;
        }
        if (board.findWinningMove() >= 0) {
            value = -1.0f;
            propagateProof(Proof::Loss);
        } else {
            // expand() can move edges_, so use the index and not a reference
            int child = -1;
            value = -expand(board, child);
            edges_[edgeIndex].child = child;
        }
        break;
    }

    // Back up the result; what is good for one player is bad for the other
    for (auto step = path_.rbegin(); step != path_.rend(); ++step) {
        Edge& edge = edges_[step->edge];
        if (edge.proof != Proof::None) value = edge.proof == Proof::Win ? 1.0f : -1.0f;
        edge.visits += 1;
        edge.valueSum += value;
        value = -value;
    }
}

void Search::propagateProof(Proof proof) {
    for (auto step = path_.rbegin(); step != path_.rend(); ++step) {
        edges_[step->edge].proof = proof;
        if (proof == Proof::Win) {
            // The player to move at this node wins, so the move into it loses
            proof = Proof::Loss;
        } else if (openMoves(nodes_[step->node]) == 0) {
            // Every move at this node loses, so the move into it wins
            proof = Proof::Win;
        } else {
            return;
        }
    }
}

}  // namespace hex
