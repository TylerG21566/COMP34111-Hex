#pragma once

#include <chrono>
#include <cstdint>
#include <vector>

#include "evaluation/Evaluator.h"
#include "game/Board.h"

namespace hex {

struct SearchConfig {
    int simulations = 10000;                    // per move, at most
    std::chrono::milliseconds timeLimit{1000};  // per move, at most
    float cPuct = 1.5f;                         // exploration strength
};

struct SearchResult {
    int bestCell = -1;
    float value = 0.0f;  // for the side to move: -1 (loss) to 1 (win)
    int simulations = 0;
};

// Single-threaded Monte Carlo tree search with the PUCT rule, as in AlphaZero:
// leaves are scored by the evaluator and there are no random playouts. Results
// that are certain replace the evaluator's estimates (MCTS-Solver). The tree is
// rebuilt for each move.
class Search {
public:
    Search(Evaluator& evaluator, SearchConfig config)
        : evaluator_(evaluator), config_(config) {}

    // The board must have an empty cell and no winner.
    SearchResult run(const Board& root);

private:
    // Whether the player who makes a move is certain to win or lose after it
    enum class Proof : std::uint8_t { None, Win, Loss };

    // A move out of a node. Values are for the player who makes the move.
    struct Edge {
        float prior = 0.0f;
        float valueSum = 0.0f;
        int visits = 0;
        int child = -1;  // node index, -1 until expanded
        std::uint8_t cell = 0;
        Proof proof = Proof::None;
    };
    struct Node {
        int firstEdge = 0;
        int edgeCount = 0;
    };
    struct Step {
        int node;
        int edge;
    };

    // Adds a node for `board`, which must have no winning move. Returns the
    // evaluator's value for the side to move.
    float expand(const Board& board, int& nodeIndex);
    int selectEdge(const Node& node) const;
    void simulate(const Board& root);
    // Records the proof on the last step of path_ and follows it up the tree.
    void propagateProof(Proof proof);
    // Moves of the node that are not proven losses
    int openMoves(const Node& node) const;
    bool hasProvenWin(const Node& node) const;

    Evaluator& evaluator_;
    SearchConfig config_;
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
    std::vector<Step> path_;  // moves followed in the current simulation
};

}  // namespace hex
