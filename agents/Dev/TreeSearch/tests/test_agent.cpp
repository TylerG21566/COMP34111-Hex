// Run from agents/Dev:
//   g++ -std=c++20 -O2 -Wall -Wextra -I TreeSearch/bot/include -o /tmp/test_agent TreeSearch/tests/test_agent.cpp TreeSearch/bot/src/*/*.cpp && /tmp/test_agent

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <string>

#include "evaluation/NetworkEvaluator.h"
#include "mcts/Search.h"
#include "protocol/Protocol.h"

using namespace hex;

static int failures = 0;

static void check(bool condition, const std::string& name) {
    std::cout << (condition ? "ok    " : "FAIL  ") << name << "\n";
    if (!condition) ++failures;
}

static int cell(int x, int y) { return x * kSize + y; }

// Red on column 5 except row 10; Blue on 10 cells that connect nothing.
static std::string nearlyWonByRed(bool blueHasExtraStone) {
    std::string text;
    for (int x = 0; x < kSize; ++x) {
        std::string row(kSize, '0');
        if (x < kSize - 1) {
            row[5] = 'R';
            if (x % 2 == 0) row[0] = 'B';
            if (x % 2 == 0 && x > 0) row[10] = 'B';
        }
        if (x == 10 && blueHasExtraStone) row[0] = 'B';
        text += (x ? "," : "") + row;
    }
    return text;
}

static void testNetwork() {
    // The same position for each colour: stones mirrored and colours swapped.
    // Blue moves in the first (3 Red, 2 Blue), Red in the second.
    std::string blueText, redText;
    for (int x = 0; x < kSize; ++x) {
        blueText += (x ? "," : "") + std::string(kSize, '0');
        redText += (x ? "," : "") + std::string(kSize, '0');
    }
    auto put = [](std::string& text, int x, int y, char c) { text[x * (kSize + 1) + y] = c; };
    const int redStones[][2] = {{1, 2}, {5, 6}, {9, 3}};
    const int blueStones[][2] = {{4, 4}, {7, 10}};
    for (auto& [x, y] : redStones) put(blueText, x, y, 'R'), put(redText, y, x, 'B');
    for (auto& [x, y] : blueStones) put(blueText, x, y, 'B'), put(redText, y, x, 'R');
    const Board blue = Board::fromProtocol(blueText), red = Board::fromProtocol(redText);
    check(blue.toMove() == Colour::Blue && red.toMove() == Colour::Red, "network: test positions");

    bool inRange = true, repeats = true, symmetric = true, differs = false;
    for (int seed = 0; seed < 20; ++seed) {
        NetworkEvaluator evaluator(Network::withRandomWeights(seed));
        NetworkEvaluator same(Network::withRandomWeights(seed));
        NetworkEvaluator other(Network::withRandomWeights(seed + 100));
        const Evaluation a = evaluator.evaluate(blue), b = evaluator.evaluate(red);

        float total = 0.0f;
        for (float p : a.policy) {
            inRange &= std::isfinite(p) && p >= 0.0f;
            total += p;
        }
        inRange &= std::fabs(total - 1.0f) < 1e-4f && std::fabs(a.value) <= 1.0f;

        const Evaluation again = same.evaluate(blue);
        repeats &= again.value == a.value && again.policy == a.policy;
        differs |= other.evaluate(blue).value != a.value;

        symmetric &= a.value == b.value;
        for (int x = 0; x < kSize; ++x) {
            for (int y = 0; y < kSize; ++y) symmetric &= a.policy[cell(x, y)] == b.policy[cell(y, x)];
        }
    }
    check(inRange, "network: value in [-1, 1], policy sums to 1");
    check(repeats, "network: same seed gives the same output");
    check(differs, "network: other weights give another output");
    check(symmetric, "network: both colours are treated the same");
}

// The search must find forced results even when the evaluator knows nothing.
static void testSearch() {
    bool findsWin = true, findsBlock = true, provesWin = true, seesLoss = true;
    for (int seed = 0; seed < 20; ++seed) {
        NetworkEvaluator evaluator(Network::withRandomWeights(seed));
        Search search(evaluator, SearchConfig{});

        // Red wins at (10,4) or (10,5), the two cells below (9,5)
        Board red = Board::fromProtocol(nearlyWonByRed(true));
        if (red.toMove() != Colour::Red) { findsWin = false; break; }
        const int chosen = search.run(red).bestCell;
        findsWin &= chosen == cell(10, 4) || chosen == cell(10, 5);

        // With (10,4) filled by Blue, Blue must block at (10,5)
        std::string text = nearlyWonByRed(false);
        text[10 * (kSize + 1) + 4] = 'B';
        text[8 * (kSize + 1) + 0] = '0';
        Board blue = Board::fromProtocol(text);
        if (blue.toMove() != Colour::Blue) { findsBlock = false; break; }
        findsBlock &= search.run(blue).bestCell == cell(10, 5);

        // Without (9,5), that cell threatens both (10,4) and (10,5). An untrained
        // evaluator can mislead the search for a while, so it gets more time.
        std::string fork = nearlyWonByRed(true);
        fork[9 * (kSize + 1) + 5] = '0';
        fork[10 * (kSize + 1) + 0] = '0';
        Board forkBoard = Board::fromProtocol(fork);
        if (forkBoard.toMove() != Colour::Red) { provesWin = false; break; }
        SearchConfig longConfig;
        longConfig.simulations = 100000;
        longConfig.timeLimit = std::chrono::seconds(20);
        Search longSearch(evaluator, longConfig);
        provesWin &= longSearch.run(forkBoard).value == 1.0f;

        // Blue cannot stop both (10,4) and (10,5), but must still play legally.
        // The search stops at the last unproven move, so its value is not checked.
        Board lost = Board::fromProtocol(nearlyWonByRed(false));
        if (lost.toMove() != Colour::Blue) { seesLoss = false; break; }
        const SearchResult result = search.run(lost);
        seesLoss &= result.bestCell >= 0 && lost.isEmpty(result.bestCell);
    }
    check(findsWin, "search plays a win in one (20 seeds)");
    check(findsBlock, "search blocks a loss in one (20 seeds)");
    check(provesWin, "search proves a double threat wins (20 seeds)");
    check(seesLoss, "search plays a legal move when every move loses (20 seeds)");
}

int main() {
    const std::string empty = [] {
        std::string text;
        for (int x = 0; x < kSize; ++x) text += (x ? "," : "") + std::string(kSize, '0');
        return text;
    }();

    Board board = Board::fromProtocol(empty);
    check(board.toMove() == Colour::Red, "Red moves first");

    // Red goes straight down column 0, Blue plays along column 5
    bool early = false, won = false;
    for (int i = 0; i < kSize; ++i) {
        won = board.play(cell(i, 0));
        if (i < kSize - 1) early |= won | board.play(cell(i, 5));
    }
    check(!early && won, "Red wins from top to bottom, and not before");

    board = Board::fromProtocol(empty);
    early = false;
    for (int i = 0; i < kSize; ++i) {
        early |= board.play(cell(i < 5 ? 0 : 10, i));
        won = board.play(cell(5, i));
        if (i < kSize - 1) early |= won;
    }
    check(!early && won, "Blue wins from left to right, and not before");

    board = Board::fromProtocol(empty);
    board.play(cell(1, 2));
    const std::string afterSwap = "SWAP;;0R" + empty.substr(2) + ";3;";
    check(parseRequest(afterSwap).board.toMove() == Colour::Blue, "Blue moves after a swap");
    check(formatMove(cell(3, 10)) == "3,10", "moves are written as x,y");

    for (const std::string& bad : {std::string(""), std::string("CHANGE;0,0;000,000,000;2;")}) {
        bool threw = false;
        try { parseRequest(bad); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "invalid message is refused: '" + bad + "'");
    }

    // Random games: findWinningMove must agree with trying every move
    std::mt19937 rng(1);
    bool agrees = true;
    for (int game = 0; game < 500; ++game) {
        Board b = Board::fromProtocol(empty);
        while (true) {
            bool canWin = false;
            for (int c = 0; c < kCells; ++c) {
                Board copy = b;
                if (b.isEmpty(c)) canWin |= copy.play(c);
            }
            const int found = b.findWinningMove();
            Board copy = b;
            agrees &= found >= 0 ? b.isEmpty(found) && copy.play(found) : !canWin;

            int c;
            do { c = static_cast<int>(rng() % kCells); } while (!b.isEmpty(c));
            if (b.play(c)) break;
        }
    }
    check(agrees, "findWinningMove agrees with trying every move (500 games)");

    testNetwork();
    testSearch();

    std::cout << (failures ? "FAILED\n" : "all passed\n");
    return failures ? 1 : 0;
}
