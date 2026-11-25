#include "game/Board.h"

#include <stdexcept>

namespace hex {

namespace {

constexpr int kDx[] = {0, 0, -1, -1, 1, 1};
constexpr int kDy[] = {-1, 1, 0, 1, -1, 0};

// Calls f(neighbour) for each neighbour of `cell` on the board
template <typename F>
void forEachNeighbour(int cell, F f) {
    const int x = cell / kSize, y = cell % kSize;
    for (int i = 0; i < 6; ++i) {
        const int nx = x + kDx[i], ny = y + kDy[i];
        if (nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) f(nx * kSize + ny);
    }
}

// How far `cell` is along the direction `colour` connects: 0 to kSize - 1
int distanceAlong(int cell, Colour colour) {
    return colour == Colour::Red ? cell / kSize : cell % kSize;
}

}  // namespace

Board Board::fromProtocol(const std::string& text) {
    Board board;
    int x = 0, y = 0, red = 0, blue = 0;
    for (char c : text) {
        if (c == ',') {
            if (y != kSize) throw std::invalid_argument("board row has the wrong length");
            ++x;
            y = 0;
            continue;
        }
        if (x >= kSize || y >= kSize) throw std::invalid_argument("board is too large");
        Colour& cell = board.cells_[x * kSize + y];
        if (c == 'R') {
            cell = Colour::Red;
            ++red;
        } else if (c == 'B') {
            cell = Colour::Blue;
            ++blue;
        } else if (c != '0') {
            throw std::invalid_argument("unknown character in board");
        }
        ++y;
    }
    if (x != kSize - 1 || y != kSize) throw std::invalid_argument("board is too small");

    // Red has the first stone and a swap adds none, so the stone counts always
    // give the side to move, before and after a swap.
    board.toMove_ = red > blue ? Colour::Blue : Colour::Red;
    return board;
}

bool Board::play(int cell) {
    cells_[cell] = toMove_;
    toMove_ = opposite(toMove_);
    return connectsSides(cell);
}

bool Board::connectsSides(int cell) const {
    const Colour colour = cells_[cell];
    std::array<int, kCells> stack;
    int stackSize = 0;
    std::bitset<kCells> seen;
    bool firstSide = false, lastSide = false;

    stack[stackSize++] = cell;
    seen.set(cell);
    while (stackSize > 0) {
        const int current = stack[--stackSize];
        firstSide |= distanceAlong(current, colour) == 0;
        lastSide |= distanceAlong(current, colour) == kSize - 1;
        if (firstSide && lastSide) return true;

        forEachNeighbour(current, [&](int next) {
            if (!seen[next] && cells_[next] == colour) {
                seen.set(next);
                stack[stackSize++] = next;
            }
        });
    }
    return false;
}

std::bitset<kCells> Board::stonesConnectedTo(bool lastSide) const {
    std::array<int, kCells> stack;
    int stackSize = 0;
    std::bitset<kCells> seen;
    auto visit = [&](int cell) {
        if (!seen[cell] && cells_[cell] == toMove_) {
            seen.set(cell);
            stack[stackSize++] = cell;
        }
    };

    const int line = lastSide ? kSize - 1 : 0;
    for (int i = 0; i < kSize; ++i) {
        visit(toMove_ == Colour::Red ? line * kSize + i : i * kSize + line);
    }
    while (stackSize > 0) forEachNeighbour(stack[--stackSize], visit);
    return seen;
}

int Board::findWinningMove() const {
    const std::bitset<kCells> first = stonesConnectedTo(false);
    const std::bitset<kCells> last = stonesConnectedTo(true);
    for (int cell = 0; cell < kCells; ++cell) {
        if (!isEmpty(cell)) continue;
        bool touchesFirst = distanceAlong(cell, toMove_) == 0;
        bool touchesLast = distanceAlong(cell, toMove_) == kSize - 1;
        forEachNeighbour(cell, [&](int next) {
            touchesFirst |= first[next];
            touchesLast |= last[next];
        });
        if (touchesFirst && touchesLast) return cell;
    }
    return -1;
}

}  // namespace hex
