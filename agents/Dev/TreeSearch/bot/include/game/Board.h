#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <string>

namespace hex {

constexpr int kSize = 11;
constexpr int kCells = kSize * kSize;

enum class Colour : std::uint8_t { Empty, Red, Blue };

inline Colour opposite(Colour c) { return c == Colour::Red ? Colour::Blue : Colour::Red; }

// Cell (x, y) is stored at index x * kSize + y: x is the row, y the column.
// Red connects rows 0 and 10, Blue connects columns 0 and 10.
class Board {
public:
    // Reads the protocol format: rows of R/B/0 joined by commas.
    // Throws std::invalid_argument if the text is not an 11x11 board.
    static Board fromProtocol(const std::string& text);

    Colour at(int cell) const { return cells_[cell]; }
    bool isEmpty(int cell) const { return cells_[cell] == Colour::Empty; }
    Colour toMove() const { return toMove_; }

    // Places a stone for the side to move and passes the turn.
    // Returns true if that stone wins the game.
    bool play(int cell);

    // A cell where the side to move wins at once, or -1 if there is none.
    int findWinningMove() const;

private:
    // True if the chain containing `cell` touches both of its colour's sides.
    bool connectsSides(int cell) const;
    // The side to move's stones that are connected to its first (row or column
    // 0) or last side.
    std::bitset<kCells> stonesConnectedTo(bool lastSide) const;

    std::array<Colour, kCells> cells_{};
    Colour toMove_ = Colour::Red;
};

}  // namespace hex
