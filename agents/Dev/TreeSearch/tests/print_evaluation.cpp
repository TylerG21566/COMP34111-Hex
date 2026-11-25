// Used by EvalNetwork/training/test_match.py: print_evaluation <weights file>
// Reads one board per line, and prints its value and the policy of each cell.

#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

#include "evaluation/NetworkEvaluator.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: print_evaluation <weights file>\n";
        return 2;
    }
    try {
        hex::NetworkEvaluator evaluator(hex::Network::load(argv[1]));
        std::cout << std::setprecision(9);
        std::string line;
        while (std::getline(std::cin, line)) {
            const hex::Evaluation result = evaluator.evaluate(hex::Board::fromProtocol(line));
            std::cout << result.value;
            for (float p : result.policy) std::cout << ' ' << p;
            std::cout << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << "print_evaluation: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
