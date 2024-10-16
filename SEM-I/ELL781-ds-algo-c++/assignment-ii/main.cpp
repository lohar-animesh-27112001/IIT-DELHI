// # include "puzzle_solver.h"
# include "puzzle_solver_dma.h"
// # include "puzzle_solver_LL.h"

using namespace std;

// Main function
int main() {
    // Initialize river and not_success arrays
    string river[2] = {"PWCG", ""};
    string not_success[2] = {"WG", "CG"};

    // Create PuzzleSolve object using automatic memory management
    PuzzleSolve puzzle(river, not_success);

    // Start counting successful transitions
    puzzle.count_successes();

    cout << "Hello World" << endl;
    return 0;
}
