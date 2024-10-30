#include <bits/stdc++.h>
#include "graph_LL.h"
#include "priority_queue_LL.h"
#include "kruskal_algo_LL.h"
#include "prims_algo_LL.h"

using namespace std;

int main() {
    // Read input from file (example provided inline for simplicity)
    int n = 6;
    vector<vector<int>> costMatrix = {
        {0, 6, 1, 5, -1, -1},
        {6, 0, 5, -1, 3, -1},
        {1, 5, 0, 5, 6, 4},
        {5, -1, 5, 0, -1, 2},
        {-1, 3, 6, -1, 0, 6},
        {-1, -1, 4, 2, 6, 0}
    };

    Graph graph(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (costMatrix[i][j] > 0)
                graph.addEdge(i, j, costMatrix[i][j]);

    // Run Prim's algorithm
    auto start = high_resolution_clock::now();
    Graph primMSTGraph = primMST(graph);
    auto end = high_resolution_clock::now();
    printMST(primMSTGraph, "Prim's algorithm", duration_cast<milliseconds>(end - start));

    // Run Kruskal's algorithm
    start = high_resolution_clock::now();
    Graph kruskalMSTGraph = kruskalMST(graph);
    end = high_resolution_clock::now();
    printMST(kruskalMSTGraph, "Kruskal's algorithm", duration_cast<milliseconds>(end - start));

    return 0;
}
