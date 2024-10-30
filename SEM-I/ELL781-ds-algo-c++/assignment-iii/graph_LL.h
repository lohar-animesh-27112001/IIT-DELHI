#ifndef HEADER10_H
#define HEADER10_H

#include <bits/stdc++.h>

using namespace std;
using namespace std::chrono; // Using the chrono namespace for time measurement

// Define Edge structure
struct Edge {
    int src, dest, weight;
    Edge(int s, int d, int w) : src(s), dest(d), weight(w) {}
};

// Node structure for PriorityQueue using linked list
struct Node {
    int vertex, key;
    Node* next;
    Node(int v, int k) : vertex(v), key(k), next(nullptr) {}
};

// Graph class to represent weighted, undirected graph using adjacency list
class Graph {
    int vertices;
    vector<vector<Edge>> adjList;

public:
    Graph(int v) : vertices(v), adjList(v) {}

    void addEdge(int u, int v, int w) {
        adjList[u].emplace_back(u, v, w);
        adjList[v].emplace_back(v, u, w);
    }

    int getVertices() const { return vertices; }
    const vector<Edge>& getEdges(int u) const { return adjList[u]; }
};

// Function to print MST and calculate total weight
void printMST(Graph& mst, const string& algoName, milliseconds duration) {
    int totalCost = 0;
    cout << algoName << " MST: " << endl;
    for (int u = 0; u < mst.getVertices(); ++u) {
        for (const Edge& edge : mst.getEdges(u)) {
            if (edge.src < edge.dest) {
                cout << "(" << edge.src + 1 << "," << edge.dest + 1 << ") " << endl;
                totalCost += edge.weight;
            }
        }
    }
    cout << "(total cost:" << totalCost << "; runtime: " << duration.count() << "ms)" << endl;
}

#endif
#pragma once
