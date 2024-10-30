#ifndef HEADER12_H
#define HEADER12_H

#include <bits/stdc++.h>
#include "graph_LL.h"
#include "priority_queue_LL.h"

using namespace std;

// Prim's MST Algorithm
Graph primMST(Graph& graph) {
    int vertices = graph.getVertices();
    PriorityQueue pq;
    vector<int> key(vertices, INT_MAX), parent(vertices, -1);
    vector<bool> inMST(vertices, false);

    pq.insert(0, 0);
    key[0] = 0;

    Graph mst(vertices);

    while (!pq.isEmpty()) {
        int u = pq.deleteMin();
        inMST[u] = true;

        for (const Edge& edge : graph.getEdges(u)) {
            int v = edge.dest;
            if (!inMST[v] && edge.weight < key[v]) {
                key[v] = edge.weight;
                pq.decreaseKey(v, key[v]);
                parent[v] = u;
                pq.insert(v, key[v]);
            }
        }
    }

    for (int i = 1; i < vertices; ++i)
        if (parent[i] != -1)
            mst.addEdge(parent[i], i, key[i]);

    return mst;
}

#endif
#pragma once
