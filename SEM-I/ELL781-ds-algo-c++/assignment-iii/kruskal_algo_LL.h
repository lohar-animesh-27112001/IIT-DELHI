#ifndef HEADER13_H
#define HEADER13_H

#include <bits/stdc++.h>
#include "graph_LL.h"
#include "priority_queue_LL.h"
#include "union_find_LL.h"

using namespace std;

// Kruskal's MST Algorithm
Graph kruskalMST(Graph& graph) {
    int vertices = graph.getVertices();
    vector<Edge> edges;
    Graph mst(vertices);

    for (int u = 0; u < vertices; ++u)
        for (const Edge& edge : graph.getEdges(u))
            if (edge.src < edge.dest)
                edges.push_back(edge);

    sort(edges.begin(), edges.end(), [](Edge a, Edge b) { return a.weight < b.weight; });

    UnionFind uf(vertices);
    for (const Edge& edge : edges) {
        int uRoot = uf.find(edge.src);
        int vRoot = uf.find(edge.dest);
        if (uRoot != vRoot) {
            mst.addEdge(edge.src, edge.dest, edge.weight);
            uf.unionSets(uRoot, vRoot);
        }
    }

    return mst;
}

#endif
#pragma once
