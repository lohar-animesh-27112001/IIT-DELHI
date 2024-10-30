#ifndef HEADER11_H
#define HEADER11_H

#include <bits/stdc++.h>
#include "graph_LL.h"

using namespace std;
using namespace std::chrono;

// PriorityQueue class with DECREASE-KEY, INSERT, and DELETEMIN operations
class PriorityQueue {
    Node* head;

public:
    PriorityQueue() : head(nullptr) {}

    void insert(int vertex, int key) {
        Node* newNode = new Node(vertex, key);
        newNode->next = head;
        head = newNode;
    }

    int deleteMin() {
        if (!head) return -1;

        Node* minNode = head;
        Node* minPrev = nullptr;
        Node* prev = nullptr;
        Node* curr = head;

        while (curr) {
            if (curr->key < minNode->key) {
                minNode = curr;
                minPrev = prev;
            }
            prev = curr;
            curr = curr->next;
        }

        int minVertex = minNode->vertex;
        if (minPrev) minPrev->next = minNode->next;
        else head = minNode->next;
        delete minNode;

        return minVertex;
    }

    void decreaseKey(int vertex, int newKey) {
        Node* curr = head;
        while (curr) {
            if (curr->vertex == vertex) {
                curr->key = newKey;
                break;
            }
            curr = curr->next;
        }
    }

    bool isEmpty() {
        return head == nullptr;
    }
};

#endif
#pragma once
