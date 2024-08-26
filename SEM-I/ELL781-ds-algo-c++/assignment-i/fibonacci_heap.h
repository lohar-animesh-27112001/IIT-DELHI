// header8
#ifndef HEADER5_H
#define HEADER5_H

#include <bits/stdc++.h>
#include "fibonacci_heap_node.h"

using namespace std;

#endif
#pragma once

class FibonacciHeap {
private:
    FibonacciHeapNode* MIN;
    int count;

public:
    // Creation of Heap
    FibonacciHeap () {
        this->MIN = NULL;
        count = 0;
    }

    FibonacciHeapNode* findMIN() {
        cout << "MIN elements key value is: " << this->MIN->key << endl;
        return MIN;
    }

    // Inserting Fibonacci Heap Node
    int insertFibHeapNode(int value, bool marked) {
        FibonacciHeapNode* newNode = new(nothrow) FibonacciHeapNode;
        if (newNode == NULL) {
            cout << "Memory allocation failed!" << endl;
            return 0;
        }

        newNode->key = value;
        newNode->marked = marked;
        newNode->degree = 0;
        newNode->parent = NULL;
        newNode->child = NULL;
        newNode->next = newNode;
        newNode->prev = newNode;

        if (this->MIN == NULL) {
            this->MIN = newNode;
        } else {

            newNode->next = this->MIN->next;
            newNode->prev = this->MIN;
            this->MIN->next->prev = newNode;
            this->MIN->next = newNode;
            if (newNode->key < this->MIN->key) {
                this->MIN = newNode;
            }
        }
        this->count++;
        cout << "Inserted the node, value: " << value << endl;
        return 1;
    }

    // Union operation between two Fibonacci heaps
    int unionFibonacciHeap(FibonacciHeap* fibHeap) {
        if (fibHeap == NULL || fibHeap->MIN == NULL) {
            return 0;
        }

        if (this->MIN == NULL) {
            this->MIN = fibHeap->MIN;
            this->count = fibHeap->count;
            return 1;
        }
        FibonacciHeapNode* temp = this->MIN->next;
        this->MIN->next = fibHeap->MIN->next;
        fibHeap->MIN->next->prev = this->MIN;
        fibHeap->MIN->next = temp;
        temp->prev = fibHeap->MIN;
        if (fibHeap->MIN->key < this->MIN->key) {
            this->MIN = fibHeap->MIN;
        }

        this->count += fibHeap->count;
        return 1;
    }

    // Delete MIN operation
    void deleteMin() {
        if (this->MIN == NULL) {
            cout << "Heap is empty, nothing to delete." << endl;
            return;
        }

        FibonacciHeapNode* oldMin = this->MIN;
        if (this->MIN->child != NULL) {
            FibonacciHeapNode* child = this->MIN->child;
            do {
                child->parent = NULL;
                child = child->next;
            } while (child != this->MIN->child);
            FibonacciHeapNode* minNext = this->MIN->next;
            FibonacciHeapNode* childPrev = this->MIN->child->prev;
            this->MIN->child->prev = this->MIN->prev;
            this->MIN->prev->next = this->MIN->child;
            childPrev->next = minNext;
            minNext->prev = childPrev;
        }
        this->MIN->prev->next = this->MIN->next;
        this->MIN->next->prev = this->MIN->prev;

        if (this->MIN == this->MIN->next) {
            delete this->MIN;
            this->MIN = NULL;
        } else {
            this->MIN = this->MIN->next;
            consolidate();
        }

        count--;
        cout << "Deleted minimum node with value: " << oldMin->key << endl;
        delete oldMin;
    }

    void printAllNodeDegrees() {
        if (MIN == nullptr) {
            cout << "Heap is empty!" << endl;
            return;
        }
        printDegrees(MIN);
    }

    void printHeap() {
        if (MIN == nullptr) {
            cout << "Heap is empty!" << endl;
            return;
        }

        cout << "Fibonacci Heap Structure:" << endl;
        printHeap(MIN, 0);
    }

    void decreaseKey(int prevKey, int newKey) {
        if (prevKey <= newKey) {
            cout << "New key is not smaller than the current key." << endl;
            return;
        }

        FibonacciHeapNode* node = findNode(prevKey);
        if (node == nullptr) {
            cout << "Node with key " << prevKey << " not found." << endl;
            return;
        }

        node->key = newKey;
        FibonacciHeapNode* parent = node->parent;

        if (parent && node->key < parent->key) {
            cut(node, parent);
            cascadingCut(parent);
        }

        if (node->key < MIN->key) {
            MIN = node;
        }
    }

    void deleteHeapNode(int value) {
        decreaseKey(value, -2147483645);
        deleteMin();
    }

    void deleteFibonacciHeap() {
        deleteFibonacciHeap(this->MIN);
        this->MIN = NULL;
    }

private:
    void deleteFibonacciHeap(FibonacciHeapNode* node) {
        if (node == NULL) return;

        FibonacciHeapNode* start = node;
        do {
            if (node->child != NULL) {
                deleteFibonacciHeap(node->child);
            }
            FibonacciHeapNode* temp = node;
            node = node->next;
            cout << "Deleted node: " << temp->key << endl;
            delete temp;
        } while (node != start);
    }

    void printDegrees(FibonacciHeapNode* node) {
        if (node == nullptr) {
            return;
        }

        FibonacciHeapNode* start = node;
        do {
            cout << "Node with key " << node->key << " has degree " << node->degree << endl;
            if (node->child != nullptr) {
                printDegrees(node->child);
            }
            node = node->next;
        } while (node != start);
    }

    void consolidate() {
        if (this->MIN == NULL) return;

        int maxDegree = floor(log2(count)) + 1;
        vector<FibonacciHeapNode*> degreeTable(maxDegree, NULL);

        vector<FibonacciHeapNode*> rootList;
        FibonacciHeapNode* current = this->MIN;
        do {
            rootList.push_back(current);
            current = current->next;
        } while (current != this->MIN);

        for (FibonacciHeapNode* node : rootList) {
            int degree = node->degree;
            while (degreeTable[degree] != NULL) {
                FibonacciHeapNode* other = degreeTable[degree];
                if (node->key > other->key) {
                    swap(node, other);
                }
                linkHeaps(node, other);
                degreeTable[degree] = NULL;
                degree++;
            }
            degreeTable[degree] = node;
        }

        this->MIN = NULL;
        for (FibonacciHeapNode* node : degreeTable) {
            if (node != NULL) {
                if (this->MIN == NULL) {
                    this->MIN = node;
                    node->next = node;
                    node->prev = node;
                } else {
                    node->next = this->MIN->next;
                    node->prev = this->MIN;
                    this->MIN->next->prev = node;
                    this->MIN->next = node;
                    if (node->key < this->MIN->key) {
                        this->MIN = node;
                    }
                }
            }
        }
    }

    // Link two Fibonacci Heap trees of the same degree
    void linkHeaps(FibonacciHeapNode* minNode, FibonacciHeapNode* other) {
        other->prev->next = other->next;
        other->next->prev = other->prev;

        other->parent = minNode;
        if (minNode->child == NULL) {
            minNode->child = other;
            other->next = other;
            other->prev = other;
        } else {
            other->next = minNode->child;
            other->prev = minNode->child->prev;
            minNode->child->prev->next = other;
            minNode->child->prev = other;
        }
        minNode->degree++;
        other->marked = false;
    }

    void cut(FibonacciHeapNode* node, FibonacciHeapNode* parent) {
        if (node->next == node) {
            parent->child = nullptr;
        } else {
            node->prev->next = node->next;
            node->next->prev = node->prev;
            if (parent->child == node) {
                parent->child = node->next;
            }
        }
        parent->degree--;
        node->next = MIN->next;
        node->prev = MIN;
        MIN->next->prev = node;
        MIN->next = node;
        node->parent = nullptr;
        node->marked = false;
    }

    void cascadingCut(FibonacciHeapNode* node) {
        FibonacciHeapNode* parent = node->parent;
        if (parent != nullptr) {
            if (!node->marked) {
                node->marked = true;
            } else {
                cut(node, parent);
                cascadingCut(parent);
            }
        }
    }

    FibonacciHeapNode* findNode(int key) {
        if (MIN == nullptr) return nullptr;
        FibonacciHeapNode* start = MIN;
        do {
            if (start->key == key) return start;
            FibonacciHeapNode* child = start->child;
            if (child != nullptr) {
                FibonacciHeapNode* found = findNodeInChildList(child, key);
                if (found) return found;
            }
            start = start->next;


        } while (start != MIN);
            return nullptr;
        }
        FibonacciHeapNode* findNodeInChildList(FibonacciHeapNode* node, int key) {
        FibonacciHeapNode* start = node;
        do {
            if (start->key == key) return start;
            if (start->child != nullptr) {
                FibonacciHeapNode* found = findNodeInChildList(start->child, key);
                if (found) return found;
            }
            start = start->next;
        } while (start != node);
        return nullptr;
    }

    void printHeap(FibonacciHeapNode* node, int level) {
        if (node == nullptr) return;

        FibonacciHeapNode* start = node;
        do {
            for (int i = 0; i < level; ++i) {
                cout << "  ";
            }
            cout << "Key: " << node->key << " (Degree: " << node->degree << ", Marked: " << (node->marked ? "true" : "false") << ")" << endl;

            if (node->child != nullptr) {
                printHeap(node->child, level + 1);
            }
            node = node->next;
        } while (node != start);
    }

};
