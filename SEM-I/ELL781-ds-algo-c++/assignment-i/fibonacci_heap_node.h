// header7
#ifndef HEADER5_H
#define HEADER5_H

#include <bits/stdc++.h>

using namespace std;

#endif
#pragma once

class FibonacciHeapNode {
public:
    bool marked;
    int key;
    FibonacciHeapNode* next;
    FibonacciHeapNode* prev;
    FibonacciHeapNode* parent;
    FibonacciHeapNode* child;
    int degree;
};
