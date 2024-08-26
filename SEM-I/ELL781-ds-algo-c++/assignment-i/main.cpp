#include <bits/stdc++.h>
#include "fibonacci_heap.h"

using namespace std;

int main() {
    FibonacciHeap* heap = new(nothrow) FibonacciHeap;
    if (heap == NULL) {
        cout << "Memory allocation failed!" << endl;
        return 1;
    }

    heap->insertFibHeapNode(20, false);
    heap->insertFibHeapNode(23, false);
    heap->insertFibHeapNode(54, false);
    heap->insertFibHeapNode(13, false);
    heap->insertFibHeapNode(3, false);
    heap->insertFibHeapNode(42, false);

    heap->printAllNodeDegrees();
    heap->deleteMin();
    heap->printAllNodeDegrees();

    heap->decreaseKey(54, 19);
    heap->printHeap();

    heap->deleteFibonacciHeap();

    cout << "Hello world!" << endl;
    return 0;
}
