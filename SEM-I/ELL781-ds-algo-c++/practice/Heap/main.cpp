#include <iostream>
#include "heap_node.h"
#include "min_heap.h";

using namespace std;

int main()
{
    MINheap* heaptree1 = new(nothrow) MINheap(20);
    heaptree1->insertNode(30);
    heaptree1->insertNode(40);
    heaptree1->insertNode(50);
    heaptree1->insertNode(60);
    heaptree1->insertNode(70);
    heaptree1->deleteMINheap();
    cout << "Hello world!" << endl;
    return 0;
}
