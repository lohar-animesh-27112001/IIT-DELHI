#include <iostream>
#include <bits/stdc++.h>
#include "node.h"
#include "linked_list.h"
#include "doubly_node.h"
#include "doubly_linked_list.h"

using namespace std;

int main()
{
    cout << "SINGLY LINKED LIST:" << endl << endl;
    Linked_List* myLinkedList_1 = new(nothrow) Linked_List(20);
    myLinkedList_1->printLinkedList();
    myLinkedList_1->insertAtBegining(30);
    myLinkedList_1->insertAtEnd(10);
    myLinkedList_1->insertAtEnd(50);
    myLinkedList_1->printLinkedList();
    myLinkedList_1->deleteByValue(10);
    myLinkedList_1->printLinkedList();
    myLinkedList_1->deleteByValue(30);
    myLinkedList_1->deleteByValue(9);
    myLinkedList_1->printLinkedList();
    myLinkedList_1->insertAtBegining(100);
    myLinkedList_1->insertAtBegining(101);
    myLinkedList_1->printLinkedList();
    myLinkedList_1->insertAtEnd(200);
    myLinkedList_1->printLinkedList();
    myLinkedList_1->deleteLinkedList();
    delete myLinkedList_1;

    cout << endl << endl << "DOUBLY LINKED LIST :" << endl << endl;
    Doubly_Linked_List* myLinkedList_2 = new(nothrow) Doubly_Linked_List(100);
    myLinkedList_2->insertAtBegning(200);
    myLinkedList_2->insertAtBegning(300);
    myLinkedList_2->printDoublyLinkedList();
    myLinkedList_2->deleteDoublyLinkedList();
    delete myLinkedList_2;

    cout << "Hello world!" << endl;
    return 0;
}
