// header1.h
#ifndef HEADER4_H
#define HEADER4_H

#include <iostream>
#include <bits/stdc++.h>
#include "doubly_node.h"

using namespace std;
// Other declarations and definitions

#endif // HEADER2_H
#pragma once

// Declarations and definitions

class Doubly_Linked_List {
public:
    Doubly_Node* head;

    Doubly_Linked_List() {
        this->head = NULL;
    }

    Doubly_Linked_List(int value) {
        this->head = new(nothrow) Doubly_Node;
        this->head->data = value;
        this->head->next = NULL;
        this->head->prev = NULL;
    }

    int insertAtBegning(int value) {
        Doubly_Node* tempNode = new(nothrow) Doubly_Node;
        tempNode->data = value;
        Doubly_Node* temp = this->head;
        this->head = tempNode;
        this->head->next = temp;
        this->head->prev = NULL;
        temp->prev = this->head;
        return 0;
    }

    int printDoublyLinkedList() {
        Doubly_Node* temp = head;
        while(temp != NULL) {
            if(temp->next != NULL) {
                cout << temp->data << " -> ";
            } else {
                cout << temp->data << endl;
            }
            temp = temp->next;
        }
        return 0;
    }

    int deleteDoublyLinkedList() {
        Doubly_Node* tempNode = head;
        Doubly_Node* temp;
        while(tempNode != NULL) {
            temp = tempNode->next;
            delete tempNode;
            tempNode = temp;
        }
        cout << "Doubly linked list deleted successfully" << endl;
        return 0;
    }
};
