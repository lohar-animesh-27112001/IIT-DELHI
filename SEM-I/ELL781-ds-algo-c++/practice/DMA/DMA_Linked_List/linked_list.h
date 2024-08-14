// header2.h
#ifndef HEADER2_H
#define HEADER2_H

#include <iostream>
#include <bits/stdc++.h>
#include "node.h"

using namespace std;
// Other declarations and definitions

#endif // HEADER2_H

#pragma once

class Linked_List {
public:
    Node* head;

    Linked_List() {
        this->head = NULL;
        cout << "Linked list created successfully" << endl;
    }

    Linked_List(int value) {
        this->head = new(nothrow) Node;
        this->head->data = value;
        this->head->next = NULL;
        cout << "Linked list created successfully" << endl;
    }

    void printLinkedList() {
        Node* temp = this->head;
        if(temp == NULL) {
            cout << "Here is no node" << endl;
        } else {
            while(temp != NULL) {
                if(temp->next != NULL) {
                    cout << temp->data << " -> ";
                } else {
                    cout << temp->data;
                }
                temp = temp->next;
            }
            cout << endl;
        }
    }

    void insertAtEnd(int value) {
        Node* tempNode = new(nothrow) Node;
        Node* temp = this->head;
        tempNode->data = value;
        tempNode->next = NULL;
        while(temp->next != NULL) {
            temp = temp->next;
        }
        temp->next = tempNode;
    }

    void insertAtBegining(int value) {
        Node* tempNode = new(nothrow) Node;
        tempNode->data = value;
        tempNode->next = NULL;
        Node* temp = this->head;
        head = tempNode ;
        head->next = temp;
    }

    int deleteByValue(int value) {
        if (this->head == NULL) {
            cout << "List is empty" << endl;
            return -1;
        }
        if(this->head->data == value) {
            Node* temp = this->head;
            this->head = temp->next;
            delete temp;
            return 0;
        }
        Node* temp = this->head;
        Node* tempNode = NULL;
        while(temp != NULL && temp->data != value) {
            tempNode = temp;
            temp = temp->next;
        }
        if(temp != NULL) {
            tempNode->next = temp->next;
            delete temp;
        } else {
            cout << "Element not found in the Linked List" << endl;
        }
        return 0;
    }

    void deleteLinkedList() {
        Node* temp = this->head;
        while(temp != NULL) {
            Node* next_tempnode = temp->next;
            delete temp;
            temp = next_tempnode;
        }
        this->head = NULL;
        cout << "Linked list deleted successfully" << endl;
    }
};
