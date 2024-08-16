// header6
#ifndef HEADER6_H
#define HEADER6_H

#include <iostream>
#include <bits/stdc++.h>
#include "heap_node.h"

using namespace std;
// Other declarations and definitions

#endif // HEADER6_H
#pragma once

// Declarations and definitions

class MINheap {
private:
    Heap_Node* root;
    int size;

public:
    MINheap() {
        this->root = NULL;
        this->size = 0;
    }

    MINheap(int value) {
        this->size = 0;
        this->root = NULL;
        this->insertNode(value);
    }

    int insertNode(int value) {
        if(this->root == NULL) {
            this->root = new(nothrow) Heap_Node;
            this->root->data = value;
            this->root->leftChild = NULL;
            this->root->rightChild = NULL;
            this->root->parent = NULL;
            cout << "pushed in index : " << this->size << endl;
            this->size = this->size + 1;
            return 1;
        }
        int height = floor(log2(this->size)) + 1;
		int number_elements = pow(2, (height - 1)) ;
		int last_row_elements = this->size - pow(2, (height - 1)) + 1;

		Heap_Node* tempNode = root;

		if(number_elements == last_row_elements) {
			while(tempNode->leftChild != NULL) {
                if(tempNode->data > value) {
                    int temp = tempNode->data;
                    tempNode->data = value;
                    value = temp;
                }
				tempNode = tempNode->leftChild ;
			}
			Heap_Node* newNode = new(nothrow) Heap_Node;
			tempNode->leftChild = newNode;
			newNode->data = value;
			newNode->leftChild = NULL;
			newNode->rightChild = NULL;
			newNode->parent = tempNode;
			cout << "pushed in index : " << this->size << endl;
			this->size ++ ;
			return 0 ;
		}

		for(int i = 0 ; i < height ; i++) {
			if(last_row_elements >= (number_elements / 2)) {
				if(tempNode->rightChild == NULL) {
                    Heap_Node* newNode = new(nothrow) Heap_Node;
                    newNode->data = value;
                    newNode->leftChild = NULL;
                    newNode->rightChild = NULL;
                    newNode->parent = tempNode;

					tempNode->rightChild = newNode ;
					cout << "pushed in index : " << this->size << endl;
					this->size ++ ;
					return 0 ;
				}
                if(tempNode->data > value) {
                    int temp = tempNode->data;
                    tempNode->data = value;
                    value = temp;
                }

				tempNode = tempNode->rightChild ;
				last_row_elements = last_row_elements - (number_elements / 2) ;
			} else {
				if(tempNode->leftChild == NULL) {
                    Heap_Node* newNode = new(nothrow) Heap_Node;
                    newNode->data = value;
                    newNode->leftChild = NULL;
                    newNode->rightChild = NULL;
                    newNode->parent = tempNode;

					tempNode->leftChild = newNode ;
					cout << "pushed in index : " << this->size << endl;
					this->size ++ ;
					return 0 ;
				}
                if(tempNode->data > value) {
                    int temp = tempNode->data;
                    tempNode->data = value;
                    value = temp;
                }
				tempNode = tempNode->leftChild ;
			}
			number_elements = number_elements / 2 ;
		}

		return 0 ;
    }

    void deleteMINheap() {
        this->deleteMINheap(this->root);
    }

private:

    void deleteMINheap(Heap_Node* node) {
        if(node != NULL) {
            deleteMINheap(node->leftChild);
            deleteMINheap(node->rightChild);
            cout << "Deleted successfully : " << node->data << endl;
            delete node;
        }
    }

};
