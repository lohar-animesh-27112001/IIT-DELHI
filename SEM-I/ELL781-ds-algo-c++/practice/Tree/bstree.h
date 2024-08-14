// header1.h
#ifndef HEADER3_H
#define HEADER3_H

#include <iostream>
#include <bits/stdc++.h>
#include "tree_node.h"

using namespace std;
// Other declarations and definitions

#endif // HEADER2_H
#pragma once

// Declarations and definitions

class Binary_Search_Tree {
public:
    Tree_Node* root = NULL;

    Binary_Search_Tree () {
        this->root = NULL;
    }

    Binary_Search_Tree (int value) {
        this->addTreeNode(value);
    }

    void addTreeNode (int value) {
        int i = 0;
        if(this->root == NULL) {
            this->root = new(nothrow) Tree_Node;
            this->root->data = value;
            this->root->leftNode = NULL;
            this->root->rightNode = NULL;
            cout << value << " is added in the BSTree" << endl;
        } else {
            Tree_Node* tempNode = this->root;
            Tree_Node* temp;
            while(tempNode != NULL) {
                if(tempNode->data > value) {
                    temp = tempNode;
                    tempNode = tempNode->leftNode;
                } else if(tempNode->data < value) {
                    temp = tempNode;
                    tempNode = tempNode->rightNode;
                } else {
                    i = 1;
                    cout << "The value is already present in the tree" << endl;
                    break;
                }
            }
            if(i != 1) {
                delete tempNode;
                if(temp->data > value) {
                    Tree_Node* tempNode = new (nothrow) Tree_Node;
                    temp->leftNode = tempNode;
                    tempNode->data = value;
                    tempNode->rightNode = NULL;
                    tempNode->leftNode = NULL;
                    cout << value << " is added in the BSTree" << endl;
                } else if(temp->data < value) {
                    Tree_Node* tempNode = new (nothrow) Tree_Node;
                    temp->rightNode = tempNode;
                    tempNode->data = value;
                    tempNode->rightNode = NULL;
                    tempNode->leftNode = NULL;
                    cout << value << " is added in the BSTree" << endl;
                }
            }
        }
    }

    void printBSTree() {
        cout << "In which order do you print the Binary search tree?" << endl
        << "For Preorder enter pre, inorder enter in, postorder enter post: "
        << endl;
        string str;
        cin >> str;
        if(str == "pre") {
            this->preorder(this->root);
        } else if(str == "in") {
            this->inorder(this->root);
        } else if(str == "post") {
            this->postorder(this->root);
        }
        cout << endl;
    }

    void deleteBSTree() {
        this->deleteBST(this->root);
    }

    void deleteNode(int value) {
        this->deleteNode(this->root, value);
        cout << "Deleted successfully : " << value << endl;
    }

private:
    void deleteBST(Tree_Node* node) {
        if(node != NULL) {
            deleteBST(node->leftNode);
            deleteBST(node->rightNode);
            cout << "Deleted successfully : " << node->data << endl;
            delete node;
        }
    }

    void preorder(Tree_Node* node) {
        if(node != NULL) {
            cout << node->data << " ";
            preorder(node->leftNode);
            preorder(node->rightNode);
        }
    }

    void inorder(Tree_Node* node) {
        if(node != NULL) {
            inorder(node->leftNode);
            cout << node->data << " ";
            inorder(node->rightNode);
        }
    }

    void postorder(Tree_Node* node) {
        if(node != NULL) {
            postorder(node->leftNode);
            postorder(node->rightNode);
            cout << node->data << " ";
        }
    }

    Tree_Node* deleteNode(Tree_Node* node, int value) {
        if(node == NULL) {
            return NULL;
        }
        if(node->data > value) {
            node->leftNode = deleteNode(node->leftNode, value);
        } else if(node->data < value) {
            node->rightNode = deleteNode(node->rightNode, value);
        } else {
            if(node->leftNode == NULL && node->rightNode == NULL) {
                delete node;
                return NULL;
            } else if(node->leftNode != NULL && node->rightNode == NULL) {
                Tree_Node* tempNode = node->leftNode;
                delete node;
                return tempNode;
            } else if(node->leftNode == NULL && node->rightNode != NULL) {
                Tree_Node* tempNode = node->rightNode;
                delete node;
                return tempNode;
            } else {
                Tree_Node* tempNode = node->rightNode;
                while (tempNode->leftNode != NULL) {
                    tempNode = tempNode->leftNode;
                }
                node->data = tempNode->data;
                node->rightNode = deleteNode(node->rightNode, tempNode->data);
            }
        }
        return node;
    }
};
