#include <iostream>
#include <bits/stdc++.h>
#include "tree_node.h"
#include "bstree.h"

using namespace std;

int main()
{
    Binary_Search_Tree* bst = new (nothrow) Binary_Search_Tree(20);
    bst->addTreeNode(30);
    bst->addTreeNode(10);
    bst->addTreeNode(40);
    bst->addTreeNode(30);
    bst->addTreeNode(12);
    bst->deleteNode(20);
    bst->printBSTree();
    bst->deleteBSTree();
    delete bst;
    cout << "Hello world!" << endl;
    return 0;
}
