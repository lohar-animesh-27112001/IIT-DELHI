#ifndef HEADER4_H
#define HEADER4_H

#include <bits/stdc++.h>

using namespace std;
#pragma once

// Node structure for linked list
struct Node {
    string data;
    Node* next;
    Node(string val) : data(val), next(nullptr) {}
};

class PuzzleSolve {
private:
    Node* river;         // Linked list for river states
    Node* not_success;   // Linked list for unsuccessful states
    int number_of_success;  // Count of valid solutions

public:
    // Constructor that initializes the linked lists
    PuzzleSolve(string arr1[], string arr2[]) {
        // Initialize river linked list with two nodes
        river = new Node(arr1[0]);     // Left side of the river (starting state)
        river->next = new Node(arr1[1]);  // Right side (initially empty)

        // Initialize not_success linked list with two nodes
        not_success = new Node(arr2[0]);
        not_success->next = new Node(arr2[1]);

        number_of_success = 0;  // Initialize success count

        // Display initial state
        show_river();
        show_not_success();
    }

    // Destructor to deallocate linked list memory
    ~PuzzleSolve() {
        delete_linked_list(river);       // Free linked list memory for river
        delete_linked_list(not_success); // Free linked list memory for not_success
    }

    // Function to delete linked list nodes
    void delete_linked_list(Node* head) {
        while (head != nullptr) {
            Node* temp = head;
            head = head->next;
            delete temp;
        }
    }

    // Function to start the process of counting successes
    void count_successes() {
        string str1 = river->data;            // Left side of the river
        string str2 = river->next->data;      // Right side of the river
        vector<pair<string, string>> path;    // To store transitions
        count_successes(str1, str2, path);    // Recursive call to solve the puzzle
        cout << "Total number of solutions: " << number_of_success << endl;
    }

    // Recursive function to explore the state space and count valid solutions
    void count_successes(string str1, string str2, vector<pair<string, string>> &path) {
        vector<string> combinations;
        bool check = false;

        // Call the function to generate combinations of the specified length
        generateCombinations(str2, "", 0, str2.size(), combinations);
        for (auto comb : combinations) {
            if (comb == str2) {
                check = true;
            }
        }

        if (str1.empty() && check == true) {
            number_of_success++;
            return;
        }

        // Prevent cycles: check if the current state has already been visited
        for (auto p : path) {
            if (p.first == str1 && p.second == str2) {
                return;  // If already visited, skip this state
            }
        }

        // Record current state to prevent cycles
        path.push_back({str1, str2});

        // Check if person (P) is on the left side
        if (str1.find('P') != string::npos) {
            // Move person from left to right, possibly with an item
            for (char item : str1) {
                string newLeft = str1, newRight = str2;

                // Move person and the item to the right
                newLeft.erase(newLeft.find('P'), 1);
                newRight += 'P';
                if (item != 'P') {
                    newLeft.erase(newLeft.find(item), 1);
                    newRight += item;
                }

                // Check if the new state is valid and recurse
                if (is_valid_state(newLeft) && is_valid_state(newRight)) {
                    count_successes(newLeft, newRight, path);
                }
            }
        } else {
            // Move person from right to left, possibly with an item
            for (char item : str2) {
                string newLeft = str1, newRight = str2;

                // Move person and the item to the left
                newRight.erase(newRight.find('P'), 1);
                newLeft += 'P';
                if (item != 'P') {
                    newRight.erase(newRight.find(item), 1);
                    newLeft += item;
                }

                // Check if the new state is valid and recurse
                if (is_valid_state(newLeft) && is_valid_state(newRight)) {
                    count_successes(newLeft, newRight, path);
                }
            }
        }
    }

    // Check if the current state is valid (no dangerous pairs left alone)
    bool is_valid_state(string state) {
        if (state.find('W') != string::npos && state.find('G') != string::npos && state.find('P') == string::npos) {
            return false;
        }
        if (state.find('G') != string::npos && state.find('C') != string::npos && state.find('P') == string::npos) {
            return false;
        }
        return true;
    }

private:
    // Utility function to display the river linked list (left and right sides)
    void show_river() {
        cout << "Current river state (Left -> Right): " << endl;
        cout << "Left: " << river->data << endl;
        cout << "Right: " << river->next->data << endl;
    }

    // Utility function to display not successful linked list
    void show_not_success() {
        cout << "Unsuccessful combinations: " << endl;
        Node* temp = not_success;
        while (temp != nullptr) {
            cout << temp->data << endl;
            temp = temp->next;
        }
    }

    // Function to generate combinations of a given length
    void generateCombinations(const string &s, string current, int index, int length, vector<string> &result) {
        if (current.length() == length) {
            result.push_back(current);
            return;
        }
        for (int i = index; i < s.length(); i++) {
            generateCombinations(s, current + s[i], i + 1, length, result);
        }
    }
};

#endif // HEADER4_H
