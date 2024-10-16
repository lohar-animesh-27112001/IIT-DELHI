// header1.h
#ifndef HEADER4_H
#define HEADER4_H

#include <bits/stdc++.h>

using namespace std;

// State class to represent the river crossing state

// Solving class
class PuzzleSolve {
private:
    string river[2];        // Current state of the river (left and right sides)
    string not_success[2];  // States that are not successful
    int number_of_success;  // Count of valid solutions

public:
    // Constructor that takes arrays of strings
    PuzzleSolve(string arr1[], string arr2[]) {
        // Initialize the river state
        river[0] = arr1[0];   // Left side of the river (starting state)
        river[1] = arr1[1];   // Empty right side

        // Store the unsuccessful states
        not_success[0] = arr2[0];
        not_success[1] = arr2[1];

        number_of_success = 0;  // Initialize success count

        // Display initial state
        show_river();
        show_not_success();
    }

    // Function to start the process of counting successes
    void count_successes() {
        string str1 = river[0];  // Left side of the river
        string str2 = river[1];  // Right side of the river
        vector<pair<string, string>> path;  // To store transitions
        count_successes(str1, str2, path);  // Recursive call to solve the puzzle
        cout << "Total number of solutions: " << number_of_success << endl;
    }

private:

    // Recursive function to explore the state space and count valid solutions
    void count_successes(string str1, string str2, vector<pair<string, string>> &path) {
        // Base condition: check if we have reached the goal state (everyone on the right)
        // cout << "HII" << endl;

        // Vector to store combinations
        vector<string> combinations;
        bool check = false;

        // Call the function to generate combinations of the specified length
        generateCombinations(str2, "", 0, str2.size(), combinations);
        for (auto comb : combinations) {
            if(comb == str2) {
                check = true;
            }
        }

        if (str1.empty() && check == true) {
            number_of_success = number_of_success + 1;
            // cout << "Solution " << number_of_success << " found!" << endl;
            // for (auto p : path) {
                // cout << "Left: " << p.first << ", Right: " << p.second << endl;
            // }
            // cout << "-------------------" << endl;
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
                // if (item == 'P') continue;  // Skip the person itself in the loop
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
                // if (item == 'P') continue;  // Skip the person itself in the loop
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
        // for (auto p : path) {
        //     cout << "Left: " << p.first << ", Right: " << p.second << endl;
        // }
        return;
    }

    // Check if the current state is valid (no dangerous pairs left alone)
    bool is_valid_state(string state) {
        // If both wolf and goat are present without the person, it's invalid
        if (state.find('W') != string::npos && state.find('G') != string::npos && state.find('P') == string::npos) {
            return false;
        }
        // If both goat and cabbage are present without the person, it's invalid
        if (state.find('G') != string::npos && state.find('C') != string::npos && state.find('P') == string::npos) {
            return false;
        }
        return true;
    }

    // Utility function to display the river array (left and right sides)
    void show_river() {
        cout << "Current river state (Left -> Right): " << endl;
        cout << "Left: " << river[0] << endl;
        cout << "Right: " << river[1] << endl;
    }

    // Utility function to display not successful states
    void show_not_success() {
        cout << "Unsuccessful combinations: " << endl;
        for (int i = 0; i < 2; i++) {
            cout << not_success[i] << endl;
        }
    }

    void generateCombinations(const string &s, string current, int index, int length, vector<string> &result) {
        // If the current combination has the required length, store it in the result vector
        if (current.length() == length) {
            result.push_back(current);
            return;
        }

        // Recursion to build combinations
        for (int i = index; i < s.length(); i++) {
            generateCombinations(s, current + s[i], i + 1, length, result);
        }
    }

};

#endif // HEADER2_H
#pragma once
