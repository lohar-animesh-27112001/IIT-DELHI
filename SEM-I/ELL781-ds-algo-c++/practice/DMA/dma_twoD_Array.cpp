#include<bits/stdc++.h>
using namespace std;

void printArr(int arr[], int row, int column) {
    for(int i = 0; i < row; i++) {
        for(int j = 0; j < column; j++) {
            cout << arr[i * column + j] << " ";
        }
        cout << endl;
    }
}

int main() {
    int m, n;
    cout << "Enter the number of rows you want: ";
    cin >> m;
    cout << "Enter the number of columns you want: ";
    cin >> n;
    int *arr = new(nothrow) int[m * n];

    if (!arr) {
        cout << "Memory allocation failed" << endl;
        return 1;
    }

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            int temp;
            cout << "Enter i[" << i << "][" << j << "]: ";
            cin >> temp;
            arr[i * n + j] = temp;
        }
    }

    char temp_char;
    cout << "Do you want to print the array? (Y/N) ";
    cin >> temp_char;

    if(temp_char == 'Y') {
        printArr(arr, m, n);
    }

    delete[] arr;
    return 0;
}
