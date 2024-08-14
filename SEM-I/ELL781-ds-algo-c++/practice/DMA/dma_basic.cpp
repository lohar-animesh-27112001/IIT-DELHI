#include<bits/stdc++.h>
using namespace std;

int print(int arr[], int length) {
    for(int i=0 ; i<length ; i++) {
        cout<<"Index no "<<i<<": "<<arr[i]<<endl;
    }
    return 0;
}

int main() {
    int *p = new(nothrow) int;

    if(!p) {
        cout<<"Memory allocation failed"<<endl;
    } else {
        *p = 10;
        cout<<"P: "<<*p<<endl;
    }

    float *q = new(nothrow) float(70.23);

    int n;
    int *r = new(nothrow) int[n];

    cout<<"Enter how many number of element you want in the array"<<endl;
    cin>>n;

    if(!r) {
        cout<<"Memory allocation failed"<<endl;
    } else {
        for(int i=0 ; i<n ; i++) {
            int temp;
            cout<<"For index no: "<<i<<endl;
            cin>>temp;
            r[i]=temp;
        }
    }

    print(r,n);

    delete p;
    delete q;
    delete[] r;

    return 0;
}
