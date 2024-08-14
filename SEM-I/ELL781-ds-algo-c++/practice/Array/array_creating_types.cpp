#include<bits/stdc++.h>
using namespace std;

// class Array {
//     int arraySize;
//     int arr={};
//     int index;
// public:
//     // Array(int arr[]) {
//     //     this->arr=arr;
//     //     this->index=0;
//     // }
//     // Array(int arraySize) {
//     //     this->arraySize=arraySize;
//     //     this->arr[this->arraySize];
//     // }
//     void insert_element(int value) {
//         if(this->index<(this->arraySize)){
//             this->arr[index]=value;
//             this->index=this->index+1;
//             cout<<"Element inserted successfully"<<endl;
//             this->index++;
//         } else{
//             cout<<"Array is out of bound"<<endl;
//         }
//     }
//     int find_index(int value) {
//         for(int i=0;i<this->index;i++) {
//             if(this->arr[i]=value) {
//                 cout<<"Element present in "<<i<<" index."<<endl;
//                 return i;
//             }
//         }
//         cout<<"Element is not present in the array"<<endl;
//         return -1;
//     }
//     void delete_element(int value) {
//         int element_index=this->find_index(value);
//         if(element_index=-1) {
//             cout<<"Element is not present in the array"<<endl;
//         } else {
//             for(element_index;element_index<(this->index-1);element_index++){
//                 this->arr[element_index]=this->arr[element_index+1];
//             }
//             cout<<"Element deleted successfully."<<endl;
//         }
//     }
// };

int main() {
    int arr1[] = {1,2,3,4,5};
    int arr2[5] = {1,2,3,4,5};
    int arr3[5];
    for(int i=1;i<=5;i++) {
        arr3[i]=i;
    }
    int arr4[]={};

    int *add=arr1;
    cout<<&arr1<<endl<<add<<endl<<&arr2<<endl<<arr3<<endl<<arr4<<endl;

    cout<<"[";
    for(int i=0;i<5;i++){
        cout<<*(add+i)<<",";
    }
    cout<<"]"<<endl;

    cout<<"[";
    for(int i=0;i<5;i++){
        cout<<arr1[i]<<",";
    }
    cout<<"]"<<endl;

    cout<<"[";
    for(int i=0;i<5;i++){
        cout<<*(arr1+i)<<",";
    }
    cout<<"]"<<endl;

    cout<<"[";
    for(int i=0;i<5;i++){
        cout<<*(add+i)<<",";
    }
    cout<<"]"<<endl;

    cout<<"[";
    for(int i=0;i<5;i++){
        cout<<*(&add+i)<<",";
    }
    cout<<"]"<<endl;

    return 0;
}
