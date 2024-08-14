#include <bits/stdc++.h>
using namespace std;

class Pointer {
    int value;
    int* value_pointer;
public:
    Pointer(int value) {
        this->value=value;
        this->value_pointer=&value;
    }
    void print() {
        cout<<"Value: "<<this->value<<"\nPointer: "<<this->value_pointer<<endl;
    }
};

class Reference {
    int value=1;
    int& reference_value=value;
public:
    Reference(int value) {
        this->value=value;
    }
    void print_reference_value() {
        cout<<"reference value: "<<reference_value<<endl;
    }
    void edit_reference_value(int value) {
        this->reference_value=value;
    }
    void print_value() {
        cout<<"Value: "<<value<<endl;
    }
};

int main() {
    cout<<"Hi, I am Animesh Lohar"<<endl;
    Pointer p(10);
    p.print();
    Reference r(10);
    r.print_reference_value();
    r.edit_reference_value(20);
    r.print_value();
}
