#include<iostream>
using namespace std;

class B;

class A {
private:
    int id=1211;
public:
    friend void max(A& obj1, B& obj2);
};

class B {
private:
    int id=1323;
public:
    friend void max(A& obj1, B& obj2);
};

void max(A& obj1, B& obj2) {
    if(obj1.id > obj2.id) {
        cout<<"A class id is big, that is: "<<obj1.id<<"\n";
    } else {
        cout<<"B class id is big, that is: "<<obj2.id<<"\n";
    }
}

int main() {
    A a;
    B b;
    max(a,b);
    return 0;
}
