#include <iostream>
using namespace std;

class Base {
private:
    long long int phone_number;
protected:
    string name;
public:
    Base() {
        phone_number = 7585902807;
        name = "Animesh Lohar";
    }
    friend void friendFunction(Base& obj);
    friend class FriendClass;
};

void friendFunction(Base& obj) {
    cout<<"Phone number: "<<obj.phone_number<<"\n"<<"Name: "<<obj.name<<"\n";
}

class FriendClass {
public:
    void print(Base& obj) {
        cout<<"Phone number: "<<obj.phone_number<<"\n"<<"Name: "<<obj.name<<"\n";
    }
};

int main() {
    Base b;
    friendFunction(b);
    FriendClass fclass;
    fclass.print(b);
    return 0;
}
