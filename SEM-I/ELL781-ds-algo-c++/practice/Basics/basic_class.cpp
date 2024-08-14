#include <iostream>
using namespace std;

class Person {
protected:
    string name;
    string gender;
public:
    void set_name(string name) {
        this->name = name;
    }
    void set_gender(string gender) {
        this->gender = gender;
    }
};

class Male: public Person {
protected:
    string profession;
public:
    void set_profession(string profession) {
    this->profession = profession;
    }
};

int main() {
    Male person1;
    person1.set_profession("S/W engineer");
}
