#include<iostream>>
using namespace std;

class Person {
// private:
    // ~Person() {
        //   cout<<"Private destructor"<<"\n";
    //};
public:
    string name;
    int age;
    ~Person() {
        cout<<"Public destructor called"<<"\n";
    }
};

int main() {
    Person* personPtr = new Person; // Create a Person object on the heap and store its address in personPtr
    personPtr->name = "Alice"; // Access the name member of the Person object through the pointer
    personPtr->age = 30; // Access the age member of the Person object through the pointer

    delete personPtr; // Deallocate the memory used by the Person object
    return 0;
}
