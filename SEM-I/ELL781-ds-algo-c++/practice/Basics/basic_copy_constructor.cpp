#include<iostream>
using namespace std;

class ParentClass {
private:
    int id;
    string name;
public:
    ParentClass(int id,string name) {
        this->id=id;
        this->name=name;
    }
    void print() {
    cout<<this->id<<"\n"<<this->name;
    }
};

int main() {
    ParentClass p1(01,"Animesh Lohar");
    ParentClass p2=p1;
    p2.print();
    return 0;
}
