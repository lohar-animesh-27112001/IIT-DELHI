#include<iostream>
using namespace std;

class Student {
private:
    int id;
    string name;
    string dept;
public:
    Student();
    Student(int id, string name, string dept) {
        this->id=id;
        this->name=name;
        this->dept=dept;
    }
    Student(Student& s) {
        this->id=s.id;
        this->name=s.name;
        this->dept=s.dept;
    }
    void display() {
        cout<<"Student id: "<<this->id<<"\nStudent name: "
            <<this->name<<"\nStudent dept: "<<this->dept<<"\n";
        return;
    }
};

class Department {
private:
    int id;
    string deptName;
public:
    Department(int id, string deptName) {
        this->id=id;
        this->deptName=deptName;
    }
    void display() {
        cout<<"Dept id: "<<this->id<<"\nDept name: "
            <<this->deptName<<"\n";
        return;
    }
};

int main() {
    // user defined copy constructor
    Student s1();
    Student s2(001,"Animesh Lohar","Computer Science");
    Student s3(s2);
    s3.display();
    // by default copy constructor
    Department d1(01,"CS");
    Department d2(d1);
    d2.display();
}
