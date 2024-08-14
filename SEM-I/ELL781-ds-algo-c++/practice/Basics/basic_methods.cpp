#include<iostream>
using namespace std;

void print(int a){
    cout<<a<<endl;
}

int sum(int a,int b) {
    int sums=a+b;
    print(sums);
    return sums;
}

int main() {
    int a,b;
    cin>>a>>b;
    cout<<sum(a,b);
    return 0;
}
