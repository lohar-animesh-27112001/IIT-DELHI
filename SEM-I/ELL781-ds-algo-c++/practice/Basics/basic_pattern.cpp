#include<iostream>
using namespace std;

int main() {
    int col, row;
    cin>>col>>row;
    for(int i=0;i<row;i++) {
        for(int j=0;j<col;j++){
            if(j==0 || j==(col-1) || i==0 || i==(row-1)){
                cout<<"* ";
            } else {
                cout<<"  ";
            }
        }
        cout<<"\n";
    }
    return 0;
}
