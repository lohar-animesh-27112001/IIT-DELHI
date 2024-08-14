#include <iostream>
using namespace std ;

int main() {
    int num;
    cin>>num;
    cout<<num<<"\n"<<"hello"<<endl;
    cout<<num<<endl<<"Hello"<<endl;
    int a, b, c, sum;
    cin>>a>>b>>c;
    cout<<(a+b+c)<<endl;
    sum = a+b+c;
    cout<<sum<<endl;

    // If statement
    if(a>b) {
        cout<<true<<" True"<<endl;
    } else {
        cout<<false<<" False"<<endl;
    }

    // For statement
    sum=0;
    for(int i=0;i<=num;i++){
        sum+=i;
    }
    cout<<sum<<endl;

    // While & DoWhile loop
    while(sum>0) {
        cout<<sum;
        cin>>sum;
    }
    do {
        cout<<sum;
        cin>>sum;
    } while (sum>0);

    // Switch case
    char day;
    cin>>day;
    switch(day) {
        case 'M':
            cout<<"Monday";
            break;
        case 'T':
            cout<<"Tuesday"; // Bug
        default:
            cout<<"Try again";
    }
    return 0;
}
