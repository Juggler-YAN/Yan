#include <iostream>
#include <vector>

using namespace std;

int add(int, int);
int subtract(int, int);
int multiply(int, int);
int divide(int, int);

int main() {
    int a = 1, b = 2;
    vector<int (*)(int, int)> vf{add, subtract, multiply, divide};
    for (auto i : vf) {
        cout << (*i)(a,b) << endl;
    }
    return 0;
}

int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}

int multiply(int a, int b) {
    return a * b;
}

int divide(int a, int b) {
    return b != 0 ? a / b : 0;
}