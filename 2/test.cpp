#include <iostream>

using namespace std;


int main() {
    int someValue = 0;
    int x = 0, y = 0;
    cout << x << " " << y << endl;
    someValue ? (++x, ++y) : --x, --y;
    cout << x << " " << y << endl;
    return 0;
}