#include <iostream>

using namespace std;

int main() {
    for (int k = 1; k <= 10000; ++k) {
        cout << k << " " << 100.0*k/(k+99) << endl;
    }
    return 0;
}