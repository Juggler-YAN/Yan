#include <iostream>
#include <vector>

using namespace std;

int main() {

    int ia[3][4] = {
        {0,1,2,3},
        {4,5,6,7},
        {8,9,10,11}
    };

    for (const auto &row : ia) {
        for (int i: row) {
            cout << i << " ";
        }
        cout << endl;
    }
    
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 4; j++) {
            cout << ia[i][j] << " ";
        }
        cout << endl;
    }

    for (auto p = begin(ia); p != end(ia); p++) {
        for (auto q = begin(*p); q != end(*p); q++) {
            cout << *q << " ";
        }
        cout << endl;
    }
    return 0;
}