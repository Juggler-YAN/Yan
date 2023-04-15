#include <iostream>

using namespace std;

class SmallInt {
    friend SmallInt operator+(const SmallInt &s1, const SmallInt &s2) {
        SmallInt sum(s1.val+s2.val);
        return sum;
    };
public:
    SmallInt(int i = 0) : val(i) {}
    operator int() const { return val; }
private:
    size_t val;
};


int main() {
    SmallInt s1;
    double d = s1 + 3.14;
    // double d1 = s1 + SmallInt(3.14);
    // double d2 = s1.operator int() + 3.14;
    // double d3 = static_cast<int>(s1) + 3.14;
    return 0;
}