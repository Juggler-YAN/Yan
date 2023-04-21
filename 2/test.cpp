#include <iostream>

using namespace std;

class Class {
};

class Base : public Class {
public:
    Base() { cout << "Base()" << endl; }
    Base(int i) : val(i) { cout << "Base(int)" << endl; }
    Base(const Base &b) : val(b.val) { cout << "Base(const Base&)" << endl; }
private:
    int val;
};

class D1 : virtual public Base {
public:
    D1() = default;
    D1(int i) : Base(i) { cout << "D1(int)" << endl; }
    D1(const D1 &d) : Base(d) { cout << "D1(const D1&)" << endl; }
};

class D2 : virtual public Base {
public:
    D2() = default;
    D2(int i) : Base(i) { cout << "D2(int)" << endl; }
    D2(const D2 &d) : Base(d) { cout << "D2(const D2&)" << endl; }
};

class MI : public D1, public D2 {
public:
    MI() = default;
    MI(int i) : D1(i), D2(i) { cout << "MI(int)" << endl; }
    MI(const MI &m) : D1(m), D2(m) { cout << "MI(const MI&)" << endl; }
};

// class Final : public MI, public Class {
// };

int main() {
    return 0;
}