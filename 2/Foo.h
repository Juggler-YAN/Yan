#ifndef FOO_H
#define FOO_H

#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

class Foo {
public:
    Foo sorted() &&;
    Foo sorted() const &;
private:
    vector<int> data;
};

Foo Foo::sorted() && {
    cout << "Foo::sorted() &&" << endl;
    sort(data.begin(), data.end());
    return *this;
}

Foo Foo::sorted() const & {
    cout << "Foo::sorted() const &" << endl;
    Foo ret(*this);
    return ret.sorted();
    // return Foo(*this).sorted();
}

#endif