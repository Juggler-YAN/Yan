#ifndef DEBUGDELETE_H
#define DEBUGDELETE_H

#include <iostream>
#include <string>

using namespace std;

class DebugDelete {
public:
    DebugDelete(const string &s = "Smarter Pointer", ostream &serr = cerr) : type(s), os(serr)  {}
    template <typename T>
    void operator()(T *p) const {
        os << "deleting " << type << endl;
        delete p;
    }
private:
    ostream &os;
    string type;
};

#endif