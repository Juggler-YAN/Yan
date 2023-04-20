#include <iostream>
#include <sstream>
#include <memory>
#include <string>

using namespace std;

template <typename T>
string debug_rep(const T&);
template <typename T>
string debug_rep(T*);
string debug_rep(const string&);

template <typename T>
string debug_rep(const T &t) {
    ostringstream ret;
    ret << t;
    return ret.str();
}

template <typename T>
string debug_rep(T *p) {
    ostringstream ret;
    if (p) {
        ret << " " << debug_rep(*p);
    }
    else {
        ret << " null pointer";
    }
    return ret.str();
}
string debug_rep(const string &s) {
    return '"' + s + '"';
}
template <>
string debug_rep(char *p) {
    cout << "debug_rep(char *p)" << endl;
    return debug_rep(string(p));
}
template <>
string debug_rep(const char *p) {
    cout << "debug_rep(const char *p)" << endl;
    return debug_rep(string(p));
}

int main() {
    char p[] = "abc";
    cout << debug_rep(p) << endl;
    const char cp[] = "abc";
    cout << debug_rep(cp) << endl;
    return 0;
}