#ifndef SCREEN_H
#define SCREEN_H

#include <iostream>
#include <algorithm>
#include <string>

using namespace std;

using pos = string::size_type;

template <pos, pos> class Screen;
template <pos H, pos W>
istream& operator>>(istream&, Screen<H, W>&);
template <pos H, pos W>
ostream& operator<<(ostream&, const Screen<H, W>&);

template <pos H, pos W>
class Screen {
    friend istream& operator>> <H, W>(istream&, Screen<H, W>&);
    friend ostream& operator<< <H, W>(ostream&, const Screen<H, W>&);
public:
    Screen() = default;
    Screen(char c) : contents(H*W, c) {}
    char get() const { return contents[cursor]; }
    char get(pos r, pos c) const { return contents[r*W+c]; };
    Screen &move(pos r, pos c) ;
private:
    pos cursor = 0;
    string contents;
};

template <pos H, pos W>
istream& operator>>(istream &is, Screen<H, W> &s) {
    char c;
    is >> c;
    s.contents = string(H*W, c);
    return is;
}

template <pos H, pos W>
ostream& operator<<(ostream &os, const Screen<H, W> &s) {
    os << s.contents;
    return os;
}

template <pos H, pos W>
inline Screen<H, W> &Screen<H, W>::move(pos r, pos c) {
	cursor = r * W + c;
	return *this;
}

#endif