#ifndef HASPTR_H
#define HASPTR_H

#include <string>

using namespace std;

class HasPtr {
friend void swap(HasPtr&, HasPtr&);
friend bool operator<(const HasPtr&, const HasPtr&);
public:
    HasPtr(const string &s = string()) : ps(new string(s)), i(0), use(new size_t(1)) {}
    HasPtr(const HasPtr &hp) : ps(hp.ps), i(hp.i), use(hp.use) { ++*use; }
    HasPtr(HasPtr &&hp) noexcept : ps(hp.ps), i(hp.i), use(hp.use) { hp.ps = nullptr; hp.use = nullptr; }
    HasPtr& operator=(HasPtr&hp) { swap(*this, hp); return *this; };
    HasPtr& operator=(HasPtr&&) noexcept;
    ~HasPtr();
private:
    string *ps;
    int i;
    size_t *use;
};

HasPtr& HasPtr::operator=(HasPtr&& hp) noexcept {
    if (this != &hp) {
        delete ps;
        delete use;
        ps = std::move(hp.ps);
        i = std::move(hp.i);
        use = std::move(hp.use);
    }
    return *this;
}

bool operator<(const HasPtr &lhs, const HasPtr &rhs) {
    return *lhs.ps < *rhs.ps;
}

HasPtr::~HasPtr() {
    if (--*use == 0) {
        delete ps;
        delete use;
    }
}

void swap(HasPtr &lhs, HasPtr &rhs) {
    using std::swap;
    swap(lhs.ps, rhs.ps);
    swap(lhs.i, rhs.i);
    swap(lhs.use, rhs.use);
}

#endif