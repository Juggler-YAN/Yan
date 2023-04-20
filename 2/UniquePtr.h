#ifndef UNIQUEPTR_H
#define UNIQUEPTR_H

#include "DebugDelete.h"

using namespace std;

template <typename T, typename D = DebugDelete>
class UniquePtr {
public:
    UniquePtr(T *p = nullptr, D d = DebugDelete()) : ptr(p), del(d) {}
    UniquePtr(UniquePtr &&p) : ptr(p.ptr), del(p.del) { p.ptr=nullptr; }
    UniquePtr& operator=(UniquePtr &&p);
    T operator*() const { return *ptr; }
    T* operator->() const { return ptr; }
    void reset(T *p) {
        del(ptr);
        ptr=p;
    }
    void reset(T *p, D d) {
        reset(p);
        del=d;
    }
    ~UniquePtr() { del(ptr); }
private:
    T *ptr;
    D del;
};

template <typename T, typename D>
UniquePtr<T,D> &UniquePtr<T,D>::operator=(UniquePtr<T,D> &&p) {
    if (this != &p) {
        del(ptr);
        ptr = p.ptr;
        // del = p.del;
        p.ptr = nullptr;
    }
    return *this;
}

#endif