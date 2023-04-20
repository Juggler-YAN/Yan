#ifndef SHAREDPTR_H
#define SHAREDPTR_H

#include <functional>
#include "DebugDelete.h"

using namespace std;

template <typename T>
class SharedPtr;
template <typename T>
bool operator==(const SharedPtr<T> &lhs, const SharedPtr<T> &rhs);
template <typename T>
bool operator!=(const SharedPtr<T> &lhs, const SharedPtr<T> &rhs);

template <typename T>
class SharedPtr {
    friend bool operator==<T>(const SharedPtr<T> &lhs, const SharedPtr<T> &rhs);
    friend bool operator!=<T>(const SharedPtr<T> &lhs, const SharedPtr<T> &rhs);
public:
    SharedPtr() : ptr(nullptr), cnt(nullptr) {}
    SharedPtr(T *p, function<void(T*)> d = DebugDelete()) : 
        ptr(p), del(d), cnt(new size_t(1)) {}
    SharedPtr(const SharedPtr &p) : ptr(p.ptr), del(p.del), cnt(p.cnt) {
        ++*cnt;
    }
    SharedPtr& operator=(const SharedPtr &p);
    T operator*() const { return *ptr; }
    T* operator->() const { return ptr; }
    void reset(T *p);
    void reset(T *p, function<void(T*)> d);
    ~SharedPtr();
private:
    T *ptr;
    function<void(T*)> del;
    size_t *cnt;
};

template <typename T>
bool operator==(const SharedPtr<T> &lhs, const SharedPtr<T> &rhs) {
    return lhs.ptr == rhs.ptr;
}

template <typename T>
bool operator!=(const SharedPtr<T> &lhs, const SharedPtr<T> &rhs) {
    return !(lhs == rhs);
}

template <typename T>
SharedPtr<T>& SharedPtr<T>::operator=(const SharedPtr<T> &p) {
    ++*p.cnt;
    if (--*cnt == 0) {
        del ? del(p) : delete p;
        delete cnt;
    }
    ptr = p.ptr;
    del = p.del;
    cnt = p.cnt;
    return *this;
}

template <typename T>
void SharedPtr<T>::reset(T *p) {
    if (cnt && --*cnt == 0) {
        del ? del(p) : delete p;
        delete cnt;
    }
    ptr = p;
    cnt = new size_t(1);
}

template <typename T>
void SharedPtr<T>::reset(T *p, function<void(T*)> d) {
    reset(p);
    del = d;
}

template <typename T>
SharedPtr<T>::~SharedPtr() {
    if (--*cnt == 0) {
        del ? del(ptr) : delete ptr;
        delete cnt;
    }
}

template <typename T>
SharedPtr<T> make_shared() {
    SharedPtr<T> s(new T);
    return s;
}

#endif