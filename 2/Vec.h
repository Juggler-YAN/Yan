#ifndef VEC_H
#define VEC_H

#include <utility>
#include <memory>
#include <initializer_list>

using namespace std;

template <typename T>
class Vec {
public:
    Vec() : elements(nullptr), first_free(nullptr), cap(nullptr) {};
    Vec(const Vec&);
    Vec(initializer_list<T>);
    Vec& operator=(const Vec&);
    ~Vec();
    void push_back(const T&);
    template <typename... Args> inline void emplace_back(Args&&... args) {
        chk_n_alloc();
        alloc.construct(first_free++, std::forward<Args>(args)...);
    }
    size_t size() const { return first_free - elements; }
    size_t capacity() const { return cap - elements; }
    T *begin() const { return elements; }
    T *end() const { return first_free; }
    void reserve(size_t);
    void resize(size_t);
    void resize(size_t, const T&);
private:
    allocator<T> alloc;
    void chk_n_alloc() { if (size() == capacity()) reallocate(); }
    pair<T*, T*> alloc_n_copy(const T*, const T*);
    void free();
    void reallocate();
    T *elements;
    T *first_free;
    T *cap;
};

template <typename T>
Vec<T>::Vec(const Vec<T> &s) {
    auto newdata = alloc_n_copy(s.begin(), s.end());
    elements = newdata.first;
    first_free = cap = newdata.second;
}

template <typename T>
Vec<T>::Vec(initializer_list<T> l) {
    auto newdata = alloc_n_copy(l.begin(), l.end());
    elements = newdata.first;
    first_free = cap = newdata.second;
}

template <typename T>
Vec<T>& Vec<T>::operator=(const Vec<T> &s) {
    auto newdata = alloc_n_copy(s.begin(), s.end());
    free();
    elements = newdata.first;
    first_free = cap = newdata.second;
    return *this;
}

template <typename T>
Vec<T>::~Vec() {
    free();
}

template <typename T>
void Vec<T>::push_back(const T& s) {
    chk_n_alloc();
    alloc.construct(first_free++, s);
}

template <typename T>
void Vec<T>::reserve(size_t n) {
    if (n <= capacity()) return;
    auto newdata = alloc.allocate(n);
    auto dest = newdata;
    auto elem = elements;
    for (size_t i = 0; i != size(); ++i) {
        alloc.construct(dest++, std::move(*elem++));
    }
    free();
    elements = newdata;
    first_free = dest;
    cap = elements + n;
}

template <typename T>
void Vec<T>::resize(size_t n) {
    resize(n, T());
}

template <typename T>
void Vec<T>::resize(size_t n, const T& s) {
    if (n < size()) {
        while (n < size()) {
            alloc.destroy(--first_free);
        }
    }
    if (n > size()) {
        while (n > size()) {
            push_back(s);
        }
    }
}

template <typename T>
pair<T*, T*> Vec<T>::alloc_n_copy(const T *b, const T *e) {
    auto data = alloc.allocate(e-b);
    return {data, uninitialized_copy(b, e, data)};
}

template <typename T>
void Vec<T>::free() {
    if (elements) {
        for (auto p = first_free; p != elements; ) {
            alloc.destroy(--p);
        }
        alloc.deallocate(elements, cap-elements);
    }
}

template <typename T>
void Vec<T>::reallocate() {
    auto newcapacity = size() ? 2 * size() : 1;
    auto newdata = alloc.allocate(newcapacity);
    auto dest = newdata;
    auto elem = elements;
    for (size_t i = 0; i != size(); ++i) {
        alloc.construct(dest++, std::move(*elem++));
    }
    free();
    elements = newdata;
    first_free = dest;
    cap = elements + newcapacity;
}

#endif