// String
#ifndef STRING_H
#define STRING_H

#include <iostream>
#include <algorithm>
#include <memory>
#include <cstring>

using namespace std;

class String {
    friend ostream& operator<<(ostream&, const String&);
    friend bool operator==(const String&, const String&);
    friend bool operator!=(const String&, const String&);
    friend bool operator<(const String&, const String&);
    friend bool operator>(const String&, const String&);
    friend bool operator<=(const String&, const String&);
    friend bool operator>=(const String&, const String&);
public:
    String(): elements(nullptr), first_free(nullptr) {}
    String(const char *);
    String(const String&);
    String(String&&) noexcept;
    String& operator=(const String&);
    String& operator=(String&&) noexcept;
    char& operator[](size_t n) { return elements[n]; };
    const char& operator[](size_t n) const { return elements[n]; };
    ~String();
    char * begin() const { return elements; }
    char * end() const { return first_free; }
private:
    allocator<char> alloc;
    pair<char*, char*> alloc_n_copy(const char*, const char*);
    void free();
    char * elements;
    char * first_free;
};

String::String(const char * s) {
    size_t n = strlen(s);
    auto newdata = alloc_n_copy(s, s+n);
    elements = newdata.first;
    first_free = newdata.second;
}

String::String(const String &s) {
    auto newdata = alloc_n_copy(s.begin(), s.end());
    elements = newdata.first;
    first_free = newdata.second;
}

String::String(String &&s) noexcept {
    alloc = std::move(s.alloc);
    elements = std::move(s.elements);
    first_free = std::move(s.first_free);
    s.elements = s.first_free = nullptr;
    cout << "String(String &&s) noexcept" << endl;
}

String& String::operator=(const String &s) {
    auto newdata = alloc_n_copy(s.begin(), s.end());
    free();
    elements = newdata.first;
    first_free = newdata.second;
    return *this;
}

String& String::operator=(String &&s) noexcept {
    if (this != &s) {
        free();
        alloc = std::move(s.alloc);
        elements = std::move(s.elements);
        first_free = std::move(s.first_free);
        s.elements = s.first_free = nullptr;
    }
    cout << "String& operator=(String &&s) noexcept" << endl;
    return *this;
}

String::~String() {
    free();
}

ostream& operator<<(ostream &os, const String &s) {
    for (auto i = s.elements; i != s.first_free; ++i) {
        os << *i;
    }
    return os;
}

void String::free() {
    if (elements) {
        for (auto p = first_free; p != elements; ) {
            alloc.destroy(--p);
        }
        alloc.deallocate(elements,first_free-elements);
    }
}

pair<char*, char*> String::alloc_n_copy
        (const char *b, const char *e) {
    auto data = alloc.allocate(e-b);
    return {data, uninitialized_copy(b, e, data)};
}

bool operator==(const String &lhs, const String &rhs) {
    return (lhs.first_free-lhs.elements) == (rhs.first_free-rhs.elements) &&
           equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool operator!=(const String &lhs, const String &rhs) {
    return !(lhs == rhs);
}

bool operator<(const String &lhs, const String &rhs) {
    return lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), lhs.end());
}

bool operator>(const String &lhs, const String &rhs) {
    return rhs < lhs;
}

bool operator<=(const String &lhs, const String &rhs) {
    return !(rhs < lhs);
}

bool operator>=(const String &lhs, const String &rhs) {
    return !(lhs < rhs);
}

#endif