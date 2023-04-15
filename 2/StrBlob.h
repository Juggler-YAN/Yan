#ifndef STRBLOB_H
#define STRBLOB_H

#include <memory>
#include <initializer_list>
#include <vector>
#include <string>
#include <stdexcept>

using namespace std;

class ConstStrBlobPtr;

class StrBlob {
    friend class ConstStrBlobPtr;
    friend bool operator==(const StrBlob&, const StrBlob&);
    friend bool operator!=(const StrBlob&, const StrBlob&);
    friend bool operator<(const StrBlob&, const StrBlob&);
    friend bool operator>(const StrBlob&, const StrBlob&);
    friend bool operator<=(const StrBlob&, const StrBlob&);
    friend bool operator>=(const StrBlob&, const StrBlob&);
public:
    typedef vector<string>::size_type size_type;
    StrBlob();
    StrBlob(initializer_list<string> i1);
    StrBlob(const StrBlob&);
    StrBlob& operator=(const StrBlob&);
    string& operator[](size_t n) { return (*data)[n]; }
    const string& operator[](size_t n) const { return (*data)[n]; }
    size_type size() const { return data->size(); }
    bool empty() const { return data->empty(); }
    void push_back(const string &t) { data->push_back(t); }
	void push_back(string &&t) { data->push_back(std::move(t)); }
    void pop_back();
    string& front();
    string& back();
    const string& front() const;
    const string& back() const;
	ConstStrBlobPtr begin();
	ConstStrBlobPtr end();
private:
    shared_ptr<vector<string>> data;
    void check(size_type i, const string &msg) const;
};

class ConstStrBlobPtr {
public:
    friend bool operator==(const ConstStrBlobPtr&, const ConstStrBlobPtr&);
    friend bool operator!=(const ConstStrBlobPtr&, const ConstStrBlobPtr&);
    ConstStrBlobPtr() : curr(0) {}
    ConstStrBlobPtr(const StrBlob &a, size_t sz = 0) : wptr(a.data), curr(sz) {}
    string& deref() const;
    ConstStrBlobPtr& incr();
    ConstStrBlobPtr& operator++();
    ConstStrBlobPtr& operator--();
    ConstStrBlobPtr operator++(int);
    ConstStrBlobPtr operator--(int);
    const string& operator*() const {
        auto p = check(curr, "dereference past end");
        return (*p)[curr];
    }
    const string* operator->() const {
        return & this->operator*();
    }
private:
    shared_ptr<vector<string>> check(size_t, const string&) const;
    weak_ptr<vector<string>> wptr;
    size_t curr;
};

class ConstStrBlobPtrPtr {
public:
    const string* operator->() const {
        return p->operator->();
    }
private:
    ConstStrBlobPtr * p;
};

ConstStrBlobPtr StrBlob::begin() { return ConstStrBlobPtr(*this); }
ConstStrBlobPtr StrBlob::end() {
    auto ret = ConstStrBlobPtr(*this, data->size());
    return ret;
}

StrBlob::StrBlob(): data(make_shared<vector<string>>()) {};
StrBlob::StrBlob(initializer_list<string> i1): data(make_shared<vector<string>>(i1)) {};
StrBlob::StrBlob(const StrBlob& sb) { data = make_shared<vector<string>>(*sb.data); }
StrBlob& StrBlob::operator=(const StrBlob& sb) { data = make_shared<vector<string>>(*sb.data); return *this; }

void StrBlob::check(size_type i, const string &msg) const {
    if (i >= data->size())
        throw out_of_range(msg);
}

string& StrBlob::front() {
    check(0, "front on empty StrBlob");
    return data->front();
}

string& StrBlob::back() {
    check(0, "back on empty StrBlob");
    return data->back();
}

const string& StrBlob::front() const {
    check(0, "front on empty StrBlob");
    return data->front();
}

const string& StrBlob::back() const {
    check(0, "back on empty StrBlob");
    return data->back();
}

void StrBlob::pop_back() {
    check(0, "pop_back on empty StrBlob");
    return data->pop_back();
}

shared_ptr<vector<string>> ConstStrBlobPtr::check(size_t i, const string& msg) const {
    auto ret = wptr.lock();
    if (!ret)
        throw runtime_error("unbound StrBlobPtr");
    if (i >= ret->size())
        throw out_of_range(msg);
    return ret;
}

string& ConstStrBlobPtr::deref() const {
    auto p = check(curr, "dereference past end");
    return (*p)[curr];
}

ConstStrBlobPtr& ConstStrBlobPtr::incr() {
    check(curr, "increment past end of StrBlobPtr");
    ++curr;
    return *this;
}

ConstStrBlobPtr& ConstStrBlobPtr::operator++() {
    check(curr, "increment past end of ConstStrBlobPtr");
    ++curr;
    return *this;
}

ConstStrBlobPtr& ConstStrBlobPtr::operator--() {
    --curr;
    check(curr, "decrement past begin of ConstStrBlobPtr");
    return *this;
}

ConstStrBlobPtr ConstStrBlobPtr::operator++(int) {
    ConstStrBlobPtr ret = *this;
    ++*this;
    return ret;
}

ConstStrBlobPtr ConstStrBlobPtr::operator--(int) {
    ConstStrBlobPtr ret = *this;
    --*this;
    return ret;
}

bool operator==(const StrBlob &lhs, const StrBlob &rhs) {
    return *lhs.data == *rhs.data;

}
bool operator!=(const StrBlob &lhs, const StrBlob &rhs) {
    return !(lhs == rhs);
}

bool operator==(const ConstStrBlobPtr &lhs, const ConstStrBlobPtr &rhs) {
    return lhs.wptr.lock() == rhs.wptr.lock() && lhs.curr == rhs.curr;
}

bool operator!=(const ConstStrBlobPtr &lhs, const ConstStrBlobPtr &rhs) {
    return !(lhs == rhs);
}

bool operator<(const StrBlob &lhs, const StrBlob &rhs) {
    return lexicographical_compare(lhs.data->begin(), lhs.data->end(), rhs.data->begin(), rhs.data->end());
}

bool operator>(const StrBlob &lhs, const StrBlob &rhs) {
    return rhs < lhs;
}

bool operator<=(const StrBlob &lhs, const StrBlob &rhs) {
    return !(rhs < lhs);
}

bool operator>=(const StrBlob &lhs, const StrBlob &rhs) {
    return !(lhs < rhs);
}

#endif