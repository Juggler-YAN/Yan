#ifndef BLOB_H
#define BLOB_H

#include <memory>
#include <vector>
#include <string>
#include <initializer_list>
#include <stdexcept>
#include <exception>

using namespace std;

template <typename T>
class Blob {
public:
    typedef typename vector<T>::size_type size_type;
    Blob();
    Blob(initializer_list<T> i1);
    template <typename It>
    size_type size() const { return data->size(); }
    template <typename It>
    Blob(It b, It e) try : data(make_shared<vector<T>>(b,e)) {}
    catch (const bad_alloc &err) { 
        cout << err.what() << endl;
    }
    bool empty() const { return data->empty(); }
    void push_back(const T &t) { data->push_back(t); }
    void push_back(T &&t) { data->push_back(move(t)); }
    void pop_back();
    T& back();
    T& operator[](size_type i);
private:
    shared_ptr<vector<T>> data;
    void check(size_type i, const string &msg) const;
};

template <typename T>
class BlobPtr {
public:
    BlobPtr() try : curr(0) {}
    catch (const bad_alloc &e) { 
        cout << e.what() << endl;
    }
    BlobPtr(Blob<T> &a, size_t sz = 0) try : wptr(a.data), curr(sz) {}
    catch (const bad_alloc &e) { 
        cout << e.what() << endl;
    }
    T& operator*() const {
        auto p = check(curr, "dereference past end");
        return (*p)[curr];
    }
    BlobPtr& operator++();
    BlobPtr& operator--(); 
private:
    shared_ptr<vector<T>> check(size_t, const string&) const;
    weak_ptr<vector<T>> wptr;
    size_t curr;
};


template <typename T>
Blob<T>::Blob() try : data(make_shared<vector<T>>()) {}
catch (const bad_alloc &e) { 
    cout << e.what() << endl;
}
template <typename T>
Blob<T>::Blob(initializer_list<T> i1) try : data(make_shared<vector<T>>(i1)) {}
catch (const bad_alloc &e) {
    cout << e.what() << endl;
}
template <typename T>
T& Blob<T>::back() {
    check(0, "back on empty Blob");
    return data->back();
}
template <typename T>
T& Blob<T>::operator[](size_type i) {
    check(i, "subscript out of range");
    return (*data)[i];
}
template <typename T>
void Blob<T>::pop_back() {
    check(0, "pop_back on empty Blob");
    data->pop_back();
}
template <typename T>
void Blob<T>::check(size_type i, const string &msg) const {
    if (i >= data->size())
        throw out_of_range(msg);
}

template <typename T>
BlobPtr<T>& BlobPtr<T>::operator++() {
    BlobPtr ret = *this;
    ++*this;
    return ret;
}
template <typename T>
BlobPtr<T>& BlobPtr<T>::operator--() {
    BlobPtr ret = *this;
    --*this;
    return ret;
}
template <typename T>
shared_ptr<vector<T>> BlobPtr<T>::check(size_t i, const string &msg) const {
    if (i >= wptr.lock()->size())
        throw out_of_range(msg);
}

#endif