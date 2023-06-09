### Q1

编译器将实际的模板实参代替对应的模板参数来创建出模板的实例的过程称为实例化。

### Q2

```c++
#include <iostream>

using namespace std;

template <typename T>
int compare(const T &v1, const T &v2) {
    if (v1 < v2) return -1;
    if (v2 < v1) return 1;
    return 0;
}

int main() {
    cout << compare(1,2) << endl;
    cout << compare(2.0,1.0) << endl;
    return 0;
}
```

### Q3

```c++
// Sales_data.h
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <string>

using namespace std;

struct Sales_data {

    string bookNo;
    unsigned units_sold = 0;
    double revenue = 0.0;

    string isbn() const { return bookNo; }
    double avg_price() const;

};

double Sales_data::avg_price() const {
	if (units_sold) {
		return revenue / units_sold;
    }
	else {
		return 0;
    }
}

istream &read(istream &is, Sales_data &item) {
	double price = 0;
	is >> item.bookNo >> item.units_sold >> price;
	item.revenue = price * item.units_sold;
	return is;
}

ostream &print(ostream &os, const Sales_data &item) {
	os << item.isbn() << " " << item.units_sold << " " << item.revenue << " " << item.avg_price();
	return os;
}

#endif
```

```c++
#include <iostream>
#include "Sales_data.h"

template <typename T>
int compare(const T &v1, const T &v2) {
    if (v1 < v2) return -1;
    if (v2 < v1) return 1;
    return 0;
}

int main() {
    Sales_data a{"a", 1, 1.0};
    Sales_data b{"b", 2, 2.0};
    cout << compare(a,b) << endl;
    return 0;
}
```

### Q4

```c++
#include <iostream>
#include <vector>
#include <list>

using namespace std;

template <typename It, typename T>
It find(const It beg, const It end, const T &v) {
    It iter;
    for (iter = beg; iter != end; ++iter) {
        if (*iter == v) {
            return iter;
        }
    }
    return iter;
}

int main() {
    vector<int> v{1,2,3};
    list<int> l{1,2,3};
    if (v.end() != find(v.cbegin(), v.cend(), 3)) {
        cout << *(find(v.cbegin(), v.cend(), 3)) << endl;
    }
    if (l.end() != find(l.cbegin(), l.cend(), 2)) {
        cout << *(find(l.cbegin(), l.cend(), 2)) << endl;
    }
    return 0;
}
```

### Q5

```c++
#include <iostream>

using namespace std;

template <typename T, unsigned N>
void print(const T (&a)[N]) {
    for (unsigned i = 0; i != N; ++i) {
        cout << a[i] << " ";
    }
    cout << endl;
}

int main() {
    int a1[] = {1,2,3};
    char a2[] = "hello,world";
    print(a1);
    print(a2);
    return 0;
}
```

### Q6

```c++
#include <iostream>

using namespace std;

template <typename T, unsigned N>
T* mybegin(T (&a)[N]) {
    return a;
}

template <typename T, unsigned N>
T* myend(T (&a)[N]) {
    return a+N;
}

int main() {
    int a1[] = {1,2,3};
    cout << *(mybegin(a1)) << " " << *(myend(a1)-1) << endl;
    char a2[] = "hello,world";
    cout << *(mybegin(a2)) << " " << *(myend(a2)-1) << endl;
    return 0;
}
```

### Q7

```c++
#include <iostream>

using namespace std;

template <typename T, unsigned N>
unsigned getSize(const T (&a)[N]) {
    return N;
}

int main() {
    int a1[] = {1,2,3};
    cout << getSize(a1) << endl;
    char a2[] = "hello,world";
    cout << getSize(a2) << endl;
    return 0;
}
```

### Q8

因为大多数类只定义了!=操作而没有定义<操作，使用!=可以降低对类型的限制。

### Q9

函数模板是创建函数的蓝图；类模板是创建类的蓝图

### Q10

编译器使用显示模板实参来实例化出特定的类

### Q11

```c++
template <typename elemType> class ListItem;
template <typename elemType> class List
{
public:
    List<elemType>();
    List<elemType>(const List<elemType> &);
    List<elemType>& operator=(const List<elemType> &);
    ~List();
    void insert(ListItem<elemType> *ptr, elemType value);
private:
    ListItem<elemType> *front, *end;
};
```

### Q12

```c++
#ifndef BLOB_H
#define BLOB_H

#include <memory>
#include <vector>
#include <string>
#include <initializer_list>
#include <stdexcept>

using namespace std;

template <typename T>
class Blob {
public:
    typedef typename vector<T>::size_type size_type;
    Blob();
    Blob(initializer_list<T> i1);
    size_type size() const { return data->size(); }
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
    BlobPtr() : curr(0) {}
    BlobPtr(Blob<T> &a, size_t sz = 0) : wptr(a.data), curr(sz) {}
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
Blob<T>::Blob() : data(make_shared<vector<T>>()) {}
template <typename T>
Blob<T>::Blob(initializer_list<T> i1) : data(make_shared<vector<T>>(i1)) {}
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
```

### Q13

一对一友好关系，比较相同类型实例化的BlobPtr

### Q14

```c++
#ifndef SCREEN_H
#define SCREEN_H

#include <string>

using namespace std;

using pos = string::size_type;

template <pos H, pos W>
class Screen {
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
inline Screen<H, W> &Screen<H, W>::move(pos r, pos c) {
	cursor = r * W + c;
	return *this;
}

#endif
```

### Q15

```c++
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
```

### Q16

```c++
#ifndef VEC_H
#define VEC_H

#include <utility>
#include <memory>

using namespace std;

template <typename T>
class Vec {
public:
    Vec() : elements(nullptr), first_free(nullptr), cap(nullptr) {};
    Vec(const Vec&);
    Vec& operator=(const Vec&);
    ~Vec();
    void push_back(const T&);
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
        alloc.construct(dest++, move(*elem++));
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
        alloc.construct(dest++, move(*elem++));
    }
    free();
    elements = newdata;
    first_free = dest;
    cap = elements + newcapacity;
}

#endif
```

### Q17

没有不同。
希望使用一个模板类型参数的类型成员时，使用关键字typename显式告诉编译器该名字是一个类型。

### Q18

非法，必须指明U的类型
非法，重用T为函数参数名
非法，inline应放在模板参数列表之后，返回类型之前
非法，缺少返回类型
合法，在模板作用域中，模板参数Ctype屏蔽了之前的类型别名

```c++
template <typename T, typename U, typename V> void f1(T, U, V)
template <typename T> T f2(T&)
template <typename T> inline T foo(T, unsigned int*)
template <typename T> void f4(T, T)
typedef char Ctype
template <typename Ctype> Ctype f5(Ctype)
```

### Q19

```c++
#include <iostream>
#include <vector>
#include <string>

using namespace std;

template <typename T>
void print(T &v) {
    for (typename T::size_type i = 0; i != v.size(); ++i) {
        cout << v[i] << endl;
    }
}

int main() {
    vector<int> v1{1,2,3};
    print(v1);
    vector<string> v2{"a","b","c"};
    print(v2);
    return 0;
}
```

### Q20

```c++
#include <iostream>
#include <vector>
#include <string>

using namespace std;

template <typename T>
void print(T &v) {
    for (typename T::const_iterator iter = v.cbegin(); iter != v.cend(); ++iter) {
        cout << *iter << endl;
    }
}

int main() {
    vector<int> v1{1,2,3};
    print(v1);
    vector<string> v2{"a","b","c"};
    print(v2);
    return 0;
}
```

### Q21

```c++
#ifndef DEBUGDELETE_H
#define DEBUGDELETE_H

#include <iostream>

class DebugDelete {
public:
    DebugDelete(std::ostream &s = std::cerr) : os(s) {}
    template <typename T> void operator()(T *p) const {
        os << "deleting unique_ptr" << std::endl;
        delete p;
    }
private:
    std::ostream &os;
};

#endif
```

### Q22

```c++
// DebugDelete.h
#ifndef DEBUGDELETE_H
#define DEBUGDELETE_H

#include <iostream>

using namespace std;

class DebugDelete {
public:
    DebugDelete(ostream &s = cerr) : os(s) {}
    template <typename T> void operator()(T *p) const {
        os << "deleting unique_ptr" << endl;
        delete p;
    }
private:
    ostream &os;
};

#endif
```

```c++
// TextQuery.h
#ifndef TEXTQUERY_H
#define TEXTQUERY_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <set>
#include "DebugDelete.h"

using namespace std;

class QueryResult;
class TextQuery {
public:
    using line_no = vector<string>::size_type;
    TextQuery(ifstream&);
    QueryResult query(const string&) const;
private:
    shared_ptr<vector<string>> file;
    map<string, shared_ptr<set<line_no>>> wm;
};

class QueryResult {
friend ostream& print(ostream&, const QueryResult&);
public:
    QueryResult(string s,
                shared_ptr<set<TextQuery::line_no>> p,
                shared_ptr<vector<string>> f) :
        sought(s), lines(p), file(f) {}
private:
    string sought;
    shared_ptr<set<TextQuery::line_no>> lines;
    shared_ptr<vector<string>> file;
};

TextQuery::TextQuery(ifstream &is) : file(new vector<string>) {
    string text;
    while (getline(is, text)) {
        file->push_back(text);
        int n = file->size() - 1;
        istringstream line(text);
        string word;
        while (line >> word) {
            auto &lines = wm[word];
            if (!lines)
                lines.reset(new set<line_no>, DebugDelete());
            lines->insert(n);
        }
    }
}

QueryResult TextQuery::query(const string &sought) const {
    static shared_ptr<set<line_no>> nodata(new set<line_no>);
    auto loc = wm.find(sought);
    if (loc == wm.end())
        return QueryResult(sought, nodata, file);
    else
        return QueryResult(sought, loc->second, file);
}

string make_plural(size_t ctr, const string &word, const string &ending) {
    return (ctr > 1) ? word + ending : word;
}

ostream &print(ostream & os, const QueryResult &qr) {
    os << qr.sought << " occurs " << qr.lines->size() << " "
        << make_plural(qr.lines->size(), "times", "s") << endl;
    for (auto num : *qr.lines)
        os << "\t(line " << num+1 << ") " << *(qr.file->begin()+num) << endl;
    return os;
}

#endif
```

```c++
#include <iostream>
#include <fstream>
#include "TextQuery.h"

void runQueries(ifstream &);

int main() {
    ifstream in("test.txt");
    runQueries(in);
    return 0;
}

void runQueries(ifstream &infile) {
    TextQuery tq(infile);
    while (true) {
        cout << "enter word to look for, or q to quit: ";
        string s;
        if (!(cin >> s) || s == "q") break;
        print(cout, tq.query(s)) << endl;
    }
}
```

### Q23

销毁shared_ptr时，调用DebugDelete()

### Q24

```c++
// Blob.h
#ifndef BLOB_H
#define BLOB_H

#include <memory>
#include <vector>
#include <string>
#include <initializer_list>
#include <stdexcept>

using namespace std;

template <typename T>
class Blob {
public:
    typedef typename vector<T>::size_type size_type;
    Blob();
    Blob(initializer_list<T> i1);
    template <typename It>
    Blob(It b, It e) : data(make_shared<vector<T>>(b,e)) {}
    size_type size() const { return data->size(); }
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
    BlobPtr() : curr(0) {}
    BlobPtr(Blob<T> &a, size_t sz = 0) : wptr(a.data), curr(sz) {}
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
Blob<T>::Blob() : data(make_shared<vector<T>>()) {}
template <typename T>
Blob<T>::Blob(initializer_list<T> i1) : data(make_shared<vector<T>>(i1)) {}
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
```

```c++
#include <iostream>
#include <vector>
#include "Blob.h"

int main() {
    vector<int> v{1,2,3};
    Blob<int> a(v.begin(), v.end());
    cout << a.size() << endl;
    return 0;
}
```

### Q25

实例化声明模板类vector，不在本文件中实例化；实例化定义模板类vector<Sales_data>，在本文件中实例化。

### Q26

NoDefault没有默认构造函数，在不提供参数的情况下，NoDefault无法初始化，所以vector<NoDefault>无法实例化

### Q27

（a）实例化，在函数中定义；
（b）实例化，在类中定义；
（c）实例化，在类中定义；
（d）未实例化，使用时实例化；
（e）未实例化，使用时在（a）处实例化；
（f）实例化，sizeof需要知道Stack的定义才能给出一个Stack对象的大小，会实例化。

### Q28

```c++
// DebugDelete.h
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
```

```c++
// SharedPtr.h
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
```

```c++
// UniquePtr.h
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
```

```c++
#include <iostream>
#include "DebugDelete.h"
#include "SharedPtr.h"
#include "UniquePtr.h"

int main() {
    SharedPtr<int> p1(new int(1), DebugDelete("shared_ptr"));
    UniquePtr<int> p2(new int(2), DebugDelete("unique_ptr"));
    return 0;
}
```

### Q29

```c++
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
```

```c++
// SharedPtr.h
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
```

```c++
// Blob.h
#ifndef BLOB_H
#define BLOB_H

#include <memory>
#include <vector>
#include <string>
#include <initializer_list>
#include <stdexcept>
#include "SharedPtr.h"

using namespace std;

template <typename T>
class Blob {
public:
    typedef typename vector<T>::size_type size_type;
    Blob();
    Blob(initializer_list<T> i1);
    template <typename It>
    Blob(It b, It e) : data(new vector<T>(b,e)) {}
    size_type size() const { return data->size(); }
    bool empty() const { return data->empty(); }
    void push_back(const T &t) { data->push_back(t); }
    void push_back(T &&t) { data->push_back(move(t)); }
    void pop_back();
    T& back();
    T& operator[](size_type i);
private:
    SharedPtr<vector<T>> data;
    void check(size_type i, const string &msg) const;
};

template <typename T>
class BlobPtr {
public:
    BlobPtr() : curr(0) {}
    BlobPtr(Blob<T> &a, size_t sz = 0) : wptr(a.data), curr(sz) {}
    T& operator*() const {
        auto p = check(curr, "dereference past end");
        return (*p)[curr];
    }
    BlobPtr& operator++();
    BlobPtr& operator--(); 
private:
    SharedPtr<vector<T>> check(size_t, const string&) const;
    weak_ptr<vector<T>> wptr;
    size_t curr;
};


template <typename T>
Blob<T>::Blob() : data(make_shared<vector<T>>()) {}
template <typename T>
Blob<T>::Blob(initializer_list<T> i1) : data(make_shared<vector<T>>(i1)) {}
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
SharedPtr<vector<T>> BlobPtr<T>::check(size_t i, const string &msg) const {
    if (i >= wptr.lock()->size())
        throw out_of_range(msg);
}

#endif
```

```c++
#include <iostream>
#include <vector>
#include "Blob.h"

int main() {
    std::vector<int> v{1,2,3};
    Blob<int> a(v.begin(), v.end());
    std::cout << a.size() << std::endl;
    return 0;
}
```

### Q30

见Q29

### Q31

unique_ptr将DebugDelete设置为默认删除器，编译时将执行DebugDelete。

### Q32

在模板实参推断过程中，编译器使用函数调用中的实参类型来寻找模板实参，用这些模板实参生成的函数版本与给定的函数调用最为匹配。

### Q33

const转换：可以将一个非const对象的引用（或指针）传递给一个const的引用（或指针）形参。
数组或函数指针转换：如果函数形参不是引用类型，则可以对数组或函数类型的实参应用正常的指针转换。一个数组实参可以转换为一个指向其首元素的指针。类似的，一个函数实参可以转换为一个该函数类型的指针。

### Q34

```
非法，const char[](3)和const char[](6)
合法，const char[](4)和const char[](4)
```

### Q35

合法，T为char
合法，T为double
合法，T为char
非法，T的类型无法确定

### Q36

（a）f1(int*, int*)；
（b）f2(int*, int*)；
（c）f1(const int*, const int*)；
（d）f2(const int*, const int*)；
（e）出错；
（f）f2(int*, const int*)。

### Q37

可以，显示指定模板实参

```c++
#include <iostream>
#include <algorithm>

using namespace std;

int main() {
    int a = 1;
    double b = 2.0;
    cout << max<int>(a,b) << endl;
    cout << max<double>(a,b) << endl;
    return 0;
}
```

### Q38

用户控制返回类型

### Q39

```c++
#include <iostream>
#include <string>

using namespace std;

template <typename T>
int compare(const T &v1, const T &v2) {
    if (v1 < v2) return -1;
    if (v2 < v1) return 1;
    return 0;
}

int main() {
    cout << compare<string>("A","B") << endl;
    return 0;
}
```

### Q40

合法，传递的实参必须支持+0操作，返回类型由+操作的返回类型决定。

### Q41

```c++
#include <iostream>

using namespace std;

template <typename T1, typename T2>
auto sum(T1 lhs, T2 rhs) -> decltype(lhs + rhs) {
    return lhs + rhs;
}

int main() {
    auto res = sum(11111111111, 2);
    cout << res << endl;
    return 0;
}
```

### Q42

（a）int&，int&
（b）const int&，const int&
（c）int，int&&

### Q43

int&

### Q44

T
（a）int
（b）int
（c）int
const T&
（a）int
（b）int
（c）int

### Q45

42，T推断为int，val推断为int&&，vector<int> v；
int，T推断为int&，val推断为int& &&，折叠后为int&，vector<int&> v，引用不能作为容器的元素，会报错。

### Q46

将elem开始的内存中的对象逐个移动到dest开始的内存中。
在每一次循环中，对elem的解引用操作*当中，会返回一个左值，std::move函数将该左值转换为右值，提供给construct函数。

### Q47

```c++
#include <iostream>
#include <utility>

using namespace std;

void g(int &&, int &);

template <typename F, typename T1, typename T2>
void flip(F f, T1 &&t1, T2 &&t2) {
    f(std::forward<T2>(t2), std::forward<T1>(t1));
}

int main() {
    int i = 1;
    flip(g, i, 2);
    return 0;
}

void g(int &&i, int &j) {
    cout << i << " " << j << endl;
}
```

### Q48

```c++
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

template <typename T>
string debug_rep(const T&);
template <typename T>
string debug_rep(T*);
string debug_rep(const string&);
string debug_rep(char *);
string debug_rep(const char *);

int main() {
    return 0;
}

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
string debug_rep(char *p) {
    return debug_rep(string(p));
}
string debug_rep(const char *p) {
    return debug_rep(string(p));
}
```

### Q49

g(T) T:int
g(T*) T:int
g(T) T:const int
g(T*) T:const int
f(T) T:int
f(T) T:int*
f(T) T:const int
f(const T*) T:int

### Q50

```c++
#include <iostream>

using namespace std;

template <typename T>
void f(T) {
    cout << "f(T)" << endl;
}
template <typename T>
void f(const T*) {
    cout << "f(const T*)" << endl;
}
template <typename T>
void g(T) {
    cout << "g(T)" << endl;
}
template <typename T>
void g(T*) {
    cout << "g(T*)" << endl;
}

int main() {
    int i = 42, *p = &i;
    const int ci = 0, *p2 = &ci;
    g(42); g(p); g(ci); g(p2);
    f(42); f(p); f(ci); f(p2);
    return 0;
}
```

### Q51

```c++
#include <iostream>
#include <string>

using namespace std;

template <typename T, typename... Args>
void foo(const T &t, const Args& ... args) {
    cout << sizeof...(Args) << endl;
    cout << sizeof...(args) << endl;
}

int main() {
    int i = 0;
    double d = 3.14;
    string s = "how now brown cow";
    foo(i, s, 42, d);
    foo(s, 42, "hi");
    foo(d, s);
    foo("hi");
    return 0;
}
```

### Q52

见Q51

### Q53

```c++
#include <iostream>
#include <string>

using namespace std;

template <typename T>
ostream &print(ostream &os, const T &t) {
    return os << t;
}
template <typename T, typename... Args>
ostream &print(ostream &os, const T &t, const Args&... rest) {
    os << t << " ";
    return print(os, rest...);
}

int main() {
    int i = 1;
    char c = 'x';
    double d = 0.1;
    string s{"aaa"};
    char a[] = "bbb";
    print(cout, i) << endl;
    print(cout, i, c, d) << endl;
    print(cout, i, c, d, s, a) << endl;
    return 0;
}
```

### Q54

print要求函数参数类型支持<<运算符，所以会报错。

### Q55

当非可变参数版本放在可变参数版本后时，可视为“定义可变参数版本时，非可变参数版本声明不在作用域中”，可变参数版本会无限递归。

### Q56

```c++
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

template <typename T>
string debug_rep(const T&);
template <typename T>
string debug_rep(T*);
string debug_rep(const string&);
string debug_rep(char *);
string debug_rep(const char *);

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
string debug_rep(char *p) {
    return debug_rep(string(p));
}
string debug_rep(const char *p) {
    return debug_rep(string(p));
}

template <typename T>
ostream &print(ostream &os, const T &t) {
    return os << t;
}
template <typename T, typename... Args>
ostream &print(ostream &os, const T &t, const Args&... rest) {
    os << t << " ";
    return print(os, rest...);
}

template <typename... Args>
ostream &errorMsg(ostream &os, const Args&... rest) {
    return print(os, debug_rep(rest)...);
}

int main() {
    string s{"aaa"};
    errorMsg(cout, 1, 0.1, 'a', "aa", "aaa");
    return 0;
}
```

### Q57

error_msg函数中使用initializer_list，但initializer_list只能接受相同类型（或它们的类型可以转换为同一个公共类型）的可变数目实参的函数；而errorMsg函数可以接受不同类型的可变数目实参，更加灵活。

### Q58

```c++
// StrVec.h
#ifndef STRVEC_H
#define STRVEC_H

#include <utility>
#include <memory>
#include <algorithm>
#include <string>
#include <initializer_list>

class StrVec {
    friend bool operator==(StrVec&, StrVec&);
    friend bool operator!=(StrVec&, StrVec&);
    friend bool operator<(StrVec&, StrVec&);
    friend bool operator>(StrVec&, StrVec&);
    friend bool operator<=(StrVec&, StrVec&);
    friend bool operator>=(StrVec&, StrVec&);
public:
    StrVec() : elements(nullptr), first_free(nullptr), cap(nullptr) {};
    StrVec(std::initializer_list<std::string>);
    StrVec(const StrVec&);
    StrVec(StrVec&&) noexcept;
    StrVec& operator=(const StrVec&);
    StrVec& operator=(StrVec&&) noexcept;
    std::string& operator[](std::size_t n) { return elements[n]; }
    const std::string& operator[](std::size_t n) const { return elements[n]; }
    ~StrVec();
    void push_back(const std::string&);
    template <typename... Args> inline void emplace_back(Args&&...);
    size_t size() const { return first_free - elements; }
    size_t capacity() const { return cap - elements; }
    std::string *begin() const { return elements; }
    std::string *end() const { return first_free; }
    void reserve(size_t);
    void resize(size_t);
    void resize(size_t, const std::string&);
private:
    std::allocator<std::string> alloc;
    void chk_n_alloc() { if (size() == capacity()) reallocate(); }
    std::pair<std::string*, std::string*> alloc_n_copy(const std::string*, const std::string*);
    void free();
    void reallocate();
    std::string *elements;
    std::string *first_free;
    std::string *cap;
};

StrVec::StrVec(const StrVec &s) {
    auto newdata = alloc_n_copy(s.begin(), s.end());
    elements = newdata.first;
    first_free = cap = newdata.second;
}

StrVec::StrVec(StrVec&& s) noexcept {
    alloc = std::move(s.alloc);
    elements = std::move(s.elements);
    first_free = std::move(s.first_free);
    cap = std::move(s.cap);
    s.elements = s.first_free = s.cap = nullptr;
}

template <typename... Args>
inline void StrVec::emplace_back(Args&&... args) {
    chk_n_alloc();
    alloc.construct(first_free++, std::forward<Args>(args)...);
}

StrVec& StrVec::operator=(const StrVec &s) {
    auto newdata = alloc_n_copy(s.begin(), s.end());
    free();
    elements = newdata.first;
    first_free = cap = newdata.second;
    return *this;
}

StrVec& StrVec::operator=(StrVec&& s) noexcept {
    if (this != &s) {
        free();
        alloc = std::move(s.alloc);
        elements = std::move(s.elements);
        first_free = std::move(s.first_free);
        cap = std::move(s.cap);
        s.elements = s.first_free = s.cap = nullptr;
    }
    return *this;
}

StrVec::StrVec(std::initializer_list<std::string> l) {
    auto newdata = alloc_n_copy(l.begin(), l.end());
    elements = newdata.first;
    first_free = cap = newdata.second;
}

StrVec::~StrVec() {
    free();
}

void StrVec::push_back(const std::string& s) {
    chk_n_alloc();
    alloc.construct(first_free++, s);
}

void StrVec::reserve(size_t n) {
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

void StrVec::resize(size_t n) {
    resize(n, std::string());
}

void StrVec::resize(size_t n, const std::string& s) {
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

std::pair<std::string*, std::string*> StrVec::alloc_n_copy
        (const std::string *b, const std::string *e) {
    auto data = alloc.allocate(e-b);
    return {data, uninitialized_copy(b, e, data)};
}

void StrVec::free() {

    if (elements) {
        std::for_each(elements, first_free, [this](std::string &p) { alloc.destroy(&p); });
        // for (auto p = first_free; p != elements; ) {
        //     alloc.destroy(--p);
        // }
        alloc.deallocate(elements, cap-elements);
    }
}

void StrVec::reallocate() {
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

bool operator==(StrVec &lhs, StrVec &rhs) {
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}
bool operator!=(StrVec &lhs, StrVec &rhs) {
    return !(lhs == rhs);
}

bool operator<(StrVec &lhs, StrVec &rhs) {
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

bool operator>(StrVec &lhs, StrVec &rhs) {
    return rhs < lhs;
}

bool operator<=(StrVec &lhs, StrVec &rhs) {
    return !(rhs < lhs);
}

bool operator>=(StrVec &lhs, StrVec &rhs) {
    return !(lhs < rhs);
}

#endif
```

```c++
#include <iostream>
#include "StrVec.h"

int main() {
    StrVec s = {"a", "b"};
    s.emplace_back("c");
    s.emplace_back(3,'d');
    for (const auto &i : s) {
        cout << i << endl;
    }
    return 0;
}
```

```c++
// Vec.h
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
```

```c++
#include <iostream>
#include <string>
#include "Vec.h"

int main() {
    Vec<string> s = {"a", "b"};
    s.emplace_back("c");
    s.emplace_back(3,'d');
    for (const auto &i : s) {
        cout << i << endl;
    }
    return 0;
}
```

### Q59

在construct函数中转发函数参数扩展包

### Q60

make_shared是一个可变参数模版函数，它将参数包扩展转发给新构造的对象，最后生成指向该对象的智能指针。

### Q61

```c++
template <typename T, typename... Args>
SharedPtr<T> make_shared(Args&&... args) {
    return new T(std::forward<Args>(args)...);
}
```

### Q62

```c++
// Sales_data.h
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <string>

struct Sales_data {

    friend std::istream& operator>>(std::istream&, Sales_data&);
    friend std::ostream& operator<<(std::ostream&, const Sales_data&);
    friend Sales_data operator+(const Sales_data&, const Sales_data&);
    friend bool operator==(const Sales_data&, const Sales_data&);
	friend class std::hash<Sales_data>;

public:
    Sales_data(std::string s, unsigned n, double p) :
                bookNo(s), units_sold(n), revenue(p*n) {};
    Sales_data() : Sales_data("", 0, 0) {}
    Sales_data(std::string s) : Sales_data(s, 0, 0) {}
    Sales_data(std::istream &is) : Sales_data() { is >> *this; }
    std::string isbn() const { return bookNo; }
    Sales_data& operator+=(const Sales_data&);

private:
    inline double avg_price() const;
    std::string bookNo;
    unsigned units_sold = 0;
    double revenue = 0.0;

};

inline double Sales_data::avg_price() const {
	if (units_sold) {
		return revenue / units_sold;
    }
	else {
		return 0;
    }
}

Sales_data& Sales_data::operator+=(const Sales_data &rhs) {
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}

std::istream& operator>>(std::istream &is, Sales_data &item) {
	double price = 0;
	is >> item.bookNo >> item.units_sold >> price;
    if (is) {
        item.revenue = price * item.units_sold;
    }
	else {
        item = Sales_data();
    }
	return is;
}

std::ostream& operator<<(std::ostream &os, const Sales_data &item) {
	os << item.isbn() << " " << item.units_sold << " " << item.revenue << " " << item.avg_price();
	return os;
}

Sales_data operator+(const Sales_data &lhs, const Sales_data &rhs) {
	Sales_data sum = lhs;
	sum += rhs;
	return sum;
}

bool operator==(const Sales_data &lhs, const Sales_data &rhs) {
	 return lhs.isbn() == rhs.isbn() && 
        lhs.units_sold == rhs.units_sold && 
        lhs.revenue == rhs.revenue;
}

#endif
```

```c++
#include <iostream>
#include <string>
#include <unordered_set>
#include "Sales_data.h"

namespace std {
    template<>
    struct hash<Sales_data> {
        typedef size_t result_type;
        typedef Sales_data argument_type;
        size_t operator()(const Sales_data&) const;
    };
    size_t hash<Sales_data>::operator()(const Sales_data &s) const {
        return hash<string>()(s.bookNo) ^
               hash<unsigned>() (s.units_sold) ^
               hash<double>() (s.revenue);
    }
}

int main() {
    unordered_multiset<Sales_data> s;
    Sales_data sales_data1("001", 1, 100);
    s.emplace(sales_data1);
    s.emplace("002", 1, 200);
    Sales_data sales_data3("003");
    s.emplace(sales_data3);
    for(const auto &item : s)
        cout << "the hash code of " << item.isbn() <<":\n0x" << hex << hash<Sales_data>()(item) << "\n";
    return 0;
}
```

### Q63

```c++
#include <iostream>
#include <vector>
#include <string>

using namespace std;

template <typename T>
size_t get_nums(vector<T> &v, T t) {
    size_t cnt = 0;
    for (auto it = v.begin(); it != v.end(); ++it) {
        if (*it == t) {
            ++cnt;
        }
    }
    return cnt;
}

int main() {
    vector<int> v1{1,1,2,3,5};
    vector<double> v2{1.0,1.0,2.0,3.0,5.0};
    vector<string> v3{"a","a","aa","aaa","aaaaa"};
    cout << get_nums(v1,1) << endl;
    cout << get_nums(v2,1.0) << endl;
    cout << get_nums(v3,string("a")) << endl;
    return 0;
}
```

### Q64

```c++
#include <iostream>
#include <vector>
#include <string>
#include <cstring>

using namespace std;

template <typename T>
size_t get_nums(vector<T> &v, T t) {
    size_t cnt = 0;
    for (auto it = v.begin(); it != v.end(); ++it) {
        if (*it == t) {
            ++cnt;
        }
    }
    return cnt;
}
template <>
size_t get_nums(vector<const char*> &v, const char* t) {
    size_t cnt = 0;
    for (auto it = v.begin(); it != v.end(); ++it) {
        if (strcmp(*it, t) == 0) {
            ++cnt;
        }
    }
    return cnt;
}

int main() {
    vector<int> v1{1,1,2,3,5};
    vector<double> v2{1.0,1.0,2.0,3.0,5.0};
    vector<string> v3{"a","a","aa","aaa","aaaaa"};
    vector<const char*> v4{"a","a","aa","aaa","aaaaa"};
    cout << get_nums(v1,1) << endl;
    cout << get_nums(v2,1.0) << endl;
    cout << get_nums(v3,string("a")) << endl;
    cout << get_nums(v4,"a") << endl;
    return 0;
}
```

### Q65

```c++
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
```

### Q66

函数匹配顺序不同，多个具有同样好的匹配函数优先选择非模板版本

### Q67

不会，特例化版本本质上是一个，而非函数名的一个重载版本