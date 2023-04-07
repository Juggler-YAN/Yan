# Chapter 14

### Q1

区别
当一个重载运算符是成员函数时，this绑定到左侧的运算对象，成员运算符函数的显示参数数量比运算对象的数量少一个；
逻辑与运算符、逻辑或运算符和逗号运算符的运算对象求值顺序规则无法保留下来。
一样
对于一个重载的运算符来说，其优先级和结合律与对应的内置运算保持一致。

### Q2

```c++
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <string>

struct Sales_data {

    friend std::istream& operator>>(std::istream&, Sales_data&);
    friend std::ostream& operator<<(std::ostream&, const Sales_data&);
    friend Sales_data operator+(const Sales_data&, const Sales_data&);

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

#endif
```

### Q3

无对应
string
vector
string

### Q4

不应该
应该
应该
必须是
不应该
不应该
不应该
必须是

### Q5

```c++
#ifndef BOOK_H
#define BOOK_H

#include <string>

class Book {

    friend std::istream& operator>>(std::istream&, Book&);
    friend std::ostream& operator<<(std::ostream&, const Book&);
    friend bool operator==(const Book&, const Book&);
    friend bool operator!=(const Book&, const Book&);

public:
    Book() = default;
    Book(unsigned int a, std::string b, std::string c) : 
        no(a), name(b), author(c) {}
    Book(std::istream &is) { is >> *this; }

private:
    unsigned int no;
    std::string name;
    std::string author;

};

std::istream& operator>>(std::istream &is, Book &book) {
    is >> book.no >> book.name >> book.author;
    return is;
}

std::ostream& operator<<(std::ostream &os, const Book &book) {
    os << book.no << " " << book.name << " " << book.author;
    return os;
}

bool operator==(const Book &book1, const Book &book2) {
    return book1.no == book2.no;
}

bool operator!=(const Book &book1, const Book &book2) {
    return !(book1 == book2);
}

#endif
```

### Q6

见Q2

### Q7

```c++
#ifndef STRING_H
#define STRING_H

#include <iostream>
#include <algorithm>
#include <memory>
#include <cstring>

class String {
    friend std::ostream& operator<<(std::ostream&, const String&);
public:
    String(): elements(nullptr), first_free(nullptr) {}
    String(const char *);
    String(const String&);
    String(String&&) noexcept;
    String& operator=(const String&);
    String& operator=(String&&) noexcept;
    ~String();
    char * begin() const { return elements; }
    char * end() const { return first_free; }
private:
    std::allocator<char> alloc;
    std::pair<char*, char*> alloc_n_copy(const char*, const char*);
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
    std::cout << "String(String &&s) noexcept" << std::endl;
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
    std::cout << "String& operator=(String &&s) noexcept" << std::endl;
    return *this;
}

String::~String() {
    free();
}

std::ostream& operator<<(std::ostream &os, const String &s) {
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

std::pair<char*, char*> String::alloc_n_copy
        (const char *b, const char *e) {
    auto data = alloc.allocate(e-b);
    return {data, std::uninitialized_copy(b, e, data)};
}

#endif
```

### Q8

见Q5

### Q9

见Q2

### Q10

输入正确
输入错误，10读入bookNo，24读入units_sold，.95读入revenue，与想要得到的结果不符

### Q11

没有对读取操作失败的处理，与上题结果一样

### Q12

```c++
#ifndef BOOK_H
#define BOOK_H

#include <string>

class Book {

    friend std::istream& operator>>(std::istream&, Book&);
    friend std::ostream& operator<<(std::ostream&, const Book&);
    friend bool operator==(const Book&, const Book&);
    friend bool operator!=(const Book&, const Book&);

public:
    Book() = default;
    Book(unsigned int a, std::string b, std::string c) : 
        no(a), name(b), author(c) {}
    Book(std::istream &is) { is >> *this; }

private:
    unsigned int no;
    std::string name;
    std::string author;

};

std::istream& operator>>(std::istream &is, Book &book) {
    is >> book.no >> book.name >> book.author;
    if (!is) {
        book = Book();
    }
    return is;
}

std::ostream& operator<<(std::ostream &os, const Book &book) {
    os << book.no << " " << book.name << " " << book.author;
    return os;
}

bool operator==(const Book &book1, const Book &book2) {
    return book1.no == book2.no;
}

bool operator!=(const Book &book1, const Book &book2) {
    return !(book1 == book2);
}

#endif
```

### Q13

见Q2

### Q14

用operator+=会避免创建一个临时对象

### Q15

见Q5

### Q16

```c++
// StrBlob, StrBlobPtr
#ifndef STRBLOB_H
#define STRBLOB_H

#include <memory>
#include <initializer_list>
#include <vector>
#include <string>
#include <stdexcept>

class ConstStrBlobPtr;

class StrBlob {
    friend class ConstStrBlobPtr;
    friend bool operator==(const StrBlob&, const StrBlob&);
    friend bool operator!=(const StrBlob&, const StrBlob&);
public:
    typedef std::vector<std::string>::size_type size_type;
    StrBlob();
    StrBlob(std::initializer_list<std::string> i1);
    StrBlob(const StrBlob&);
    StrBlob& operator=(const StrBlob&);
    size_type size() const { return data->size(); }
    bool empty() const { return data->empty(); }
    void push_back(const std::string &t) { data->push_back(t); }
	void push_back(std::string &&t) { data->push_back(std::move(t)); }
    void pop_back();
    std::string& front();
    std::string& back();
    const std::string& front() const;
    const std::string& back() const;
	ConstStrBlobPtr begin();
	ConstStrBlobPtr end();
private:
    std::shared_ptr<std::vector<std::string>> data;
    void check(size_type i, const std::string &msg) const;
};

class ConstStrBlobPtr {
public:
    friend bool operator==(const ConstStrBlobPtr&, const ConstStrBlobPtr&);
    friend bool operator!=(const ConstStrBlobPtr&, const ConstStrBlobPtr&);
    ConstStrBlobPtr() : curr(0) {}
    ConstStrBlobPtr(const StrBlob &a, size_t sz = 0) : wptr(a.data), curr(sz) {}
    std::string& deref() const;
    ConstStrBlobPtr& incr();
private:
    std::shared_ptr<std::vector<std::string>> check(std::size_t, const std::string&) const;
    std::weak_ptr<std::vector<std::string>> wptr;
    std::size_t curr;
};

ConstStrBlobPtr StrBlob::begin() { return ConstStrBlobPtr(*this); }
ConstStrBlobPtr StrBlob::end() {
    auto ret = ConstStrBlobPtr(*this, data->size());
    return ret;
}

StrBlob::StrBlob(): data(std::make_shared<std::vector<std::string>>()) {};
StrBlob::StrBlob(std::initializer_list<std::string> i1): data(std::make_shared<std::vector<std::string>>(i1)) {};
StrBlob::StrBlob(const StrBlob& sb) { data = std::make_shared<std::vector<std::string>>(*sb.data); }
StrBlob& StrBlob::operator=(const StrBlob& sb) { data = std::make_shared<std::vector<std::string>>(*sb.data); return *this; }

void StrBlob::check(size_type i, const std::string &msg) const {
    if (i >= data->size())
        throw std::out_of_range(msg);
}

std::string& StrBlob::front() {
    check(0, "front on empty StrBlob");
    return data->front();
}

std::string& StrBlob::back() {
    check(0, "back on empty StrBlob");
    return data->back();
}

const std::string& StrBlob::front() const {
    check(0, "front on empty StrBlob");
    return data->front();
}

const std::string& StrBlob::back() const {
    check(0, "back on empty StrBlob");
    return data->back();
}

void StrBlob::pop_back() {
    check(0, "pop_back on empty StrBlob");
    return data->pop_back();
}

std::shared_ptr<std::vector<std::string>> ConstStrBlobPtr::check(std::size_t i, const std::string& msg) const {
    auto ret = wptr.lock();
    if (!ret)
        throw std::runtime_error("unbound StrBlobPtr");
    if (i >= ret->size())
        throw std::out_of_range(msg);
    return ret;
}

std::string& ConstStrBlobPtr::deref() const {
    auto p = check(curr, "dereference past end");
    return (*p)[curr];
}

ConstStrBlobPtr& ConstStrBlobPtr::incr() {
    check(curr, "increment past end of StrBlobPtr");
    ++curr;
    return *this;
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

#endif
```

```c++
// StrVec
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
public:
    StrVec() : elements(nullptr), first_free(nullptr), cap(nullptr) {};
    StrVec(std::initializer_list<std::string>);
    StrVec(const StrVec&);
    StrVec(StrVec&&) noexcept;
    StrVec& operator=(const StrVec&);
    StrVec& operator=(StrVec&&) noexcept;
    ~StrVec();
    void push_back(const std::string&);
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

StrVec::StrVec(std::initializer_list<std::string> l) {
    auto newdata = alloc_n_copy(l.begin(), l.end());
    elements = newdata.first;
    first_free = cap = newdata.second;
}

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
#endif
```

```c++
// String
#ifndef STRING_H
#define STRING_H

#include <iostream>
#include <algorithm>
#include <memory>
#include <cstring>

class String {
    friend std::ostream& operator<<(std::ostream&, const String&);
    friend bool operator==(const String&, const String&);
    friend bool operator!=(const String&, const String&);
public:
    String(): elements(nullptr), first_free(nullptr) {}
    String(const char *);
    String(const String&);
    String(String&&) noexcept;
    String& operator=(const String&);
    String& operator=(String&&) noexcept;
    ~String();
    char * begin() const { return elements; }
    char * end() const { return first_free; }
private:
    std::allocator<char> alloc;
    std::pair<char*, char*> alloc_n_copy(const char*, const char*);
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
    std::cout << "String(String &&s) noexcept" << std::endl;
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
    std::cout << "String& operator=(String &&s) noexcept" << std::endl;
    return *this;
}

String::~String() {
    free();
}

std::ostream& operator<<(std::ostream &os, const String &s) {
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

std::pair<char*, char*> String::alloc_n_copy
        (const char *b, const char *e) {
    auto data = alloc.allocate(e-b);
    return {data, std::uninitialized_copy(b, e, data)};
}

bool operator==(const String &lhs, const String &rhs) {
    return (lhs.first_free-lhs.elements) == (rhs.first_free-rhs.elements) &&
           std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool operator!=(const String &lhs, const String &rhs) {
    return !(lhs == rhs);
}

#endif
```

### Q17

见Q5

### Q18

```c++
// StrBlob
#ifndef STRBLOB_H
#define STRBLOB_H

#include <memory>
#include <initializer_list>
#include <vector>
#include <string>
#include <stdexcept>

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
    typedef std::vector<std::string>::size_type size_type;
    StrBlob();
    StrBlob(std::initializer_list<std::string> i1);
    StrBlob(const StrBlob&);
    StrBlob& operator=(const StrBlob&);
    size_type size() const { return data->size(); }
    bool empty() const { return data->empty(); }
    void push_back(const std::string &t) { data->push_back(t); }
	void push_back(std::string &&t) { data->push_back(std::move(t)); }
    void pop_back();
    std::string& front();
    std::string& back();
    const std::string& front() const;
    const std::string& back() const;
	ConstStrBlobPtr begin();
	ConstStrBlobPtr end();
private:
    std::shared_ptr<std::vector<std::string>> data;
    void check(size_type i, const std::string &msg) const;
};

class ConstStrBlobPtr {
public:
    friend bool operator==(const ConstStrBlobPtr&, const ConstStrBlobPtr&);
    friend bool operator!=(const ConstStrBlobPtr&, const ConstStrBlobPtr&);
    ConstStrBlobPtr() : curr(0) {}
    ConstStrBlobPtr(const StrBlob &a, size_t sz = 0) : wptr(a.data), curr(sz) {}
    std::string& deref() const;
    ConstStrBlobPtr& incr();
private:
    std::shared_ptr<std::vector<std::string>> check(std::size_t, const std::string&) const;
    std::weak_ptr<std::vector<std::string>> wptr;
    std::size_t curr;
};

ConstStrBlobPtr StrBlob::begin() { return ConstStrBlobPtr(*this); }
ConstStrBlobPtr StrBlob::end() {
    auto ret = ConstStrBlobPtr(*this, data->size());
    return ret;
}

StrBlob::StrBlob(): data(std::make_shared<std::vector<std::string>>()) {};
StrBlob::StrBlob(std::initializer_list<std::string> i1): data(std::make_shared<std::vector<std::string>>(i1)) {};
StrBlob::StrBlob(const StrBlob& sb) { data = std::make_shared<std::vector<std::string>>(*sb.data); }
StrBlob& StrBlob::operator=(const StrBlob& sb) { data = std::make_shared<std::vector<std::string>>(*sb.data); return *this; }

void StrBlob::check(size_type i, const std::string &msg) const {
    if (i >= data->size())
        throw std::out_of_range(msg);
}

std::string& StrBlob::front() {
    check(0, "front on empty StrBlob");
    return data->front();
}

std::string& StrBlob::back() {
    check(0, "back on empty StrBlob");
    return data->back();
}

const std::string& StrBlob::front() const {
    check(0, "front on empty StrBlob");
    return data->front();
}

const std::string& StrBlob::back() const {
    check(0, "back on empty StrBlob");
    return data->back();
}

void StrBlob::pop_back() {
    check(0, "pop_back on empty StrBlob");
    return data->pop_back();
}

std::shared_ptr<std::vector<std::string>> ConstStrBlobPtr::check(std::size_t i, const std::string& msg) const {
    auto ret = wptr.lock();
    if (!ret)
        throw std::runtime_error("unbound StrBlobPtr");
    if (i >= ret->size())
        throw std::out_of_range(msg);
    return ret;
}

std::string& ConstStrBlobPtr::deref() const {
    auto p = check(curr, "dereference past end");
    return (*p)[curr];
}

ConstStrBlobPtr& ConstStrBlobPtr::incr() {
    check(curr, "increment past end of StrBlobPtr");
    ++curr;
    return *this;
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
    return std::lexicographical_compare(lhs.data->begin(), lhs.data->end(), rhs.data->begin(), rhs.data->end());
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
```

```c++
// StrVec
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
    ~StrVec();
    void push_back(const std::string&);
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

StrVec::StrVec(std::initializer_list<std::string> l) {
    auto newdata = alloc_n_copy(l.begin(), l.end());
    elements = newdata.first;
    first_free = cap = newdata.second;
}

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
// String
#ifndef STRING_H
#define STRING_H

#include <iostream>
#include <algorithm>
#include <memory>
#include <cstring>

class String {
    friend std::ostream& operator<<(std::ostream&, const String&);
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
    ~String();
    char * begin() const { return elements; }
    char * end() const { return first_free; }
private:
    std::allocator<char> alloc;
    std::pair<char*, char*> alloc_n_copy(const char*, const char*);
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
    std::cout << "String(String &&s) noexcept" << std::endl;
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
    std::cout << "String& operator=(String &&s) noexcept" << std::endl;
    return *this;
}

String::~String() {
    free();
}

std::ostream& operator<<(std::ostream &os, const String &s) {
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

std::pair<char*, char*> String::alloc_n_copy
        (const char *b, const char *e) {
    auto data = alloc.allocate(e-b);
    return {data, std::uninitialized_copy(b, e, data)};
}

bool operator==(const String &lhs, const String &rhs) {
    return (lhs.first_free-lhs.elements) == (rhs.first_free-rhs.elements) &&
           std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool operator!=(const String &lhs, const String &rhs) {
    return !(lhs == rhs);
}


bool operator<(const String &lhs, const String &rhs) {
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), lhs.end());
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
```

### Q19

```c++
#ifndef BOOK_H
#define BOOK_H

#include <string>

class Book {

    friend std::istream& operator>>(std::istream&, Book&);
    friend std::ostream& operator<<(std::ostream&, const Book&);
    friend bool operator==(const Book&, const Book&);
    friend bool operator!=(const Book&, const Book&);
    friend bool operator<(const Book&, const Book&);
    friend bool operator>(const Book&, const Book&);
    friend bool operator<=(const Book&, const Book&);
    friend bool operator>=(const Book&, const Book&);

public:
    Book() = default;
    Book(unsigned int a, std::string b, std::string c) : 
        no(a), name(b), author(c) {}
    Book(std::istream &is) { is >> *this; }

private:
    unsigned int no;
    std::string name;
    std::string author;

};

std::istream& operator>>(std::istream &is, Book &book) {
    is >> book.no >> book.name >> book.author;
    if (!is) {
        book = Book();
    }
    return is;
}

std::ostream& operator<<(std::ostream &os, const Book &book) {
    os << book.no << " " << book.name << " " << book.author;
    return os;
}

bool operator==(const Book &book1, const Book &book2) {
    return book1.no == book2.no;
}

bool operator!=(const Book &book1, const Book &book2) {
    return !(book1 == book2);
}

bool operator<(const Book &book1, const Book &book2) {
    return book1.no < book2.no;
}

bool operator>(const Book &book1, const Book &book2) {
    return book2 < book1;
}

bool operator<=(const Book &book1, const Book &book2) {
    return !(book2 < book1);
}

bool operator>=(const Book &book1, const Book &book2) {
    return !(book1 < book2);
}

#endif
```

### Q20

见Q2

### Q21

多使用了一个Sales_data的临时变量

```c++
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <string>

struct Sales_data {

    friend std::istream& operator>>(std::istream&, Sales_data&);
    friend std::ostream& operator<<(std::ostream&, const Sales_data&);
    friend Sales_data operator+(const Sales_data&, const Sales_data&);

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

// Sales_data operator+(const Sales_data &lhs, const Sales_data &rhs) {
// 	Sales_data sum = lhs;
// 	sum += rhs;
// 	return sum;
// }

// Sales_data& Sales_data::operator+=(const Sales_data &rhs) {
//     units_sold += rhs.units_sold;
//     revenue += rhs.revenue;
//     return *this;
// }

Sales_data operator+(const Sales_data &lhs, const Sales_data &rhs) {
	Sales_data sum;
    sum.units_sold = lhs.units_sold + rhs.units_sold;
	sum.revenue = lhs.revenue + rhs.revenue;
	return sum;
}

Sales_data& Sales_data::operator+=(const Sales_data &rhs) {
    Sales_data data;
    *this = data + rhs;
    return *this;
}

#endif
```

### Q22

```c++
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <string>

struct Sales_data {

    friend std::istream& operator>>(std::istream&, Sales_data&);
    friend std::ostream& operator<<(std::ostream&, const Sales_data&);
    friend Sales_data operator+(const Sales_data&, const Sales_data&);

public:
    Sales_data(std::string s, unsigned n, double p) :
                bookNo(s), units_sold(n), revenue(p*n) {};
    Sales_data() : Sales_data("", 0, 0) {}
    Sales_data(std::string s) : Sales_data(s, 0, 0) {}
    Sales_data(std::istream &is) : Sales_data() { is >> *this; }
    std::string isbn() const { return bookNo; }
    Sales_data& operator+=(const Sales_data&);
    Sales_data& operator=(const std::string&);

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

Sales_data& Sales_data::operator+=(const Sales_data &rhs) {
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}

Sales_data& Sales_data::operator=(const std::string &s) {
    *this = Sales_data(s);
    return *this;
}

#endif
```

### Q23

```c++
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
    StrVec& operator=(const std::initializer_list<std::string>);
    ~StrVec();
    void push_back(const std::string&);
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

StrVec::StrVec(std::initializer_list<std::string> l) {
    auto newdata = alloc_n_copy(l.begin(), l.end());
    elements = newdata.first;
    first_free = cap = newdata.second;
}

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

StrVec& StrVec::operator=(const std::initializer_list<std::string> l) {
    auto newdata = alloc_n_copy(l.begin(), l.end());
    free();
    elements = newdata.first;
    first_free = cap = newdata.second;
    return *this;
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

### Q24

```c++
#ifndef BOOK_H
#define BOOK_H

#include <string>

class Book {

    friend std::istream& operator>>(std::istream&, Book&);
    friend std::ostream& operator<<(std::ostream&, const Book&);
    friend bool operator==(const Book&, const Book&);
    friend bool operator!=(const Book&, const Book&);
    friend bool operator<(const Book&, const Book&);
    friend bool operator>(const Book&, const Book&);
    friend bool operator<=(const Book&, const Book&);
    friend bool operator>=(const Book&, const Book&);

public:
    Book() = default;
    Book(unsigned int a, std::string b, std::string c) : 
        no(a), name(b), author(c) {}
    Book(std::istream &is) { is >> *this; }
    Book& operator=(const Book &);
    Book& operator=(Book&&) noexcept;

private:
    unsigned int no;
    std::string name;
    std::string author;

};

std::istream& operator>>(std::istream &is, Book &book) {
    is >> book.no >> book.name >> book.author;
    if (!is) {
        book = Book();
    }
    return is;
}

std::ostream& operator<<(std::ostream &os, const Book &book) {
    os << book.no << " " << book.name << " " << book.author;
    return os;
}

bool operator==(const Book &book1, const Book &book2) {
    return book1.no == book2.no;
}

bool operator!=(const Book &book1, const Book &book2) {
    return !(book1 == book2);
}

bool operator<(const Book &book1, const Book &book2) {
    return book1.no < book2.no;
}

bool operator>(const Book &book1, const Book &book2) {
    return book2 < book1;
}

bool operator<=(const Book &book1, const Book &book2) {
    return !(book2 < book1);
}

bool operator>=(const Book &book1, const Book &book2) {
    return !(book1 < book2);
}

Book& Book::operator=(const Book &book) {
    no = book.no;
    name = book.name;
    author = book.author;
    return *this;
}

Book& Book::operator=(Book &&book) noexcept {
    if (this != &book) {
        no = std::move(book.no);
        name = std::move(book.name);
        author = std::move(book.author);
    }
    return *this;
}
#endif
```

### Q25

见Q24

### Q26

```c++
// StrBlob
#ifndef STRBLOB_H
#define STRBLOB_H

#include <memory>
#include <initializer_list>
#include <vector>
#include <string>
#include <stdexcept>

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
    typedef std::vector<std::string>::size_type size_type;
    StrBlob();
    StrBlob(std::initializer_list<std::string> i1);
    StrBlob(const StrBlob&);
    StrBlob& operator=(const StrBlob&);
    std::string& operator[](size_t n) { return (*data)[n]; }
    const std::string& operator[](size_t n) const { return (*data)[n]; }
    size_type size() const { return data->size(); }
    bool empty() const { return data->empty(); }
    void push_back(const std::string &t) { data->push_back(t); }
	void push_back(std::string &&t) { data->push_back(std::move(t)); }
    void pop_back();
    std::string& front();
    std::string& back();
    const std::string& front() const;
    const std::string& back() const;
	ConstStrBlobPtr begin();
	ConstStrBlobPtr end();
private:
    std::shared_ptr<std::vector<std::string>> data;
    void check(size_type i, const std::string &msg) const;
};

class ConstStrBlobPtr {
public:
    friend bool operator==(const ConstStrBlobPtr&, const ConstStrBlobPtr&);
    friend bool operator!=(const ConstStrBlobPtr&, const ConstStrBlobPtr&);
    ConstStrBlobPtr() : curr(0) {}
    ConstStrBlobPtr(const StrBlob &a, size_t sz = 0) : wptr(a.data), curr(sz) {}
    std::string& deref() const;
    ConstStrBlobPtr& incr();
private:
    std::shared_ptr<std::vector<std::string>> check(std::size_t, const std::string&) const;
    std::weak_ptr<std::vector<std::string>> wptr;
    std::size_t curr;
};

ConstStrBlobPtr StrBlob::begin() { return ConstStrBlobPtr(*this); }
ConstStrBlobPtr StrBlob::end() {
    auto ret = ConstStrBlobPtr(*this, data->size());
    return ret;
}

StrBlob::StrBlob(): data(std::make_shared<std::vector<std::string>>()) {};
StrBlob::StrBlob(std::initializer_list<std::string> i1): data(std::make_shared<std::vector<std::string>>(i1)) {};
StrBlob::StrBlob(const StrBlob& sb) { data = std::make_shared<std::vector<std::string>>(*sb.data); }
StrBlob& StrBlob::operator=(const StrBlob& sb) { data = std::make_shared<std::vector<std::string>>(*sb.data); return *this; }

void StrBlob::check(size_type i, const std::string &msg) const {
    if (i >= data->size())
        throw std::out_of_range(msg);
}

std::string& StrBlob::front() {
    check(0, "front on empty StrBlob");
    return data->front();
}

std::string& StrBlob::back() {
    check(0, "back on empty StrBlob");
    return data->back();
}

const std::string& StrBlob::front() const {
    check(0, "front on empty StrBlob");
    return data->front();
}

const std::string& StrBlob::back() const {
    check(0, "back on empty StrBlob");
    return data->back();
}

void StrBlob::pop_back() {
    check(0, "pop_back on empty StrBlob");
    return data->pop_back();
}

std::shared_ptr<std::vector<std::string>> ConstStrBlobPtr::check(std::size_t i, const std::string& msg) const {
    auto ret = wptr.lock();
    if (!ret)
        throw std::runtime_error("unbound StrBlobPtr");
    if (i >= ret->size())
        throw std::out_of_range(msg);
    return ret;
}

std::string& ConstStrBlobPtr::deref() const {
    auto p = check(curr, "dereference past end");
    return (*p)[curr];
}

ConstStrBlobPtr& ConstStrBlobPtr::incr() {
    check(curr, "increment past end of StrBlobPtr");
    ++curr;
    return *this;
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
    return std::lexicographical_compare(lhs.data->begin(), lhs.data->end(), rhs.data->begin(), rhs.data->end());
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
```

```c++
// StrVec
#ifndef STRVEC_H
#define STRVEC_H

#include <utility>
#include <memory>
#include <algorithm>
#include <string>

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

StrVec& StrVec::operator=(const std::initializer_list<std::string> l) {
    auto newdata = alloc_n_copy(l.begin(), l.end());
    free();
    elements = newdata.first;
    first_free = cap = newdata.second;
    return *this;
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
// String
#ifndef STRING_H
#define STRING_H

#include <iostream>
#include <algorithm>
#include <memory>
#include <cstring>

class String {
    friend std::ostream& operator<<(std::ostream&, const String&);
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
    std::allocator<char> alloc;
    std::pair<char*, char*> alloc_n_copy(const char*, const char*);
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
    std::cout << "String(String &&s) noexcept" << std::endl;
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
    std::cout << "String& operator=(String &&s) noexcept" << std::endl;
    return *this;
}

String::~String() {
    free();
}

std::ostream& operator<<(std::ostream &os, const String &s) {
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

std::pair<char*, char*> String::alloc_n_copy
        (const char *b, const char *e) {
    auto data = alloc.allocate(e-b);
    return {data, std::uninitialized_copy(b, e, data)};
}

bool operator==(const String &lhs, const String &rhs) {
    return (lhs.first_free-lhs.elements) == (rhs.first_free-rhs.elements) &&
           std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool operator!=(const String &lhs, const String &rhs) {
    return !(lhs == rhs);
}

bool operator<(const String &lhs, const String &rhs) {
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), lhs.end());
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
```

### Q27

```c++
#ifndef STRBLOB_H
#define STRBLOB_H

#include <memory>
#include <initializer_list>
#include <vector>
#include <string>
#include <stdexcept>

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
    typedef std::vector<std::string>::size_type size_type;
    StrBlob();
    StrBlob(std::initializer_list<std::string> i1);
    StrBlob(const StrBlob&);
    StrBlob& operator=(const StrBlob&);
    std::string& operator[](size_t n) { return (*data)[n]; }
    const std::string& operator[](size_t n) const { return (*data)[n]; }
    size_type size() const { return data->size(); }
    bool empty() const { return data->empty(); }
    void push_back(const std::string &t) { data->push_back(t); }
	void push_back(std::string &&t) { data->push_back(std::move(t)); }
    void pop_back();
    std::string& front();
    std::string& back();
    const std::string& front() const;
    const std::string& back() const;
	ConstStrBlobPtr begin();
	ConstStrBlobPtr end();
private:
    std::shared_ptr<std::vector<std::string>> data;
    void check(size_type i, const std::string &msg) const;
};

class ConstStrBlobPtr {
public:
    friend bool operator==(const ConstStrBlobPtr&, const ConstStrBlobPtr&);
    friend bool operator!=(const ConstStrBlobPtr&, const ConstStrBlobPtr&);
    ConstStrBlobPtr() : curr(0) {}
    ConstStrBlobPtr(const StrBlob &a, size_t sz = 0) : wptr(a.data), curr(sz) {}
    std::string& deref() const;
    ConstStrBlobPtr& incr();
    ConstStrBlobPtr& operator++();
    ConstStrBlobPtr& operator--();
    ConstStrBlobPtr operator++(int);
    ConstStrBlobPtr operator--(int);
private:
    std::shared_ptr<std::vector<std::string>> check(std::size_t, const std::string&) const;
    std::weak_ptr<std::vector<std::string>> wptr;
    std::size_t curr;
};

ConstStrBlobPtr StrBlob::begin() { return ConstStrBlobPtr(*this); }
ConstStrBlobPtr StrBlob::end() {
    auto ret = ConstStrBlobPtr(*this, data->size());
    return ret;
}

StrBlob::StrBlob(): data(std::make_shared<std::vector<std::string>>()) {};
StrBlob::StrBlob(std::initializer_list<std::string> i1): data(std::make_shared<std::vector<std::string>>(i1)) {};
StrBlob::StrBlob(const StrBlob& sb) { data = std::make_shared<std::vector<std::string>>(*sb.data); }
StrBlob& StrBlob::operator=(const StrBlob& sb) { data = std::make_shared<std::vector<std::string>>(*sb.data); return *this; }

void StrBlob::check(size_type i, const std::string &msg) const {
    if (i >= data->size())
        throw std::out_of_range(msg);
}

std::string& StrBlob::front() {
    check(0, "front on empty StrBlob");
    return data->front();
}

std::string& StrBlob::back() {
    check(0, "back on empty StrBlob");
    return data->back();
}

const std::string& StrBlob::front() const {
    check(0, "front on empty StrBlob");
    return data->front();
}

const std::string& StrBlob::back() const {
    check(0, "back on empty StrBlob");
    return data->back();
}

void StrBlob::pop_back() {
    check(0, "pop_back on empty StrBlob");
    return data->pop_back();
}

std::shared_ptr<std::vector<std::string>> ConstStrBlobPtr::check(std::size_t i, const std::string& msg) const {
    auto ret = wptr.lock();
    if (!ret)
        throw std::runtime_error("unbound StrBlobPtr");
    if (i >= ret->size())
        throw std::out_of_range(msg);
    return ret;
}

std::string& ConstStrBlobPtr::deref() const {
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
    return std::lexicographical_compare(lhs.data->begin(), lhs.data->end(), rhs.data->begin(), rhs.data->end());
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
```

### Q28

见Q27

### Q29

递增递减会改变对象，没有必要定义成const

### Q30

```c++
#ifndef STRBLOB_H
#define STRBLOB_H

#include <memory>
#include <initializer_list>
#include <vector>
#include <string>
#include <stdexcept>

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
    typedef std::vector<std::string>::size_type size_type;
    StrBlob();
    StrBlob(std::initializer_list<std::string> i1);
    StrBlob(const StrBlob&);
    StrBlob& operator=(const StrBlob&);
    std::string& operator[](size_t n) { return (*data)[n]; }
    const std::string& operator[](size_t n) const { return (*data)[n]; }
    size_type size() const { return data->size(); }
    bool empty() const { return data->empty(); }
    void push_back(const std::string &t) { data->push_back(t); }
	void push_back(std::string &&t) { data->push_back(std::move(t)); }
    void pop_back();
    std::string& front();
    std::string& back();
    const std::string& front() const;
    const std::string& back() const;
	ConstStrBlobPtr begin();
	ConstStrBlobPtr end();
private:
    std::shared_ptr<std::vector<std::string>> data;
    void check(size_type i, const std::string &msg) const;
};

class ConstStrBlobPtr {
public:
    friend bool operator==(const ConstStrBlobPtr&, const ConstStrBlobPtr&);
    friend bool operator!=(const ConstStrBlobPtr&, const ConstStrBlobPtr&);
    ConstStrBlobPtr() : curr(0) {}
    ConstStrBlobPtr(const StrBlob &a, size_t sz = 0) : wptr(a.data), curr(sz) {}
    std::string& deref() const;
    ConstStrBlobPtr& incr();
    ConstStrBlobPtr& operator++();
    ConstStrBlobPtr& operator--();
    ConstStrBlobPtr operator++(int);
    ConstStrBlobPtr operator--(int);
    const std::string& operator*() const {
        auto p = check(curr, "dereference past end");
        return (*p)[curr];
    }
    const std::string* operator->() const {
        return & this->operator*();
    }
private:
    std::shared_ptr<std::vector<std::string>> check(std::size_t, const std::string&) const;
    std::weak_ptr<std::vector<std::string>> wptr;
    std::size_t curr;
};

ConstStrBlobPtr StrBlob::begin() { return ConstStrBlobPtr(*this); }
ConstStrBlobPtr StrBlob::end() {
    auto ret = ConstStrBlobPtr(*this, data->size());
    return ret;
}

StrBlob::StrBlob(): data(std::make_shared<std::vector<std::string>>()) {};
StrBlob::StrBlob(std::initializer_list<std::string> i1): data(std::make_shared<std::vector<std::string>>(i1)) {};
StrBlob::StrBlob(const StrBlob& sb) { data = std::make_shared<std::vector<std::string>>(*sb.data); }
StrBlob& StrBlob::operator=(const StrBlob& sb) { data = std::make_shared<std::vector<std::string>>(*sb.data); return *this; }

void StrBlob::check(size_type i, const std::string &msg) const {
    if (i >= data->size())
        throw std::out_of_range(msg);
}

std::string& StrBlob::front() {
    check(0, "front on empty StrBlob");
    return data->front();
}

std::string& StrBlob::back() {
    check(0, "back on empty StrBlob");
    return data->back();
}

const std::string& StrBlob::front() const {
    check(0, "front on empty StrBlob");
    return data->front();
}

const std::string& StrBlob::back() const {
    check(0, "back on empty StrBlob");
    return data->back();
}

void StrBlob::pop_back() {
    check(0, "pop_back on empty StrBlob");
    return data->pop_back();
}

std::shared_ptr<std::vector<std::string>> ConstStrBlobPtr::check(std::size_t i, const std::string& msg) const {
    auto ret = wptr.lock();
    if (!ret)
        throw std::runtime_error("unbound StrBlobPtr");
    if (i >= ret->size())
        throw std::out_of_range(msg);
    return ret;
}

std::string& ConstStrBlobPtr::deref() const {
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
    return std::lexicographical_compare(lhs.data->begin(), lhs.data->end(), rhs.data->begin(), rhs.data->end());
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
```

### Q31

使用默认的即可。

### Q32

```c++
#ifndef STRBLOB_H
#define STRBLOB_H

#include <memory>
#include <initializer_list>
#include <vector>
#include <string>
#include <stdexcept>

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
    typedef std::vector<std::string>::size_type size_type;
    StrBlob();
    StrBlob(std::initializer_list<std::string> i1);
    StrBlob(const StrBlob&);
    StrBlob& operator=(const StrBlob&);
    std::string& operator[](size_t n) { return (*data)[n]; }
    const std::string& operator[](size_t n) const { return (*data)[n]; }
    size_type size() const { return data->size(); }
    bool empty() const { return data->empty(); }
    void push_back(const std::string &t) { data->push_back(t); }
	void push_back(std::string &&t) { data->push_back(std::move(t)); }
    void pop_back();
    std::string& front();
    std::string& back();
    const std::string& front() const;
    const std::string& back() const;
	ConstStrBlobPtr begin();
	ConstStrBlobPtr end();
private:
    std::shared_ptr<std::vector<std::string>> data;
    void check(size_type i, const std::string &msg) const;
};

class ConstStrBlobPtr {
public:
    friend bool operator==(const ConstStrBlobPtr&, const ConstStrBlobPtr&);
    friend bool operator!=(const ConstStrBlobPtr&, const ConstStrBlobPtr&);
    ConstStrBlobPtr() : curr(0) {}
    ConstStrBlobPtr(const StrBlob &a, size_t sz = 0) : wptr(a.data), curr(sz) {}
    std::string& deref() const;
    ConstStrBlobPtr& incr();
    ConstStrBlobPtr& operator++();
    ConstStrBlobPtr& operator--();
    ConstStrBlobPtr operator++(int);
    ConstStrBlobPtr operator--(int);
    const std::string& operator*() const {
        auto p = check(curr, "dereference past end");
        return (*p)[curr];
    }
    const std::string* operator->() const {
        return & this->operator*();
    }
private:
    std::shared_ptr<std::vector<std::string>> check(std::size_t, const std::string&) const;
    std::weak_ptr<std::vector<std::string>> wptr;
    std::size_t curr;
};

class ConstStrBlobPtrPtr {
public:
    const std::string* operator->() const {
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

StrBlob::StrBlob(): data(std::make_shared<std::vector<std::string>>()) {};
StrBlob::StrBlob(std::initializer_list<std::string> i1): data(std::make_shared<std::vector<std::string>>(i1)) {};
StrBlob::StrBlob(const StrBlob& sb) { data = std::make_shared<std::vector<std::string>>(*sb.data); }
StrBlob& StrBlob::operator=(const StrBlob& sb) { data = std::make_shared<std::vector<std::string>>(*sb.data); return *this; }

void StrBlob::check(size_type i, const std::string &msg) const {
    if (i >= data->size())
        throw std::out_of_range(msg);
}

std::string& StrBlob::front() {
    check(0, "front on empty StrBlob");
    return data->front();
}

std::string& StrBlob::back() {
    check(0, "back on empty StrBlob");
    return data->back();
}

const std::string& StrBlob::front() const {
    check(0, "front on empty StrBlob");
    return data->front();
}

const std::string& StrBlob::back() const {
    check(0, "back on empty StrBlob");
    return data->back();
}

void StrBlob::pop_back() {
    check(0, "pop_back on empty StrBlob");
    return data->pop_back();
}

std::shared_ptr<std::vector<std::string>> ConstStrBlobPtr::check(std::size_t i, const std::string& msg) const {
    auto ret = wptr.lock();
    if (!ret)
        throw std::runtime_error("unbound StrBlobPtr");
    if (i >= ret->size())
        throw std::out_of_range(msg);
    return ret;
}

std::string& ConstStrBlobPtr::deref() const {
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
    return std::lexicographical_compare(lhs.data->begin(), lhs.data->end(), rhs.data->begin(), rhs.data->end());
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
```

### Q33

重载运算符函数的参数数量和该运算符作用的运算对象的数量一样多。 函数调用运算符最多可以传256个参数，因此在重载时，最多也能接受256个参数用作运算。

### Q34

```c++
#include <iostream>
#include <string>

struct if_then_else {
    std::string operator() (const bool a, const std::string b, const std::string c) {
        return a ? b : c;
    }
};

int main() {
    if_then_else obj;
    std::cout << obj(true, "a", "b") << std::endl;
    std::cout << obj(false, "a", "b") << std::endl;
    return 0;
}
```

### Q35

```c++
#include <iostream>
#include <string>

struct ReadString {
public:
    ReadString(std::istream &i = std::cin) : is(i) {}
    const std::string& operator() () {
        is >> str;
        return str;
    }
private:
    std::istream &is;
    std::string str;
};

int main() {
    ReadString obj;
    std::cout << obj() << std::endl;
    return 0;
}
```

### Q36

```c++
#include <iostream>
#include <vector>
#include <string>

struct ReadString {
public:
    ReadString(std::istream &i = std::cin) : is(i) {}
    const std::string operator() () {
        std::string s;
        getline(is, s);
        return s;
    }
private:
    std::istream &is;
};

int main() {
    ReadString obj;
    std::vector<std::string> v;
    std::string s;
    do {
        s = obj();
        v.push_back(s);
    } while (!s.empty());
    for (const auto &i : v) {
        std::cout << i << std::endl;
    }
    return 0;
}
```

### Q37

```c++
#include <iostream>
#include <algorithm>
#include <vector>

class Equal {
public:
    Equal(int i) : num(i) {}
    bool operator() (const int n) { return num == n; }
private:
    int num;
};

int main() {
    std::vector<int> v = {1,1,2,3,5};
    std::replace_if(v.begin(), v.end(), Equal(1), 5);
    for (const auto &i : v) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

### Q38

```c++
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>

class CompareString {
public:
    CompareString(size_t n) : sz(n) {}
    bool operator() (const std::string &s) const {
        return s.size() == sz;
    }
private:
    size_t sz;
};

int main() {
    std::ifstream ifs("./data/14-38");
    std::vector<std::string> v;
    std::string s;
    while (ifs >> s) {
        v.push_back(s);
    }
    for (size_t i = 0; i != 11; ++i) {
        size_t n = 0;
        for (auto iter = v.begin(); iter != v.end(); ) {
            iter = find_if(iter, v.end(), CompareString(i));
            if (iter != v.end()) {
                ++n;
                ++iter;
            }
        }
        std::cout << "length" << i << ":" << n << std::endl;
    }
    return 0;
}
```

### Q39

```c++
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>

class CompareString1 {
public:
    CompareString1(size_t n) : sz(n) {}
    bool operator() (const std::string &s) const {
        return s.size() < sz;
    }
private:
    size_t sz;
};

class CompareString2 {
public:
    CompareString2(size_t n) : sz(n) {}
    bool operator() (const std::string &s) const {
        return s.size() >= sz;
    }
private:
    size_t sz;
};


int main() {
    std::ifstream ifs("./data/14-39");
    std::vector<std::string> v;
    std::string s;
    while (ifs >> s) {
        v.push_back(s);
    }

    size_t n = 0;
    for (auto iter = v.begin(); iter != v.end(); ) {
        iter = find_if(iter, v.end(), CompareString1(10));
        if (iter != v.end()) {
            ++n;
            ++iter;
        }
    }
    std::cout << "length1~9" << ":" << n << std::endl;
    

    n = 0;
    for (auto iter = v.begin(); iter != v.end(); ) {
        iter = find_if(iter, v.end(), CompareString2(10));
        if (iter != v.end()) {
            ++n;
            ++iter;
        }
    }
    std::cout << "length>10" << ":" << n << std::endl;
    return 0;
}
```

### Q40

```c++
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

class CompareString {
public:
    bool operator() (const std::string a, const std::string b) { return a.size() < b.size(); }
};

class BiggerString {
public:
    BiggerString(std::vector<std::string>::size_type n) : sz(n) {}
    bool operator() (const std::string &s) { return s.size() >= sz; }
private:
    std::vector<std::string>::size_type sz;
};

class PrintString {
public:
    PrintString(std::ostream &o = std::cout) : os(o) {}
    void operator() (const std::string s) { os << s << " "; }
private:
    std::ostream &os;
};

void elimDups(std::vector<std::string> &);
void biggies(std::vector<std::string> &, std::vector<std::string>::size_type);

int main() {
    std::vector<std::string> words;
    std::string s;
    while (std::cin >> s) {
        words.push_back(s);
    }
    biggies(words, 4);
    return 0;
}

void elimDups(std::vector<std::string> &words) {
    sort(words.begin(), words.end());
    auto end_unique = unique(words.begin(), words.end());
    words.erase(end_unique, words.end());
}

void biggies(std::vector<std::string> &words, std::vector<std::string>::size_type sz) {
    elimDups(words);
    stable_sort(words.begin(), words.end(), CompareString());
    auto wc = find_if(words.begin(), words.end(), BiggerString(sz));
    for_each(wc, words.end(), PrintString());
}
```

### Q41

当传递的函数实现简单且使用次数少，用lambda，反之用类

### Q42

```c++
#include <iostream>
#include <algorithm>
#include <functional>
#include <vector>
#include <string>

int main() {
    std::vector<int> v1{1,2,4,8,16,32,64,128,256,512,1024,2048};
    std::vector<std::string> v2{"pooh", "pooh", "test1", "pooh", "test2"};
    std::vector<int> v3{1,2,4,8,16};
    std::vector<int> v4(v3.size());
    std::cout << std::count_if(v1.begin(), v1.end(),
                               std::bind(std::greater<int>(), std::placeholders::_1, 1024))
              << std::endl;
    std::cout << *(std::find_if(v2.begin(), v2.end(),
                                std::bind(std::not_equal_to<std::string>(), std::placeholders::_1, "pooh")))
              << std::endl;
    std::transform(v3.begin(), v3.end(), v4.begin(), std::bind(std::multiplies<int>(), std::placeholders::_1, 2));
    std::for_each(v4.begin(), v4.end(), [](const int a) { std::cout << a << " "; });
    std::cout << std::endl;
    return 0;
}
```

### Q43

```c++
#include <iostream>
#include <algorithm>
#include <functional>
#include <vector>

int main() {
    std::vector<int> v{2,4,8};
    int input;
    std::cin >> input;
    std::modulus<int> mod;
    std::cout << (std::find_if(v.begin(), v.end(), [&mod, input](const int &i) { return mod(input,i); }) == v.end()) << std::endl;
    return 0;
}
```

### Q44

```c++
#include <iostream>
#include <functional>
#include <map>
#include <string>

int add(int, int);

int main() {
    auto mod = [](int i, int j) { return i % j; };
    struct divide {
        int operator() (int denominator, int divisor) {
            return denominator / divisor;
        }
    };
    std::map<std::string, std::function<int(int,int)>> binops = {
        {"+", add},
        {"-", std::minus<int>()},
        {"*", [](int i, int j) { return i*j; }},
        {"/", divide()},
        {"%", mod}
    };
    std::cout << binops["+"](10,5) << std::endl;
    std::cout << binops["-"](10,5) << std::endl;
    std::cout << binops["*"](10,5) << std::endl;
    std::cout << binops["/"](10,5) << std::endl;
    std::cout << binops["%"](10,5) << std::endl;
    return 0;
}

int add(int i, int j) {
    return i+j;
}
```

### Q45

```c++
// bookNo
// revenue/units_sold

#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <string>

struct Sales_data {

    friend std::istream& operator>>(std::istream&, Sales_data&);
    friend std::ostream& operator<<(std::ostream&, const Sales_data&);
    friend Sales_data operator+(const Sales_data&, const Sales_data&);

public:
    Sales_data(std::string s, unsigned n, double p) :
                bookNo(s), units_sold(n), revenue(p*n) {};
    Sales_data() : Sales_data("", 0, 0) {}
    Sales_data(std::string s) : Sales_data(s, 0, 0) {}
    Sales_data(std::istream &is) : Sales_data() { is >> *this; }
    std::string isbn() const { return bookNo; }
    Sales_data& operator+=(const Sales_data&);
    Sales_data& operator=(const std::string&);
    explicit operator std::string() const { return bookNo; }
    explicit operator double() const { return avg_price(); }

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

Sales_data& Sales_data::operator+=(const Sales_data &rhs) {
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}

Sales_data& Sales_data::operator=(const std::string &s) {
    *this = Sales_data(s);
    return *this;
}

#endif
```

### Q46

不应该，类类型和转换类型之间不存在明显的映射关系
应该，防止异常情况发生

### Q47

无意义，编译器会忽略掉
不能改变对象

### Q48

应该，应该，为防止异常情况发生

```c++
#ifndef BOOK_H
#define BOOK_H

#include <string>

class Book {

    friend std::istream& operator>>(std::istream&, Book&);
    friend std::ostream& operator<<(std::ostream&, const Book&);
    friend bool operator==(const Book&, const Book&);
    friend bool operator!=(const Book&, const Book&);
    friend bool operator<(const Book&, const Book&);
    friend bool operator>(const Book&, const Book&);
    friend bool operator<=(const Book&, const Book&);
    friend bool operator>=(const Book&, const Book&);

public:
    Book() = default;
    Book(unsigned int a, std::string b, std::string c) : 
        no(a), name(b), author(c) {}
    Book(std::istream &is) { is >> *this; }
    Book& operator=(const Book &);
    Book& operator=(Book&&) noexcept;
    explicit operator bool() const { return no; }

private:
    unsigned int no;
    std::string name;
    std::string author;

};

std::istream& operator>>(std::istream &is, Book &book) {
    is >> book.no >> book.name >> book.author;
    if (!is) {
        book = Book();
    }
    return is;
}

std::ostream& operator<<(std::ostream &os, const Book &book) {
    os << book.no << " " << book.name << " " << book.author;
    return os;
}

bool operator==(const Book &book1, const Book &book2) {
    return book1.no == book2.no;
}

bool operator!=(const Book &book1, const Book &book2) {
    return !(book1 == book2);
}

bool operator<(const Book &book1, const Book &book2) {
    return book1.no < book2.no;
}

bool operator>(const Book &book1, const Book &book2) {
    return book2 < book1;
}

bool operator<=(const Book &book1, const Book &book2) {
    return !(book2 < book1);
}

bool operator>=(const Book &book1, const Book &book2) {
    return !(book1 < book2);
}

Book& Book::operator=(const Book &book) {
    no = book.no;
    name = book.name;
    author = book.author;
    return *this;
}

Book& Book::operator=(Book &&book) noexcept {
    if (this != &book) {
        no = std::move(book.no);
        name = std::move(book.name);
        author = std::move(book.author);
    }
    return *this;
}
#endif
```

### Q49

见Q48

### Q50

不合法，存在二义性
合法

### Q51

void calc(int)
算术类型转换优先于类类型转换

### Q52

```c++
二义性
double operator+(int,double) SmallInt->int,LongDouble->double
float operator+(int,float) SmallInt->int,LongDouble->float

精确匹配
LongDouble operator(const SmallInt&)
double operator+(double, int) LongDouble->double,SmallInt->int
float operator+(float, int) LongDouble->float,SmallInt->int
```

### Q53

不合法

```c++
#include <iostream>

class SmallInt {
    friend SmallInt operator+(const SmallInt &s1, const SmallInt &s2) {
        SmallInt sum(s1.val+s2.val);
        return sum;
    };
public:
    SmallInt(int i = 0) : val(i) {}
    operator int() const { return val; }
private:
    std::size_t val;
};


int main() {
    SmallInt s1;
    double d1 = s1 + SmallInt(3.14);
    // double d2 = s1.operator int() + 3.14;
    double d2 = static_cast<int>(s1) + 3.14;
    return 0;
}
```