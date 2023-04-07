# Chapter 19

### Q1

```c++
void *operator new(size_t size) {
    if (void *mem = malloc(size)) {
        return mem;
    }
    else {
        throw std::bad_alloc();
    }
}
void operator delete(void *mem) noexcept { free(mem); }
```

### Q2

```c++
// StrVec.h
#ifndef STRVEC_H
#define STRVEC_H

#include <utility>
#include <memory>
#include <algorithm>
#include <string>
#include <initializer_list>

void *operator new(size_t size) {
    if (void *mem = malloc(size)) {
        return mem;
    }
    else {
        throw std::bad_alloc();
    }
}
void operator delete(void *mem) noexcept { free(mem); }

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
        std::cout << i << std::endl;
    }
    return 0;
}
```

### Q3

成功
失败，pb是指向B类型对象的指针，不能转换为指向C类型对象的指针
失败，A *pa = new D具有二义性

### Q4

```c++
#include <iostream>
#include <typeinfo>

class A {
public:
    virtual ~A() = default;
};

class B : public A {
};

class C : public B {
};

int main() {
    A *pa = new C;
    try {
        C &c = dynamic_cast<C&>(*pa);
    } catch(std::bad_cast &e) {
        std::cout << e.what() << std::endl;
    }
    return 0;
}
```

### Q5

当想使用基类对象的指针或引用执行某个派生类操作并且该操作不是虚函数

### Q6

Query_base为抽象虚类，AndQuery的构造函数为private。借助上题中的类进行验证

```c++
#include <iostream>
#include <typeinfo>

class A {
public:
    virtual ~A() = default;
};

class B : public A {
};

class C : public B {
};

int main() {
    A *pa1 = new C();
    A *pa2 = new C();
    if (typeid(*pa1) == typeid(*pa2)) {
        std::cout << "same type" << std::endl;
    }
    if (C *c1 = dynamic_cast<C*>(pa1)) {
        if (typeid(*pa1) == typeid(*c1)) {
            std::cout << "success" << std::endl;
        }
    }
    try {
        C &c2 = dynamic_cast<C&>(*pa2);
        if (typeid(*pa2) == typeid(c2)) {
            std::cout << "success" << std::endl;
        }
    } catch(std::bad_cast &e) {
        std::cout << e.what() << std::endl;
    }
    return 0;
}
```

### Q7

见Q6

### Q8

见Q6

### Q9

```c++
#include <iostream>
#include <string>

struct Sales_data {
};
struct Base {
};
struct Derived : public Base {
};

int main() {
    int arr[10];
    Derived d;
    Base *p = &d;
    std::cout << typeid(42).name() << ", "
              << typeid(arr).name() << ", "
              << typeid(Sales_data).name() << ", "
              << typeid(std::string).name() << ", "
              << typeid(p).name() << ", "
              << typeid(*p).name() << std::endl;
    return 0;
}
```

### Q10

P1A
P1A
1B

### Q11

普通的数据指针指向一个对象；类成员指针指向类的非静态成员。当初始化这样一个指针时，我们令其指向类的某个成员，但是不指定该成员所属的对象；直到使用成员指针时，才提供所属的对象。

### Q12

```c++
#include <iostream>
#include <string>

class Screen {
public:
    typedef std::string::size_type pos;
    Screen() = default;
    Screen(pos ht, pos wd, char c): height(ht), width(wd), contents(ht*wd, c) {}
    char get_cursor() const { return contents[cursor]; }
    char get(pos ht, pos wd) const { return contents[ht*width+wd]; }
    static const pos Screen::*data() { return &Screen::cursor; }
private:
    pos cursor = 0;
    pos height = 0, width = 0;
    std::string contents;
};

int main() {
    const Screen::pos Screen::*pdata = Screen::data();
    Screen myScreen(2,2,'a');
    auto s = myScreen.*pdata;
    std::cout << s << std::endl;
    return 0;
}
```

### Q13

```c++
const std::string Sales_data::*pdata;
```

### Q14

```c++
错误，不能把char (Screen::*)(Screen::pos, Screen::pos) const转换成char (Screen::*)() const
```

### Q15

和普通函数指针不同的是，在成员函数和指向该成员的指针之间不存在自动转换规则。

### Q16

```c++
using AvgPrice = double (Sales_data::*)() const;
AvgPrice avgprice = &Sales_data::avg_price;
```

### Q17

```c++
#include <iostream>
#include <string>

class Screen {
public:
    typedef std::string::size_type pos;
    Screen() = default;
    Screen(pos ht, pos wd, char c): height(ht), width(wd), contents(ht*wd, c) {}
    char get() const { return contents[cursor]; }
    char get(pos ht, pos wd) const { return contents[ht*width+wd]; }
    static const pos Screen::*data() { return &Screen::cursor; }
private:
    pos cursor = 0;
    pos height = 0, width = 0;
    std::string contents;
};

int main() {
    Screen myScreen(2,2,'a');
    using Get1 = char (Screen::*)() const;
    using Get2 = char (Screen::*)(Screen::pos, Screen::pos) const;
    Get1 get1 = &Screen::get;
    Get2 get2 = &Screen::get;
    std::cout << (myScreen.*get1)() << std::endl;
    std::cout << (myScreen.*get2)(0,0) << std::endl;
    return 0;
}
```

### Q18

```c++
#include <iostream>
#include <algorithm>
#include <functional>
#include <vector>
#include <string>

int main() {
    std::vector<std::string> v{"a", "b", "", "", "c"};
    // 方式一
    std::function<bool (const std::string&)> fcn = &std::string::empty;
    std::cout << std::count_if(v.begin(), v.end(), fcn) << std::endl;
    // 方式二
    std::cout << std::count_if(v.begin(), v.end(), std::mem_fn(&std::string::empty)) << std::endl;
    // 方式三
    std::cout << std::count_if(v.begin(), v.end(), std::bind(&std::string::empty, std::placeholders::_1)) << std::endl;
    return 0;
}
```

### Q19

```c++
// Sales_data.h
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <algorithm>
#include <functional>
#include <vector>
#include <string>
#include <stdexcept>
#include <exception>

struct Sales_data {

    friend std::vector<Sales_data>::const_iterator find_first(const std::vector<Sales_data>&, double);
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

class isbn_mismatch: public std::logic_error {
public:
    explicit isbn_mismatch(const std::string &s) : std::logic_error(s) {}
    isbn_mismatch(const std::string &s, const std::string &lhs, const std::string &rhs) : 
        std::logic_error(s), left(lhs), right(rhs) {}
    const std::string left, right;
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
    if (isbn() != rhs.isbn())
        throw isbn_mismatch("wrong isbns", isbn(), rhs.isbn());
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

std::vector<Sales_data>::const_iterator find_first(const std::vector<Sales_data> &v, double d) {
    auto fun = std::bind(&Sales_data::avg_price, std::placeholders::_1);
    return find_if(v.cbegin(), v.cend(), [&](const Sales_data &s) { return fun(s) > d; });
}

#endif
```

```c++
#include <iostream>
#include <vector>
#include "Sales_data.h"

int main() {
    std::vector<Sales_data> v;
    Sales_data s;
    while (std::cin >> s) {
        v.push_back(s);
    }
    std::cout << *(find_first(v, 2.0)) << std::endl;
    return 0;
}
```

### Q20

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

class TextQuery {
public:
    class QueryResult;
    using line_no = std::vector<std::string>::size_type;
    TextQuery(std::ifstream&);
    QueryResult query(const std::string&) const;
private:
    static std::string cleanup_str(const std::string&);
    std::shared_ptr<std::vector<std::string>> file;
    std::map<std::string, std::shared_ptr<std::set<line_no>>> wm;
};

class TextQuery::QueryResult {
    friend std::ostream& print(std::ostream&, const QueryResult&);
public:
    QueryResult(std::string s,
                std::shared_ptr<std::set<TextQuery::line_no>> p,
                std::shared_ptr<std::vector<std::string>> f) :
        sought(s), lines(p), file(f) {}
private:
    std::string sought;
    std::shared_ptr<std::set<TextQuery::line_no>> lines;
    std::shared_ptr<std::vector<std::string>> file;
};

TextQuery::TextQuery(std::ifstream &is) : file(new std::vector<std::string>) {
    std::string text;
    while (getline(is, text)) {
        file->push_back(text);
        int n = file->size() - 1;
        std::istringstream line(text);
        std::string word;
        while (line >> word) {
            word = cleanup_str(word);
            auto &lines = wm[word];
            if (!lines)
                lines.reset(new std::set<line_no>);
            lines->insert(n);
        }
    }
}

TextQuery::QueryResult TextQuery::query(const std::string &sought) const {
    static std::shared_ptr<std::set<line_no>> nodata(new std::set<line_no>);
    auto loc = wm.find(sought);
    if (loc == wm.end())
        return QueryResult(sought, nodata, file);
    else
        return QueryResult(sought, loc->second, file);
}

std::string make_plural(size_t ctr, const std::string &word, const std::string &ending) {
    return (ctr > 1) ? word + ending : word;
}

std::ostream &print(std::ostream & os, const TextQuery::QueryResult &qr) {
    os << qr.sought << " occurs " << qr.lines->size() << " "
        << make_plural(qr.lines->size(), "times", "s") << std::endl;
    for (auto num : *qr.lines)
        os << "\t(line " << num+1 << ") " << *(qr.file->begin()+num) << std::endl;
    return os;
}

std::string TextQuery::cleanup_str(const std::string &word) {
    std::string ret;
    for (std::string::const_iterator it = word.begin(); it != word.end(); ++it) {
        if (!ispunct(*it))
            ret += tolower(*it);
    }
    return ret;
}

#endif
```

```c++
#include <iostream>
#include <fstream>
#include "TextQuery.h"

void runQueries(std::ifstream &);

int main() {
    std::ifstream in("./data/19-20");
    runQueries(in);
    return 0;
}

void runQueries(std::ifstream &infile) {
    TextQuery tq(infile);
    do {
        std::cout << "enter word to look for, or q to quit: ";
        std::string s;
        if (!(std::cin >> s) || s == "q") break;
        print(std::cout, tq.query(s)) << std::endl;
    } while (true);
}
```

### Q21

```c++
// Sales_data.h
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <iostream>
#include <algorithm>
#include <functional>
#include <vector>
#include <string>
#include <stdexcept>
#include <exception>

struct Sales_data {

    friend std::vector<Sales_data>::const_iterator find_first(const std::vector<Sales_data>&, double);
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

class isbn_mismatch: public std::logic_error {
public:
    explicit isbn_mismatch(const std::string &s) : std::logic_error(s) {}
    isbn_mismatch(const std::string &s, const std::string &lhs, const std::string &rhs) : 
        std::logic_error(s), left(lhs), right(rhs) {}
    const std::string left, right;
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
    if (isbn() != rhs.isbn())
        throw isbn_mismatch("wrong isbns", isbn(), rhs.isbn());
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

std::vector<Sales_data>::const_iterator find_first(const std::vector<Sales_data> &v, double d) {
    auto fun = std::bind(&Sales_data::avg_price, std::placeholders::_1);
    return find_if(v.cbegin(), v.cend(), [&](const Sales_data &s) { return fun(s) > d; });
}

#endif
```

```c++
// Token.h
#ifndef TOKEN_H
#define TOKEN_H

#include <string>
#include "Sales_data.h"

class Token {
public:
    Token(): tok(INT), ival(0) {}
    Token(const Token &t): tok(t.tok) { copyUnion(t); }
    Token &operator=(const Token&);
    Token(Token&&) noexcept;
    Token &operator=(Token&&) noexcept;
    ~Token() {
        if(tok == STR) sval.std::string::~string();
        if(tok == SD) sdval.Sales_data::~Sales_data();
    }
    Token &operator=(const std::string&);
    Token &operator=(char);
    Token &operator=(int);
    Token &operator=(double);
    Token &operator=(const Sales_data&);
private:
    enum {INT, CHAR, DBL, STR, SD} tok;
    union {
        char cval;
        int ival;
        double dval;
        std::string sval;
        Sales_data sdval;
    };
    void copyUnion(const Token&);
};

Token::Token(Token &&t) noexcept : tok(t.tok) {
    switch (tok) {
        case INT: ival=t.ival; break;
        case CHAR: cval=t.cval; break;
        case DBL: dval=t.dval; break;
        case STR: sval = std::move(t.sval); break;
        case SD: sdval = std::move(t.sdval); break;
    }
}

Token &Token::operator=(Token &&t) noexcept {
    tok = t.tok;
    switch (tok) {
        case INT: ival=t.ival; break;
        case CHAR: cval=t.cval; break;
        case DBL: dval=t.dval; break;
        case STR: sval = std::move(t.sval); break;
        case SD: sdval = std::move(t.sdval); break;
    }
    return *this;
}

Token &Token::operator=(int i) {
    if (tok == STR) sval.std::string::~string();
    if (tok == SD) sdval.Sales_data::~Sales_data();
    ival = i;
    tok = INT;
    return *this;
}

Token &Token::operator=(char c) {
    if (tok == STR) sval.std::string::~string();
    if (tok == SD) sdval.Sales_data::~Sales_data();
    cval = c;
    tok = CHAR;
    return *this;
}


Token &Token::operator=(double d) {
    if (tok == STR) sval.std::string::~string();
    if (tok == SD) sdval.Sales_data::~Sales_data();
    dval = d;
    tok = DBL;
    return *this;
}

Token &Token::operator=(const std::string &s) {
    if (tok == SD) sdval.Sales_data::~Sales_data();
    else if (tok == STR) sval = s;
    else new(&sval) std::string(s);
    tok = STR;
    return *this;
}

Token &Token::operator=(const Sales_data &sd) {
    if (tok == STR) sval.std::string::~string();
    else if (tok == SD) sdval = sd;
    else new(&sdval) Sales_data(sd);
    tok = SD;
    return *this;
}


void Token::copyUnion(const Token &t) {
    switch (t.tok) {
        case INT: ival=t.ival; break;
        case CHAR: cval=t.cval; break;
        case DBL: dval=t.dval; break;
        case STR: new(&sval) std::string(t.sval); break;
        case SD: new(&sdval) Sales_data(t.sdval); break;
    }
}

Token &Token::operator=(const Token &t) {
    if (tok == STR && t.tok != STR) sval.std::string::~string();
    else if (tok == SD && t.tok != SD) sdval.Sales_data::~Sales_data();
    else if (tok == STR && t.tok == STR) sval = t.sval;
    else if (tok == SD && t.tok == SD) sdval = t.sdval;
    else copyUnion(t);
    tok = t.tok;
    return *this;
}

#endif
```

### Q22

见Q21

### Q23

见Q21

### Q24

见Q21

### Q25

见Q21

### Q26

不合法，C语言不支持函数重载