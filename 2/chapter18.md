### Q1

range_error
expection
抛出指向局部对象的指针，错误

### Q2

指针p未被释放

### Q3

```c++
1.智能指针
shared_ptr<int> p(new int[v.size()])
2.使用类控制资源的分配
class P {
public:
    P() = default;
    P(size_t n) : p(new int[n]()) {}
    ~P() { delete[] p; }
private:
    int *p = nullptr;
}
```

### Q4

派生类异常的处理代码应出现在基类异常的处理代码之前

```c++
try {
	//使用C++标准库
} catch(overflow_error eobj) {
	//...
} catch(const runtime_error &re) {
	//...
} catch(exception) {
	//...
}
```

### Q5

```c++
#include <iostream>
#include <stdexcept>
#include <exception>
#include <cstdlib>
#include <typeinfo>

using namespace std;

int main() {
	try {
		//使用C++标准库
	} catch(bad_cast &r) {
		cout << r.what();
		abort();
	} catch(range_error &r) {
		cout << r.what();
		abort();
	} catch(underflow_error &r) {
		cout << r.what();
		abort();
	} catch(overflow_error &r) {
		cout << r.what();
		abort();
	} catch(runtime_error &r) {
		cout << r.what();
		abort();
	} catch(length_error &r) {
		cout << r.what();
		abort();
	} catch(out_of_range &r) {
		cout << r.what();
		abort();
	} catch(invalid_argument &r) {
		cout << r.what();
		abort();
	} catch(domain_error &r) {
		cout << r.what();
		abort();
	} catch(logic_error &r) {
		cout << r.what();
		abort();
	} catch(bad_alloc &r) {
		cout << r.what();
		abort();
	} catch(exception &r) {
		cout << r.what();
		abort();
    }
	return 0;
}
```

### Q6

```c++
（a）
exceptionType *p;
throw p;
（b）
可以捕获所有异常
（c）
EXCPTYPE a;
throw a;
```

### Q7

```c++
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
```

### Q8

略

### Q9

```c++
// Sales_data.h
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <string>
#include <stdexcept>
#include <exception>

using namespace std;

struct Sales_data {

    friend istream& operator>>(istream&, Sales_data&);
    friend ostream& operator<<(ostream&, const Sales_data&);
    friend Sales_data operator+(const Sales_data&, const Sales_data&);
    friend bool operator==(const Sales_data&, const Sales_data&);
	friend class hash<Sales_data>;

public:
    Sales_data(string s, unsigned n, double p) :
                bookNo(s), units_sold(n), revenue(p*n) {};
    Sales_data() : Sales_data("", 0, 0) {}
    Sales_data(string s) : Sales_data(s, 0, 0) {}
    Sales_data(istream &is) : Sales_data() { is >> *this; }
    string isbn() const { return bookNo; }
    Sales_data& operator+=(const Sales_data&);

private:
    inline double avg_price() const;
    string bookNo;
    unsigned units_sold = 0;
    double revenue = 0.0;

};

class isbn_mismatch: public logic_error {
public:
    explicit isbn_mismatch(const string &s) : logic_error(s) {}
    isbn_mismatch(const string &s, const string &lhs, const string &rhs) : 
        logic_error(s), left(lhs), right(rhs) {}
    const string left, right;
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

istream& operator>>(istream &is, Sales_data &item) {
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

ostream& operator<<(ostream &os, const Sales_data &item) {
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
#include "Sales_data.h"

int main() {
    Sales_data item1, item2, sum;
    while (cin >> item1 >> item2) {
        try {
            sum = item1 + item2;
            cout << sum << endl;
        }
        catch (const isbn_mismatch &e) {
            cerr << e.what() << ": left isbn(" << e.left
                << ") right isbn(" << e.right << ")" << endl;
        }
    }
    return 0;
}
```

### Q10

见Q9

### Q11

如果what函数抛出异常，需要try catch捕获，然后会再调用what函数，无限循环，直达内存耗尽。

### Q12

```c++
// Query.h
#ifndef QUERY_H
#define QUERY_H

#include <iostream>
#include <memory>
#include <set>
#include <string>
#include "TextQuery.h"

using namespace std;

namespace chapter15 {
    class Query_base {
        friend class Query;
    protected:
        using line_no = chapter10::TextQuery::line_no;
        virtual ~Query_base() = default;
    private:
        virtual chapter10::QueryResult eval(const chapter10::TextQuery&) const = 0;
        virtual string rep() const = 0;
    };

    class Query {
        friend Query operator~(const Query&);
        friend Query operator|(const Query&, const Query&);
        friend Query operator&(const Query&, const Query&);
    public:
        Query(const string&);
        chapter10::QueryResult eval(const chapter10::TextQuery &t) const {
            cout << "Query::eval()" << endl;
            return q->eval(t);
        }
        string rep() const { 
            cout << "Query::rep()" << endl;
            return q->rep();
        }
    private:
        Query(shared_ptr<Query_base> query): q(query) {
            cout << "Query(shared_ptr<Query_base>)" << endl;
        }
        shared_ptr<Query_base> q;
    };
    inline ostream& operator<<(ostream &os, const Query &query) {
        return os << query.rep();
    }

    class WordQuery : public Query_base {
        friend class Query;
        WordQuery(const string &s) : query_word(s) {
            cout << "WordQuery(const string &)" << endl;
        }
        chapter10::QueryResult eval(const chapter10::TextQuery &t) const {
            cout << "WordQuery::eval()" << endl;
            return t.query(query_word);
        }
        string rep() const {
            cout << "WordQuery::rep()" << endl;
            return query_word;
        }
        string query_word;
    };
    inline Query::Query(const string &s) : q(new WordQuery(s)) {
        cout << "Query(const string &)" << endl;
    }

    class NotQuery : public Query_base {
        friend Query operator~(const Query&);
        NotQuery(const Query &q) : query(q) {
            cout << "NotQuery(const Query &)" << endl;
        }
        string rep() const {
            cout << "NotQuery::rep()" << endl;
            return "~(" + query.rep() + ")";
        }
        chapter10::QueryResult eval(const chapter10::TextQuery&) const;
        Query query;
    };
    inline Query operator~(const Query &operand) {
        return shared_ptr<Query_base>(new NotQuery(operand));
    }
    chapter10::QueryResult NotQuery::eval(const chapter10::TextQuery &text) const {
        cout << "NotQuery::eval()" << endl;
        auto result = query.eval(text);
        auto ret_lines = make_shared<set<line_no>>();
        auto beg = result.begin(), end = result.end();
        auto sz = result.get_file()->size();
        for (size_t n = 0; n != sz; ++n) {
            if (beg == end || *beg != n) {
                ret_lines->insert(n);
            }
            else if (beg != end) {
                ++beg;
            }
        }
        return chapter10::QueryResult(rep(), ret_lines, result.get_file());
    }

    class BinaryQuery : public Query_base {
    protected:
        BinaryQuery(const Query &l, const Query &r, string s) : lhs(l), rhs(r), opSym(s) {
            cout << "BinaryQuery(const Query &, const Query &, string)" << endl;
        }
        string rep() const {
            cout << "BinaryQuery::rep()" << endl;
            return "(" + lhs.rep() + " " + opSym + " " + rhs.rep() + ")";
        }
        Query lhs, rhs;
        string opSym;
    };

    class AndQuery : public BinaryQuery {
        friend Query operator&(const Query&, const Query&);
        AndQuery(const Query &left, const Query &right) : BinaryQuery(left, right, "&") {
            cout << "AndQuery(const Query &, const Query &, string)" << endl;
        }
        chapter10::QueryResult eval(const chapter10::TextQuery&) const;
    };
    inline Query operator&(const Query &lhs, const Query &rhs) {
        return shared_ptr<Query_base>(new AndQuery(lhs, rhs));
    }
    chapter10::QueryResult AndQuery::eval(const chapter10::TextQuery &text) const {
        cout << "AndQuery::eval()" << endl;
        auto right = rhs.eval(text), left = lhs.eval(text);
        auto ret_lines = make_shared<set<line_no>>();
        set_intersection(left.begin(), left.end(), right.begin(), right.end(), inserter(*ret_lines, ret_lines->begin()));
        return chapter10::QueryResult(rep(), ret_lines, left.get_file());
    }

    class OrQuery : public BinaryQuery {
        friend Query operator|(const Query&, const Query&);
        OrQuery(const Query &left, const Query &right) : BinaryQuery(left, right, "|") {
            cout << "OrQuery(const Query &, const Query &, string)" << endl;
        }
        chapter10::QueryResult eval(const chapter10::TextQuery&) const;
    };
    inline Query operator|(const Query &lhs, const Query &rhs) {
        return shared_ptr<Query_base>(new OrQuery(lhs, rhs));
    }
    chapter10::QueryResult OrQuery::eval(const chapter10::TextQuery &text) const {
        cout << "OrQuery::eval()" << endl;
        auto right = rhs.eval(text), left = lhs.eval(text);
        auto ret_lines = make_shared<set<line_no>>(left.begin(), left.end());
        ret_lines->insert(right.begin(), right.end());
        return chapter10::QueryResult(rep(), ret_lines, left.get_file());
    }
}

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

using namespace std;

namespace chapter10 {
    class QueryResult;
    class TextQuery {
    public:
        using line_no = vector<string>::size_type;
        TextQuery(ifstream&);
        QueryResult query(const string&) const;
    private:
        static string cleanup_str(const string&);
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
        auto begin() const { return lines->cbegin(); }
        auto end() const { return lines->cend(); }
        auto get_file() const { return file; }
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
                word = cleanup_str(word);
                auto &lines = wm[word];
                if (!lines)
                    lines.reset(new set<line_no>);
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

    ostream& print(ostream &os, const QueryResult &qr) {
        os << qr.sought << " occurs " << qr.lines->size() << " "
            << make_plural(qr.lines->size(), "times", "s") << endl;
        for (auto num : *qr.lines)
            os << "\t(line " << num+1 << ") " << *(qr.file->begin()+num) << endl;
        return os;
    }

    string TextQuery::cleanup_str(const string &word) {
        string ret;
        for (string::const_iterator it = word.begin(); it != word.end(); ++it) {
            if (!ispunct(*it))
                ret += tolower(*it);
        }
        return ret;
    }
}

#endif
```

```c++
#include <iostream>
#include "Query.h"
#include "TextQuery.h"

int main() {
    return 0;
}
```

### Q13

需要在其所在的文件可见，在其所在的文件外不可见

### Q14

```c++
mathLib::MatrixLib::matrix mathLib::MatrixLib::operator*(const matrix&, const matrix&);
```

### Q15

对于using声明来说，我们只是简单地令名字在局部作用域有效，using指示是令整个命名空间的所有内容变得有效。

### Q16

1. ivar重复定义
2. dvar重复定义
3. ivar二义性冲突
4. ivar二义性冲突

### Q17

```c++
namespace Exercise {
    int ivar = 0;
    double dvar = 0;
    const int limit = 1000;   
}
int ivar = 0;

// using Exercise::ivar;
// using Exercise::dvar;
// using Exercise::limit;

// using namespace Exercise;

void manip() {
    // using Exercise::ivar;
    // using Exercise::dvar;
    // using Exercise::limit;

    // using namespace Exercise;

    double dvar = 3.1416;
    int iobj = limit + 1;
    ++ivar;
    ++::ivar;
}
```

### Q18

使用string版本的swap；使用实例化为int的swap。
首先在当前作用域寻找合适的函数，接着查找输出语句的外层作用域，最后因为形参是类类型，所以还会查找string类所属的命名空间，发现了string特定版本的swap函数。
当前作用域寻找到std::swap

### Q19

使用标准库的swap，如果v1.mem1和v2.mem1为用户自定义类型，将无法调用针对该类型特定的swap。

### Q20

```c++
候选函数
void compute()
void compute(int)
void compute(double, double = 3.4)
void compute(char*, char* = 0)
void compute(const void *)
可行函数
void compute(int)（最佳匹配）
void compute(double, double = 3.4)（int->double）
void compute(char*, char* = 0)（0->nullptr）
void compute(const void *)（0->nullptr）

候选函数
void compute()
void compute(const void *)
可行函数
void compute(const void *)（0->nullptr）（最佳匹配）
```

### Q21

```c++
CADVehicle公开继承CAD，私有继承Vehicle
有，重复继承
iostream公开继承istream和ostream
```

### Q22

A、B、C、X、Y、Z、MI

### Q23

允许
允许
允许
允许

### Q24

pe->print() 正确：Panda::print()；
pe->toes() 错误，不属于ZooAnimal的接口；
pe->cuddle() 错误，不属于ZooAnimal的接口；
pe->highlight() 错误，不属于ZooAnimal的接口；
delete pe 正确：Panda::~Panda()。

### Q25

MI::print()
MI::print()
MI::print()
MI::~MI()
MI::~MI()
MI::~MI()

### Q26

没有匹配的print函数

```c++
#include <iostream>
#include <vector>

using namespace std;

struct Base1 {
	void print(int) const;
protected:
	int ival;
	double dval;
	char cval;
private:
	int *id;
};
struct Base2 {
	void print(double) const;
protected:
	double fval;
private:
	double dval;
};
struct Derived : public Base1 {
	void print(string) const;
protected:
	string sval;
	double dval;
};
struct MI : public Derived, public Base2 {
	void print(vector<double>);
    void print(int);
protected:
	int *ival;
	vector<double> dvec;
};

void MI::print(int i) {
    cout << i << endl;
}

int main() {
    MI mi;
    mi.print(42);
    return 0;
}
```

### Q27

```c++
（a）
Base1：ival、dval、cval、print
Base2：fval、print
Derived：sval、dval、print
MI：ival、dvec、print、foo
（b）
存在，ival、dval、print
（c）
dval = Base1::dval + Derived::dval;
（d）
Base2::fval = dvec.back();
（e）
sval.at(0) = Base1::cval;
```

### Q28

```c++
无需限定符
Derived1::bar（派生类的bar比共享虚基类的bar优先级更高）；
Derived2::ival（派生类的ival比共享虚基类的ival优先级更高）；
需要限定符
foo（Derived1和Derived2均存在该成员）
cval（Derived1和Derived2均存在该成员）
```

### Q29

```c++
（a）
构造函数执行顺序Class、Base、D1、D2、MI、Class、Final，析构函数与之相反
（b）
一个Base两个Class
（c）
错误；错误；错误；正确
```

### Q30

```c++
#include <iostream>

using namespace std;

class Class {
};

class Base : public Class {
public:
    Base() { cout << "Base()" << endl; }
    Base(int i) : val(i) { cout << "Base(int)" << endl; }
    Base(const Base &b) : val(b.val) { cout << "Base(const Base&)" << endl; }
private:
    int val;
};

class D1 : virtual public Base {
public:
    D1() = default;
    D1(int i) : Base(i) { cout << "D1(int)" << endl; }
    D1(const D1 &d) : Base(d) { cout << "D1(const D1&)" << endl; }
};

class D2 : virtual public Base {
public:
    D2() = default;
    D2(int i) : Base(i) { cout << "D2(int)" << endl; }
    D2(const D2 &d) : Base(d) { cout << "D2(const D2&)" << endl; }
};

class MI : public D1, public D2 {
public:
    MI() = default;
    MI(int i) : D1(i), D2(i) { cout << "MI(int)" << endl; }
    MI(const MI &m) : D1(m), D2(m) { cout << "MI(const MI&)" << endl; }
};

// class Final : public MI, public Class {
// };

int main() {
    return 0;
}
```