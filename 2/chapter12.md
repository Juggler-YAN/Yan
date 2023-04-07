# Chapter 12

### Q1

b1：4个元素
b2：销毁

### Q2

```c++
// StrBlob.h
#ifndef STRBLOB_H
#define STRBLOB_H

#include <memory>
#include <initializer_list>
#include <vector>
#include <string>
#include <stdexcept>

using namespace std;

class StrBlob {
public:
    typedef vector<string>::size_type size_type;
    StrBlob();
    StrBlob(initializer_list<string> i1);
    size_type size() const { return data->size(); }
    bool empty() const { return data->empty(); }
    void push_back(const string &t) { data->push_back(t); }
    void pop_back();
    string& front();
    string& back();
    const string& front() const;
    const string& back() const;
private:
    shared_ptr<vector<string>> data;
    void check(size_type i, const string &msg) const;
};

StrBlob::StrBlob(): data(make_shared<vector<string>>()) {};
StrBlob::StrBlob(initializer_list<string> i1): data(make_shared<vector<string>>(i1)) {};

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

#endif
```

### Q3

不需要，push_back和pop_back会改变对象的内容

### Q4

size_type类型本身就是>=0的

### Q5

1. 优点：清楚地知道使用的是哪种类型；
2. 缺点：需要显式初始化。

### Q6

```c++
#include <iostream>
#include <vector>

using namespace std;

vector<int> * create();
void push(vector<int> *);
void print(vector<int> *);

int main() {
    vector<int> * pv;
    pv = create();
    push(pv);
    print(pv);
    delete pv;
    return 0;
}

vector<int> * create() {
    return new vector<int>;
}

void push(vector<int> * pv) {
    int i;
    while (cin >> i) {
        pv->push_back(i);
    }   
}

void print(vector<int> * pv) {
    for (const auto &i : (*pv)) {
        cout << i << " ";
    }
    cout << endl;
}
```

### Q7

```c++
#include <iostream>
#include <memory>
#include <vector>

using namespace std;

shared_ptr<vector<int>> create();
void push(shared_ptr<vector<int>>);
void print(shared_ptr<vector<int>>);

int main() {
    shared_ptr<vector<int>> pv;
    pv = create();
    push(pv);
    print(pv);
    return 0;
}

shared_ptr<vector<int>> create() {
    return make_shared<vector<int>>();
}

void push(shared_ptr<vector<int>> pv) {
    int i;
    while (cin >> i) {
        pv->push_back(i);
    }   
}

void print(shared_ptr<vector<int>> pv) {
    for (const auto &i : (*pv)) {
        cout << i << " ";
    }
    cout << endl;
}
```

### Q8

指针p被转换成bool值，new动态分配的内存没有释放

### Q9

r所指的内存没有释放，应该修改为delete r; r = q;

### Q10

正确

### Q11

process运行结束时，p指向的内存会被释放，再次使用p指针时会报错。

### Q12

1. 合法，智能指针
2. 不合法，内置指针转换成智能指针必须使用直接初始化
3. 不合法，内置指针转换成智能指针必须使用直接初始化
4. 合法，使用直接初始化将内置指针转换成智能指针

### Q13

sp和p指向相同的内存，p已经被销毁，等程序结束又会自动销毁sp，此时内存被第二次delete，报错

### Q14

```c++
#include <iostream>
#include <memory>
#include <string>

using namespace std;

struct destination {
    string des;
    destination(string des_) : des(des_) {}
};

struct connection {
    string conn;
    connection(string conn_) : conn(conn_) {}
};

connection connect(destination *);
void end_connection(connection *);
void disconnect(connection);
void f(destination &);

int main() {
    destination d("test");
    f(d);
    return 0;
}

connection connect(destination *des_) {
    cout << "connect to " << des_->des << endl;
    return connection(des_->des);
}

void end_connection(connection *p) {
    disconnect(*p);
}

void disconnect(connection conn_) {
    cout << "disconnect " << conn_.conn << endl;
}

void f(destination &d) {
    connection c = connect(&d);
    shared_ptr<connection> p(&c, end_connection);
    cout << "connecting now(" << p.use_count() << ")" << endl;
}
```

### Q15

```c++
#include <iostream>
#include <memory>
#include <string>

using namespace std;

struct destination {
    string des;
    destination(string des_) : des(des_) {}
};

struct connection {
    string conn;
    connection(string conn_) : conn(conn_) {}
};

connection connect(destination *);
void disconnect(connection);
void f(destination &);

int main() {
    destination d("test");
    f(d);
    return 0;
}

connection connect(destination *des_) {
    cout << "connect to " << des_->des << endl;
    return connection(des_->des);
}

void disconnect(connection conn_) {
    cout << "disconnect " << conn_.conn << endl;
}

void f(destination &d) {
    connection c = connect(&d);
    shared_ptr<connection> p(&c, [](connection *p) { disconnect(*p); });
    cout << "connecting now(" << p.use_count() << ")" << endl;
}
```

### Q16

```c++
#include <iostream>
#include <memory>

using namespace std;

int main() {
	unique_ptr<int> up1(new int(1));
	// unique_ptr<int> up2(up1);
    // unique_ptr<int> up3 = up1;
	unique_ptr<int> up4;
	up4.reset(up1.release());
	cout << *up4 << endl;

	return 0;
}
```

### Q17

1. 非法，初始化错误；
2. 编译时合法，运行时会报错，销毁pi时使用默认的delete会出错；
3. 编译时合法，但是当unique_ptr释放空间时，pi2指针会成为空悬指针；
4. 编译时合法，运行时会报错，销毁pi时使用默认的delete会出错；
5. 合法；
6. 编译时合法，但是可能会出现两次delete释放相同内存或者一个delete后另一个变为空悬指针。

### Q18

多个shared_ptr可以指向同一内存，可以直接赋值

### Q19

```c++
// StrBlob.h

#ifndef STRBLOB_H
#define STRBLOB_H

using namespace std;

#include <memory>
#include <initializer_list>
#include <vector>
#include <string>
#include <stdexcept>

class StrBlobPtr;

class StrBlob {
public:
    friend class StrBlobPtr;
    typedef vector<string>::size_type size_type;
    StrBlob();
    StrBlob(initializer_list<string> i1);
    size_type size() const { return data->size(); }
    bool empty() const { return data->empty(); }
    void push_back(const string &t) { data->push_back(t); }
    void pop_back();
    string& front();
    string& back();
    const string& front() const;
    const string& back() const;
	StrBlobPtr begin();
	StrBlobPtr end();
private:
    shared_ptr<vector<string>> data;
    void check(size_type i, const string &msg) const;
};

class StrBlobPtr {
public:
    StrBlobPtr() : curr(0) {}
    StrBlobPtr(StrBlob &a, size_t sz = 0) : wptr(a.data), curr(sz) {}
    string& deref() const;
    StrBlobPtr& incr();
private:
    shared_ptr<vector<string>> check(size_t, const string&) const;
    weak_ptr<vector<string>> wptr;
    size_t curr;
};

StrBlobPtr StrBlob::begin() { return StrBlobPtr(*this); }
StrBlobPtr StrBlob::end() {
    auto ret = StrBlobPtr(*this, data->size());
    return ret;
}

StrBlob::StrBlob(): data(make_shared<vector<string>>()) {};
StrBlob::StrBlob(initializer_list<string> i1): data(make_shared<vector<string>>(i1)) {};

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

shared_ptr<vector<string>> StrBlobPtr::check(size_t i, const string& msg) const {
    auto ret = wptr.lock();
    if (!ret)
        throw runtime_error("unbound StrBlobPtr");
    if (i >= ret->size())
        throw out_of_range(msg);
    return ret;
}

string& StrBlobPtr::deref() const {
    auto p = check(curr, "dereference past end");
    return (*p)[curr];
}

StrBlobPtr& StrBlobPtr::incr() {
    check(curr, "increment past end of StrBlobPtr");
    ++curr;
    return *this;
}

#endif
```

### Q20

```
To My Wife: With a Copy of My Poems
I can write no stately proem
As a prelude to my lay;
From a poet to a poem
I would dare to say.

For if of these fallen petals
Once to you seem fair,
Love will waft it till it settles
On your hair.

And when wind and winter harden
All the loveless land,
It will whisper of the garden,
You will understand.
```

```c++
// StrBlob.h
#ifndef STRBLOB_H
#define STRBLOB_H

#include <memory>
#include <initializer_list>
#include <vector>
#include <string>
#include <stdexcept>

using namespace std;

class StrBlobPtr;

class StrBlob {
public:
    friend class StrBlobPtr;
    typedef vector<string>::size_type size_type;
    StrBlob();
    StrBlob(initializer_list<string> i1);
    size_type size() const { return data->size(); }
    bool empty() const { return data->empty(); }
    void push_back(const string &t) { data->push_back(t); }
    void pop_back();
    string& front();
    string& back();
    const string& front() const;
    const string& back() const;
	StrBlobPtr begin();
	StrBlobPtr end();
private:
    shared_ptr<vector<string>> data;
    void check(size_type i, const string &msg) const;
};

class StrBlobPtr {
public:
    StrBlobPtr() : curr(0) {}
    StrBlobPtr(StrBlob &a, size_t sz = 0) : wptr(a.data), curr(sz) {}
	bool operator!=(const StrBlobPtr& p) { return p.curr != curr; }
    string& deref() const;
    StrBlobPtr& incr();
private:
    shared_ptr<vector<string>> check(size_t, const string&) const;
    weak_ptr<vector<string>> wptr;
    size_t curr;
};

StrBlobPtr StrBlob::begin() { return StrBlobPtr(*this); }
StrBlobPtr StrBlob::end() {
    auto ret = StrBlobPtr(*this, data->size());
    return ret;
}

StrBlob::StrBlob(): data(make_shared<vector<string>>()) {};
StrBlob::StrBlob(initializer_list<string> i1): data(make_shared<vector<string>>(i1)) {};

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

shared_ptr<vector<string>> StrBlobPtr::check(size_t i, const string& msg) const {
    auto ret = wptr.lock();
    if (!ret)
        throw runtime_error("unbound StrBlobPtr");
    if (i >= ret->size())
        throw out_of_range(msg);
    return ret;
}

string& StrBlobPtr::deref() const {
    auto p = check(curr, "dereference past end");
    return (*p)[curr];
}

StrBlobPtr& StrBlobPtr::incr() {
    check(curr, "increment past end of StrBlobPtr");
    ++curr;
    return *this;
}

#endif
```

```c++
#include <iostream>
#include <fstream>
#include <string>
#include "StrBlob.h"

using namespace std;

int main() {
    StrBlob sb;
    ifstream in("./test.txt");
    string s;
    while (getline(in ,s)) {
        sb.push_back(s);
    }
    StrBlobPtr psb(sb);
    for (auto beg = sb.begin(); beg != sb.end(); beg.incr()) {
        cout << beg.deref() << endl;
    }
    return 0;
}
```

### Q21

上一个版本，易读性更强

### Q22

```c++
// StrBlob.h

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
public:
    friend class ConstStrBlobPtr;
    typedef vector<string>::size_type size_type;
    StrBlob();
    StrBlob(initializer_list<string> i1);
    size_type size() const { return data->size(); }
    bool empty() const { return data->empty(); }
    void push_back(const string &t) { data->push_back(t); }
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
    ConstStrBlobPtr() : curr(0) {}
    ConstStrBlobPtr(const StrBlob &a, size_t sz = 0) : wptr(a.data), curr(sz) {}
    string& deref() const;
    ConstStrBlobPtr& incr();
private:
    shared_ptr<vector<string>> check(size_t, const string&) const;
    weak_ptr<vector<string>> wptr;
    size_t curr;
};

ConstStrBlobPtr StrBlob::begin() { return ConstStrBlobPtr(*this); }
ConstStrBlobPtr StrBlob::end() {
    auto ret = ConstStrBlobPtr(*this, data->size());
    return ret;
}

StrBlob::StrBlob(): data(make_shared<vector<string>>()) {};
StrBlob::StrBlob(initializer_list<string> i1): data(make_shared<vector<string>>(i1)) {};

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

#endif
```

### Q23

```c++
#include <iostream>
#include <cstring>

using namespace std;

int main() {
    const char * s1 = "hello", * s2 = "world";
    char * s = new char[strlen(s1)+strlen(s2)+1]();
    strcat(s, s1);
    strcat(s, s2);
    cout << s << endl;
    delete [] s;
    return 0;
}
```

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
    const string s1 = "hello", s2 = "world";
    string *s = new string;
    *s += s1 + s2;
    cout << *s << endl;
    delete s;
    return 0;
}
```

### Q24

```c++
#include <iostream>

using namespace std;

int main() {
    int size{0};
    cin >> size;
    char * input = new char[size+1]();
    cin.ignore();
    cin.get(input, size+1);
    cout << input;
    delete [] input;
    return 0;
}
```

### Q25

```c++
delete [] pa;
```

### Q26

```c++
#include <iostream>
#include <memory>
#include <string>

using namespace std;

int main() {
    const int n = 10;
    allocator<string> alloc;
    auto const p = alloc.allocate(n);
    auto q = p;
    alloc.construct(q++);
    alloc.construct(q++,10,'c');
    alloc.construct(q++,"hi");
    while (q != p) {
        cout << *(--q) << endl;
        alloc.destroy(q);
    }
    alloc.deallocate(p, n);
    return 0;
}
```

### Q27

```c++
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

ostream &print(ostream & os, const QueryResult &qr) {
    os << qr.sought << " occurs " << qr.lines->size() << " "
        << make_plural(qr.lines->size(), "times", "s") << endl;
    for (auto num : *qr.lines)
        os << "\t(line " << num+1 << ") " << *(qr.file->begin()+num) << endl;
    return os;
}

#endif
```

### Q28

```
Alice Emma has long flowing red hair. 
Her Daddy says when the wind blows 
through her hair, it looks almost alive, 
like a fiery bird in flight. 
A beautiful fiery bird, he tells her, 
magical but untamed. 
"Daddy, shush, there is no such thing," 
she tells him, at the same time wanting 
him to tell her more.
Shyly, she asks, "I mean, Daddy, is there?"
```

```c++
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <set>

using namespace std;

string make_plural(size_t, const string &, const string &);

int main() {
    string filename;
    cout << "Enter a filename: ";
    cin >> filename;
    ifstream in(filename);
    vector<string> lines;
    string s;
    while (getline(in, s)) {
        lines.push_back(s);
    }
    map<string, set<int>> m;
    string word;
    while ((cin >> word) && (word != "q")) {
        for (decltype(lines.size()) i = 0; i != lines.size(); ++i) {
            if (lines[i].find(word) != string::npos) {
                m[word].insert(i);
            }
        }
        cout << word << " occurs " << m[word].size() << " "
            << make_plural(m[word].size(), "times", "s") << endl;
        for (auto num : m[word])
            cout << "\t(line " << num+1 << ") " << lines[num] << endl;
    }
    return 0;
}

string make_plural(size_t ctr, const string &word, const string &ending) {
    return (ctr > 1) ? word + ending : word;
}
```

### Q29

```
Alice Emma has long flowing red hair. 
Her Daddy says when the wind blows 
through her hair, it looks almost alive, 
like a fiery bird in flight. 
A beautiful fiery bird, he tells her, 
magical but untamed. 
"Daddy, shush, there is no such thing," 
she tells him, at the same time wanting 
him to tell her more.
Shyly, she asks, "I mean, Daddy, is there?"
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

using namespace std;

void runQueries(ifstream &);

int main() {
    ifstream in("./test.txt");
    runQueries(in);
    return 0;
}

void runQueries(ifstream &infile) {
    TextQuery tq(infile);
    do {
        cout << "enter word to look for, or q to quit: ";
        string s;
        if (!(cin >> s) || s == "q") break;
        print(cout, tq.query(s)) << endl;
    } while (true);
}
```

### Q30

```
Alice Emma has long flowing red hair. 
Her Daddy says when the wind blows 
through her hair, it looks almost alive, 
like a fiery bird in flight. 
A beautiful fiery bird, he tells her, 
magical but untamed. 
"Daddy, shush, there is no such thing," 
she tells him, at the same time wanting 
him to tell her more.
Shyly, she asks, "I mean, Daddy, is there?"
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

using namespace std;

void runQueries(ifstream &);

int main() {
    ifstream in("./test.txt");
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

### Q31

vector不能确保元素是唯一的，所以这里set更好。

### Q32

```
Alice Emma has long flowing red hair. 
Her Daddy says when the wind blows 
through her hair, it looks almost alive, 
like a fiery bird in flight. 
A beautiful fiery bird, he tells her, 
magical but untamed. 
"Daddy, shush, there is no such thing," 
she tells him, at the same time wanting 
him to tell her more.
Shyly, she asks, "I mean, Daddy, is there?"
```

```c++
// StrBlob.h
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
public:
    friend class ConstStrBlobPtr;
    typedef vector<string>::size_type size_type;
    StrBlob();
    StrBlob(initializer_list<string> i1);
    size_type size() const { return data->size(); }
    bool empty() const { return data->empty(); }
    void push_back(const string &t) { data->push_back(t); }
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
    ConstStrBlobPtr() : curr(0) {}
    ConstStrBlobPtr(const StrBlob &a, size_t sz = 0) : wptr(a.data), curr(sz) {}
    string& deref() const;
    ConstStrBlobPtr& incr();
private:
    shared_ptr<vector<string>> check(size_t, const string&) const;
    weak_ptr<vector<string>> wptr;
    size_t curr;
};

ConstStrBlobPtr StrBlob::begin() { return ConstStrBlobPtr(*this); }
ConstStrBlobPtr StrBlob::end() {
    auto ret = ConstStrBlobPtr(*this, data->size());
    return ret;
}

StrBlob::StrBlob(): data(make_shared<vector<string>>()) {};
StrBlob::StrBlob(initializer_list<string> i1): data(make_shared<vector<string>>(i1)) {};

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
#include "StrBlob.h"

using namespace std;

class QueryResult;
class TextQuery {
public:
    using line_no = StrBlob::size_type;
    TextQuery(ifstream&);
    QueryResult query(const string&) const;
private:
    StrBlob file;
    map<string, shared_ptr<set<line_no>>> wm;
};

class QueryResult {
friend ostream& print(ostream&, const QueryResult&);
public:
    QueryResult(string s,
                shared_ptr<set<TextQuery::line_no>> p,
                StrBlob f) :
        sought(s), lines(p), file(f) {}
private:
    string sought;
    shared_ptr<set<TextQuery::line_no>> lines;
    StrBlob file;
};

TextQuery::TextQuery(ifstream &is) {
    string text;
    while (getline(is, text)) {
        file.push_back(text);
        int n = file.size() - 1;
        istringstream line(text);
        string word;
        while (line >> word) {
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

ostream &print(ostream & os, const QueryResult &qr) {
    os << qr.sought << " occurs " << qr.lines->size() << " "
        << make_plural(qr.lines->size(), "times", "s") << endl;
    for (auto num : *qr.lines) {
        ConstStrBlobPtr p(qr.file, num);
        os << "\t(line " << num+1 << ") " << p.deref() << endl;
    }
    return os;
}

#endif
```

```c++
#include <iostream>
#include <fstream>
#include "TextQuery.h"

using namespace std;

void runQueries(ifstream &);

int main() {
    ifstream in("./test.txt");
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

### Q33

```c++
// StrBlob.h
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
public:
    friend class ConstStrBlobPtr;
    typedef vector<string>::size_type size_type;
    StrBlob();
    StrBlob(initializer_list<string> i1);
    size_type size() const { return data->size(); }
    bool empty() const { return data->empty(); }
    void push_back(const string &t) { data->push_back(t); }
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
    ConstStrBlobPtr() : curr(0) {}
    ConstStrBlobPtr(const StrBlob &a, size_t sz = 0) : wptr(a.data), curr(sz) {}
    string& deref() const;
    ConstStrBlobPtr& incr();
private:
    shared_ptr<vector<string>> check(size_t, const string&) const;
    weak_ptr<vector<string>> wptr;
    size_t curr;
};

ConstStrBlobPtr StrBlob::begin() { return ConstStrBlobPtr(*this); }
ConstStrBlobPtr StrBlob::end() {
    auto ret = ConstStrBlobPtr(*this, data->size());
    return ret;
}

StrBlob::StrBlob(): data(make_shared<vector<string>>()) {};
StrBlob::StrBlob(initializer_list<string> i1): data(make_shared<vector<string>>(i1)) {};

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
#include "StrBlob.h"

using namespace std;

class QueryResult;
class TextQuery {
public:
    using line_no = StrBlob::size_type;
    TextQuery(ifstream&);
    QueryResult query(const string&) const;
private:
    StrBlob file;
    map<string, shared_ptr<set<line_no>>> wm;
};

class QueryResult {
friend ostream& print(ostream&, const QueryResult&);
public:
    QueryResult(string s,
                shared_ptr<set<TextQuery::line_no>> p,
                StrBlob f) :
        sought(s), lines(p), file(f) {}
    set<StrBlob::size_type>::iterator begin() const { return lines->begin(); }
    set<StrBlob::size_type>::iterator end() const { return lines->end(); }
    shared_ptr<StrBlob> get_file() const { return make_shared<StrBlob>(file); }
private:
    string sought;
    shared_ptr<set<TextQuery::line_no>> lines;
    StrBlob file;
};

TextQuery::TextQuery(ifstream &is) {
    string text;
    while (getline(is, text)) {
        file.push_back(text);
        int n = file.size() - 1;
        istringstream line(text);
        string word;
        while (line >> word) {
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

ostream &print(ostream & os, const QueryResult &qr) {
    os << qr.sought << " occurs " << qr.lines->size() << " "
        << make_plural(qr.lines->size(), "times", "s") << endl;
    for (auto num : *qr.lines) {
        ConstStrBlobPtr p(qr.file, num);
        os << "\t(line " << num+1 << ") " << p.deref() << endl;
    }
    return os;
}

#endif
```