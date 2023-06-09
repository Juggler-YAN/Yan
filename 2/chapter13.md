### Q1

如果一个构造函数的第一个参数是自身类类型的引用，且任何额外的参数都有默认值，则次构造函数是拷贝构造函数。
1. 用=定义变量
2. 将一个对象作为实参传递给一个非引用类型的形参
3. 从一个返回类型为非引用类型的函数返回一个对象
4. 用花括号列表初始化一个数组中的元素或一个聚合类的成员

### Q2

调用永远也不会成功，为了调用拷贝构造函数，我们需要拷贝它的实参，但为了拷贝实参，我们又需要调用拷贝构造函数，如此无限循环。

### Q3

拷贝StrBlob时shared_ptr会+1，拷贝StrBlobPtr时不会。

### Q4

```c++
Point global;
Point foo_bar(Point arg) //1：函数参数
{
    Point local = arg, *heap = new Point(global); //2,3：拷贝初始化
    *heap = local;
    Point pa[ 4 ] = { local, *heap }; //4,5：列表初始化
    return *heap; //6：返回值
}
```

### Q5

```c++
class HasPtr {
public:
    HasPtr(const string &s = string()) : ps(new string(s)), i(0) { }
    HasPtr(const HasPtr& hp) : ps(new string(*hp.ps)), i(hp.i) { }
private:
    string *ps;
    int i;
};
```

### Q6

1. 拷贝赋值运算符是一个名为operator=的函数，它接受一个与其所在类相同类型的参数；
2. 当赋值发生时使用该运算符；
3. 将右侧运算对象的每个非static成员赋予左侧运算对象的对应成员，对于数组类型的成员，逐个赋值数组元素，合成拷贝赋值运算符返回一个指向其左侧运算对象的引用；
4. 如果一个类未定义自己的拷贝赋值运算符。

### Q7

赋值StrBlob时shared_ptr会+1，赋值StrBlobPtr时不会。

### Q8

```c++
class HasPtr {
public:
    HasPtr(const string &s = string()) : ps(new string(s)), i(0) { }
    HasPtr(const HasPtr& hp) : ps(new string(*hp.ps)), i(hp.i) { }
    HasPtr& operator=(const HasPtr& hp) {
        if (this != &hp) {
            string *temp = new string(*hp.ps);
            delete ps;
            ps = temp;
            i = hp.i;
        }
        return *this;
    }
private:
    string *ps;
    int i;
};
```

### Q9

1. 析构函数是类的一个成员函数，名字由波浪号接类名构成，它没有返回值，也不接受参数，用于释放对象所使用的资源，并销毁对象的非static数据成员；
2. 类似拷贝构造函数和拷贝赋值运算符，对于某些类，和合成析构函数被用来阻止该类型的对象被销毁，如果不是这种情况，合成析构函数的函数体就为空；
3. 当一个类未定义自己的析构函数时，编译器会为它定义一个合成析构函数。

### Q10

StrBlob对象销毁时，shared_ptr-1，直到为0时，对象将销毁；StrBlobPtr对象销毁时，其指向的对象不会被销毁。

### Q11

```c++
class HasPtr {
public:
    HasPtr(const string &s = string()) : ps(new string(s)), i(0) {}
    HasPtr(const HasPtr& hp) : ps(new string(*hp.ps)), i(hp.i) {}
    HasPtr& operator=(const HasPtr& hp) {
        if (this != &hp) {
            string *temp = new string(*hp.ps);
            delete ps;
            ps = temp;
            i = hp.i;
        }
        return *this;
    }
    ~HasPtr() { delete ps; }
private:
    string *ps;
    int i;
};
```

### Q12

函数运行结束后accum、item1和item2销毁。

### Q13

```c++
#include <iostream>
#include <string>
#include <vector>

using namespace std;

struct X {
	X() { cout << "X()" << endl; }
	X(const X&) { cout << "X(const X&)" << endl; }
	X& operator=(const X &x)
	{
		cout << "X& operator=(const X &x)" << endl;
		return *this;
	}
	~X() { cout << "~x()" << endl; }
};

void fun1(X x) {
	cout << "void fun1(X x)" << endl;
}

void fun2(X &x) {
	cout << "void fun2(X &x)" << endl;
}

int main() {
	X x1;
	fun1(x1);
	fun2(x1);
	X *x2 = new X();
	{
		vector<X> v;
		v.reserve(2);
		v.push_back(x1);
		v.push_back(*x2);
	}
	delete x2;
	return 0;
}
```

### Q14

输出结果相同

### Q15

会，拷贝初始化时会调用拷贝构造函数，生成一个新的序号，但调用函数时f又会生成一个新的序号，所以新的输出结果会输出不同于a、b、c的mysn值。

### Q16

会，拷贝初始化时会调用拷贝构造函数，生成一个新的序号，新的输出结果会是a、b、c的mysn值。

### Q17

```c++
#include <iostream>

using namespace std;

class numbered {
friend void f (numbered);
public:
    numbered() : mysn(rand()) {}
private:
    unsigned int mysn;
};

void f (numbered s) { cout << s.mysn << endl; }

int main() {
    numbered a, b = a, c = b;
    f(a);
    f(b);
    f(c);
    return 0;
}
```

```c++
#include <iostream>

class numbered {
friend void f (numbered);
public:
    numbered() : mysn(rand()) {}
    numbered(const numbered&) : mysn(rand()) {}
private:
    unsigned int mysn;
};

void f (numbered s) { cout << s.mysn << endl; }

int main() {
    numbered a, b = a, c = b;
    f(a);
    f(b);
    f(c);
    return 0;
}
```

```c++
#include <iostream>

using namespace std;

class numbered {
friend void f (const numbered&);
public:
    numbered() : mysn(rand()) {}
    numbered(const numbered&) : mysn(rand()) {}
private:
    unsigned int mysn;
};

void f (const numbered & s) { cout << s.mysn << endl; }

int main() {
    numbered a, b = a, c = b;
    f(a);
    f(b);
    f(c);
    return 0;
}
```

### Q18

```c++
#include <iostream>
#include <string>

using namespace std;

class Employee {
friend void print(const Employee &e);
public:
    Employee() { ID = n; n++; }
    Employee(const string &s) { name = s; ID = n; n++; }
private:
    string name;
    int ID;
    static int n;
};

void print(const Employee &e) {
    cout << e.name << " " << e.ID << endl;
}

int Employee::n = 0;

int main() {
    Employee a;
    Employee b("b");
    print(a);
    print(b);
    return 0;
}
```

### Q19

不需要，实际中员工是不可拷贝的

```c++
#include <iostream>
#include <string>

using namespace std;

class Employee {
friend void print(const Employee &e);
public:
    Employee() { ID = n; n++; }
    Employee(const string &s) { name = s; ID = n; n++; }
    Employee(const Employee&) = delete;
    Employee& operator=(const Employee&) = delete;
private:
    string name;
    int ID;
    static int n;
};

void print(const Employee &e) {
    cout << e.name << " " << e.ID << endl;
}

int Employee::n = 0;

int main() {
    Employee a;
    Employee b("b");
    print(a);
    print(b);
    return 0;
}
```

### Q20

使用编译器定义的合成拷贝函数，合成赋值运算符以及合成析构函数

### Q21

不需要，因为不需要定义析构函数，所以也不需要定义拷贝控制成员。

### Q22

```c++
class HasPtr {
public:
    HasPtr(const string &s = string()) : ps(new string(s)), i(0) {}
    HasPtr(const HasPtr& hp) : ps(new string(*hp.ps)), i(hp.i) {}
    HasPtr& operator=(const HasPtr& hp) {
        auto temp = new string(*hp.ps);
        delete ps;
        ps = temp;
        i = hp.i;
        return *this;
    }
    ~HasPtr() { delete ps; }
private:
    string *ps;
    int i;
};
```

### Q23

略

### Q24

如果未定义析构函数，将会发生内存泄漏，动态内存得不到释放，直到没有内存可以申请；如果未定义拷贝构造函数，指针将被复制，可能会多次释放同一个内存。

### Q25

拷贝构造函数：拷贝vector<string>而不是指针；拷贝赋值运算符：释放当前的vector<string>，并从右侧运算对象拷贝vector<string>；析构函数：智能指针会自行销毁

```c++
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
    StrBlob(const StrBlob&);
    StrBlob& operator=(const StrBlob&);
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

#endif
```

### Q26

见Q25

### Q27

```c++
class HasPtr {
public:
    HasPtr(const string &s = string()) : ps(new string(s)), i(0), use(new size_t(1)) {}
    HasPtr(const HasPtr& hp) : ps(hp.ps), i(hp.i), use(hp.use) { ++*use; }
    HasPtr& operator=(const HasPtr&);
    ~HasPtr();
private:
    string *ps;
    int i;
    size_t *use;
};

HasPtr& HasPtr::operator=(const HasPtr& hp) {
    ++*hp.use;
    if (--*use == 0) {
        delete ps;
        delete use;
    }
    ps = hp.ps;
    i = hp.i;
    use = hp.use;
    return *this;
}

HasPtr::~HasPtr() {
    if (--*use == 0) {
        delete ps;
        delete use;
    }
}
```

### Q28

```c++
class TreeNode {
public:
    TreeNode() : value(""), count(0), left(nullptr), right(nullptr) {}
    TreeNode(const TreeNode &tn) : value(tn.value), count(tn.count), left(tn.left), right(tn.right) { ++*count; }
    TreeNode& operator=(const TreeNode &);
    ~TreeNode();
private:
	string value;
	int *count;
	TreeNode *left;
	TreeNode *right;	
};
TreeNode& TreeNode::operator=(const TreeNode &tn) {
    ++*tn.count;
    if (--*count == 0) {
        if (left) {
            delete left;
        }
        if (right) {
            delete right;
        }
        delete count;
    }
    value = tn.value;
    count = tn.count;
    left = tn.left;
    right = tn.right;
    return *this;
}
TreeNode::~TreeNode() {
    if (--*count == 0) {
        if (left) {
            delete left;
        }
        if (right) {
            delete right;
        }
        delete count;
    }
}
```

```c++
class BinStrTree{
public:
    BinStrTree() : root(new TreeNode()) {}
    BinStrTree(const BinStrTree &bst) : root(new TreeNode(*bst.root)) {}
    BinStrTree& operator=(const BinStrTree &);
    ~BinStrTree() { delete root; }
private:
	TreeNode *root;	
};

BinStrTree& BinStrTree::operator=(const BinStrTree &bst) {
    auto newroot = new TreeNode(*bst.root);
    delete root;
    root = newroot;
    return *this;
}
```

### Q29

因为函数参数不同，所以调用的函数也不同

### Q30

```c++
// HasPtr.h
#ifndef HASPTR_H
#define HASPTR_H

#include <string>

using namespace std;

class HasPtr {
friend void swap(HasPtr&, HasPtr&);
public:
    HasPtr(const string &s = string()) : ps(new string(s)), i(0), use(new size_t(1)) {}
    HasPtr(const HasPtr& hp) : ps(hp.ps), i(hp.i), use(hp.use) { ++*use; }
    HasPtr& operator=(const HasPtr&);
    ~HasPtr();
private:
    string *ps;
    int i;
    size_t *use;
};

HasPtr& HasPtr::operator=(const HasPtr& hp) {
    ++*hp.use;
    if (--*use == 0) {
        delete ps;
        delete use;
    }
    ps = hp.ps;
    i = hp.i;
    use = hp.use;
    return *this;
}

HasPtr::~HasPtr() {
    if (--*use == 0) {
        delete ps;
        delete use;
    }
}

void swap(HasPtr &lhs, HasPtr &rhs) {
    using std::swap;
    swap(lhs.ps, rhs.ps);
    swap(lhs.i, rhs.i);
    swap(lhs.use, rhs.use);
    cout << "swap()" << endl;
}

#endif
```

```c++
#include <iostream>
#include "HasPtr.h"

int main() {
    HasPtr a("a"), b("b");
    swap(a, b);
    return 0;
}
```

### Q31

```c++
// HasPtr.h
#ifndef HASPTR_H
#define HASPTR_H

#include <string>

using namespace std;

class HasPtr {
friend void swap(HasPtr&, HasPtr&);
friend bool operator<(const HasPtr&, const HasPtr&);
public:
    HasPtr(const string &s = string()) : ps(new string(s)), i(0), use(new size_t(1)) {}
    HasPtr(const HasPtr& hp) : ps(hp.ps), i(hp.i), use(hp.use) { ++*use; }
    HasPtr& operator=(const HasPtr&);
    ~HasPtr();
private:
    string *ps;
    int i;
    size_t *use;
};

HasPtr& HasPtr::operator=(const HasPtr& hp) {
    ++*hp.use;
    if (--*use == 0) {
        delete ps;
        delete use;
    }
    ps = hp.ps;
    i = hp.i;
    use = hp.use;
    return *this;
}

bool operator<(const HasPtr &lhs, const HasPtr &rhs) {
    cout << "<" << endl;
    return *lhs.ps < *rhs.ps;
}

HasPtr::~HasPtr() {
    if (--*use == 0) {
        delete ps;
        delete use;
    }
}

void swap(HasPtr &lhs, HasPtr &rhs) {
    using std::swap;
    swap(lhs.ps, rhs.ps);
    swap(lhs.i, rhs.i);
    swap(lhs.use, rhs.use);
    cout << "swap()" << endl;
}

#endif
```

```c++
#include <iostream>
#include <algorithm>
#include <vector>
#include "HasPtr.h"

int main() {
    HasPtr a("b"), b("a");
    vector<HasPtr> v{a, b};
    sort(v.begin(), v.end());
    return 0;
}
```

### Q32

不会，类指针本身就是指针交换，没有内存分配。

### Q33

需要更改Folder的值

### Q34

```c++
#ifndef MESSAGE_H
#define MESSAGE_H

#include <string>
#include <set>

using namespace std;

class Message {
    friend class Folder;
    friend void swap(Message &, Message &);
public:
    explicit Message(const string &str = "") : contents(str) {}
    Message (const Message&);
    Message& operator=(const Message&);
    ~Message();
    void save(Folder&);
    void remove(Folder&);
private:
    string contents;
    set<Folder*> folders;
    void add_to_Folders(const Message&);
    void remove_from_Folders();
};

void Message::add_to_Folders(const Message &m) {
    for (auto f : m.folders) {
        f->addMsg(this);
    }
}

void Message::remove_from_Folders() {
    for (auto f : folders) {
        f->remMsg(this);
    }
}

void Message::save(Folder &f) {
    folders.insert(&f);
    f.addMsg(this);
}

void Message::remove(Folder &f) {
    folders.erase(&f);
    f.remMsg(this);
}

Message::Message(const Message &m) : contents(m.contents), folders(m.folders) {
    add_to_Folders(m);
}

Message& Message::operator=(const Message &rhs) {
    remove_from_Folders();
    contents = rhs.contents;
    folders = rhs.folders;
    add_to_Folders(rhs);
    return *this;
}

Message::~Message() {
    remove_from_Folders();
}

void swap(Message &lhs, Message &rhs) {
    using std::swap;
    for (auto f : lhs.folders) {
        f->remMsg(&lhs);
    }
    for (auto f : rhs.folders) {
        f->remMsg(&rhs);
    }
    swap(lhs.contents, rhs.contents);
    swap(lhs.folders, rhs.folders);
    for (auto f : lhs.folders) {
        f->addMsg(&lhs);
    }
    for (auto f : rhs.folders) {
        f->addMsg(&rhs);
    }
}

#endif
```

### Q35

不会把消息添加到指向该消息的目录中，Message中保存的Folder信息与Folder中保存的Message信息不统一。

### Q36

```c++
#ifndef MESSAGE_H
#define MESSAGE_H

#include <string>
#include <set>

using namespace std;

class Folder;

class Message {
    friend class Folder;
    friend void swap(Message &, Message &);
public:
    explicit Message(const string &str = "") : contents(str) {}
    Message (const Message&);
    Message& operator=(const Message&);
    ~Message();
    void save(Folder&);
    void remove(Folder&);
private:
    string contents;
    set<Folder*> folders;
    void add_to_Folders(const Message&);
    void remove_from_Folders();
	void addFldr(Folder *f) { folders.insert(f); }
	void remFldr(Folder *f) { folders.erase(f); }
};

class Folder {
    friend class Message;
    friend void swap(Message &, Message &);
public:
    Folder() = default;
    Folder(const Folder&);
    Folder& operator=(const Folder&);
    ~Folder();
private:
    set<Message*> messages;
    void add_to_Messages(const Folder&);
    void remove_from_Messages();
    void addMsg(Message *m) { messages.insert(m); };
    void remMsg(Message *m) { messages.erase(m); };
};

void Message::add_to_Folders(const Message &m) {
    for (auto f : m.folders) {
        f->addMsg(this);
    }
}

void Message::remove_from_Folders() {
    for (auto f : folders) {
        f->remMsg(this);
    }
}

void Message::save(Folder &f) {
    folders.insert(&f);
    f.addMsg(this);
}

void Message::remove(Folder &f) {
    folders.erase(&f);
    f.remMsg(this);
}

Message::Message(const Message &m) : contents(m.contents), folders(m.folders) {
    add_to_Folders(m);
}

Message& Message::operator=(const Message &rhs) {
    remove_from_Folders();
    contents = rhs.contents;
    folders = rhs.folders;
    add_to_Folders(rhs);
    return *this;
}

Message::~Message() {
    remove_from_Folders();
}

void Folder::add_to_Messages(const Folder &f) {
    for (auto m : f.messages) {
        m->addFldr(this);
    }
}

void Folder::remove_from_Messages() {
    for (auto m : messages) {
        m->remFldr(this);
    }
}

Folder::Folder(const Folder& f) : messages(f.messages) {
    add_to_Messages(f);
}

Folder& Folder::operator=(const Folder& rhs) {
    remove_from_Messages();
    messages = rhs.messages;
    add_to_Messages(rhs);
    return *this;
}

Folder::~Folder() {
    remove_from_Messages();
}

void swap(Message &lhs, Message &rhs) {
    using std::swap;
    for (auto f : lhs.folders) {
        f->remMsg(&lhs);
    }
    for (auto f : rhs.folders) {
        f->remMsg(&rhs);
    }
    swap(lhs.contents, rhs.contents);
    swap(lhs.folders, rhs.folders);
    for (auto f : lhs.folders) {
        f->addMsg(&lhs);
    }
    for (auto f : rhs.folders) {
        f->addMsg(&rhs);
    }
}

#endif
```

### Q37

见Q36

### Q38

这里并不需要动态分配资源，使用拷贝和交换反而增加了实现的复杂度。

### Q39

```c++
#ifndef STRVEC_H
#define STRVEC_H

#include <utility>
#include <memory>
#include <string>

using namespace std;

class StrVec {
public:
    StrVec() : elements(nullptr), first_free(nullptr), cap(nullptr) {};
    StrVec(const StrVec&);
    StrVec& operator=(const StrVec&);
    ~StrVec();
    void push_back(const string&);
    size_t size() const { return first_free - elements; }
    size_t capacity() const { return cap - elements; }
    string *begin() const { return elements; }
    string *end() const { return first_free; }
    void reserve(size_t);
    void resize(size_t);
    void resize(size_t, const string&);
private:
    allocator<string> alloc;
    void chk_n_alloc() { if (size() == capacity()) reallocate(); }
    pair<string*, string*> alloc_n_copy(const string*, const string*);
    void free();
    void reallocate();
    string *elements;
    string *first_free;
    string *cap;
};

StrVec::StrVec(const StrVec &s) {
    auto newdata = alloc_n_copy(s.begin(), s.end());
    elements = newdata.first;
    first_free = cap = newdata.second;
}

StrVec& StrVec::operator=(const StrVec &s) {
    auto newdata = alloc_n_copy(s.begin(), s.end());
    free();
    elements = newdata.first;
    first_free = cap = newdata.second;
    return *this;
}

StrVec::~StrVec() {
    free();
}

void StrVec::push_back(const string& s) {
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
    resize(n, string());
}

void StrVec::resize(size_t n, const string& s) {
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

pair<string*, string*> StrVec::alloc_n_copy
        (const string *b, const string *e) {
    auto data = alloc.allocate(e-b);
    return {data, uninitialized_copy(b, e, data)};
}

void StrVec::free() {
    if (elements) {
        for (auto p = first_free; p != elements; ) {
            alloc.destroy(--p);
        }
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

#endif
```

### Q40

```c++
#ifndef STRVEC_H
#define STRVEC_H

#include <utility>
#include <memory>
#include <string>
#include <initializer_list>

using namespace std;

class StrVec {
public:
    StrVec() : elements(nullptr), first_free(nullptr), cap(nullptr) {};
    StrVec(initializer_list<string>);
    StrVec(const StrVec&);
    StrVec& operator=(const StrVec&);
    ~StrVec();
    void push_back(const string&);
    size_t size() const { return first_free - elements; }
    size_t capacity() const { return cap - elements; }
    string *begin() const { return elements; }
    string *end() const { return first_free; }
    void reserve(size_t);
    void resize(size_t);
    void resize(size_t, const string&);
private:
    allocator<string> alloc;
    void chk_n_alloc() { if (size() == capacity()) reallocate(); }
    pair<string*, string*> alloc_n_copy(const string*, const string*);
    void free();
    void reallocate();
    string *elements;
    string *first_free;
    string *cap;
};

StrVec::StrVec(initializer_list<string> l) {
    auto newdata = alloc_n_copy(l.begin(), l.end());
    elements = newdata.first;
    first_free = cap = newdata.second;
}

StrVec::StrVec(const StrVec &s) {
    auto newdata = alloc_n_copy(s.begin(), s.end());
    elements = newdata.first;
    first_free = cap = newdata.second;
}

StrVec& StrVec::operator=(const StrVec &s) {
    auto newdata = alloc_n_copy(s.begin(), s.end());
    free();
    elements = newdata.first;
    first_free = cap = newdata.second;
    return *this;
}

StrVec::~StrVec() {
    free();
}

void StrVec::push_back(const string& s) {
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
    resize(n, string());
}

void StrVec::resize(size_t n, const string& s) {
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

pair<string*, string*> StrVec::alloc_n_copy
        (const string *b, const string *e) {
    auto data = alloc.allocate(e-b);
    return {data, uninitialized_copy(b, e, data)};
}

void StrVec::free() {
    if (elements) {
        for (auto p = first_free; p != elements; ) {
            alloc.destroy(--p);
        }
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

#endif
```

### Q41

前置递增先递增再使用，会空出一个位置

### Q42

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
// StrVec.h
#ifndef STRVEC_H
#define STRVEC_H

#include <utility>
#include <memory>
#include <string>
#include <initializer_list>

class StrVec {
public:
    StrVec() : elements(nullptr), first_free(nullptr), cap(nullptr) {};
    StrVec(initializer_list<string>);
    StrVec(const StrVec&);
    StrVec& operator=(const StrVec&);
    ~StrVec();
    void push_back(const string&);
    size_t size() const { return first_free - elements; }
    size_t capacity() const { return cap - elements; }
    string *begin() const { return elements; }
    string *end() const { return first_free; }
    void reserve(size_t);
    void resize(size_t);
    void resize(size_t, const string&);
private:
    allocator<string> alloc;
    void chk_n_alloc() { if (size() == capacity()) reallocate(); }
    pair<string*, string*> alloc_n_copy(const string*, const string*);
    void free();
    void reallocate();
    string *elements;
    string *first_free;
    string *cap;
};

StrVec::StrVec(initializer_list<string> l) {
    auto newdata = alloc_n_copy(l.begin(), l.end());
    elements = newdata.first;
    first_free = cap = newdata.second;
}

StrVec::StrVec(const StrVec &s) {
    auto newdata = alloc_n_copy(s.begin(), s.end());
    elements = newdata.first;
    first_free = cap = newdata.second;
}

StrVec& StrVec::operator=(const StrVec &s) {
    auto newdata = alloc_n_copy(s.begin(), s.end());
    free();
    elements = newdata.first;
    first_free = cap = newdata.second;
    return *this;
}

StrVec::~StrVec() {
    free();
}

void StrVec::push_back(const string& s) {
    chk_n_alloc();
    alloc.construct(first_free++, s);
}

void StrVec::reserve(size_t n) {
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

void StrVec::resize(size_t n) {
    resize(n, string());
}

void StrVec::resize(size_t n, const string& s) {
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

pair<string*, string*> StrVec::alloc_n_copy
        (const string *b, const string *e) {
    auto data = alloc.allocate(e-b);
    return {data, uninitialized_copy(b, e, data)};
}

void StrVec::free() {
    if (elements) {
        for (auto p = first_free; p != elements; ) {
            alloc.destroy(--p);
        }
        alloc.deallocate(elements, cap-elements);
    }
}

void StrVec::reallocate() {
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
#include "StrVec.h"

class QueryResult;
class TextQuery {
public:
    using line_no = size_t;
    TextQuery(ifstream&);
    QueryResult query(const string&) const;
private:
    shared_ptr<StrVec> file;
    map<string, shared_ptr<set<line_no>>> wm;
};

class QueryResult {
friend ostream& print(ostream&, const QueryResult&);
public:
    QueryResult(string s,
                shared_ptr<set<TextQuery::line_no>> p,
                shared_ptr<StrVec> f) :
        sought(s), lines(p), file(f) {}
private:
    string sought;
    shared_ptr<set<TextQuery::line_no>> lines;
    shared_ptr<StrVec> file;
};

TextQuery::TextQuery(ifstream &is) : file(new StrVec) {
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

### Q43

```c++
#ifndef STRVEC_H
#define STRVEC_H

#include <utility>
#include <memory>
#include <algorithm>
#include <string>
#include <initializer_list>

using namespace std;

class StrVec {
public:
    StrVec() : elements(nullptr), first_free(nullptr), cap(nullptr) {};
    StrVec(initializer_list<string>);
    StrVec(const StrVec&);
    StrVec& operator=(const StrVec&);
    ~StrVec();
    void push_back(const string&);
    size_t size() const { return first_free - elements; }
    size_t capacity() const { return cap - elements; }
    string *begin() const { return elements; }
    string *end() const { return first_free; }
    void reserve(size_t);
    void resize(size_t);
    void resize(size_t, const string&);
private:
    allocator<string> alloc;
    void chk_n_alloc() { if (size() == capacity()) reallocate(); }
    pair<string*, string*> alloc_n_copy(const string*, const string*);
    void free();
    void reallocate();
    string *elements;
    string *first_free;
    string *cap;
};

StrVec::StrVec(initializer_list<string> l) {
    auto newdata = alloc_n_copy(l.begin(), l.end());
    elements = newdata.first;
    first_free = cap = newdata.second;
}

StrVec::StrVec(const StrVec &s) {
    auto newdata = alloc_n_copy(s.begin(), s.end());
    elements = newdata.first;
    first_free = cap = newdata.second;
}

StrVec& StrVec::operator=(const StrVec &s) {
    auto newdata = alloc_n_copy(s.begin(), s.end());
    free();
    elements = newdata.first;
    first_free = cap = newdata.second;
    return *this;
}

StrVec::~StrVec() {
    free();
}

void StrVec::push_back(const string& s) {
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
    resize(n, string());
}

void StrVec::resize(size_t n, const string& s) {
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

pair<string*, string*> StrVec::alloc_n_copy
        (const string *b, const string *e) {
    auto data = alloc.allocate(e-b);
    return {data, uninitialized_copy(b, e, data)};
}

void StrVec::free() {

    if (elements) {
        for_each(elements, first_free, [this](string &p) { alloc.destroy(&p); });
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

#endif
```

### Q44

```c++
#ifndef STRING_H
#define STRING_H

#include <algorithm>
#include <memory>
#include <cstring>

using namespace std;

class String {
public:
    String(): elements(nullptr), first_free(nullptr) {}
    String(const char *);
    String(const String&);
    String& operator=(const String&);
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

String& String::operator=(const String &s) {
    auto newdata = alloc_n_copy(s.begin(), s.end());
    free();
    elements = newdata.first;
    first_free = newdata.second;
    return *this;
}

String::~String() {
    free();
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

#endif
```

### Q45

左值引用即绑定到左值的引用，左值有持久的状态；右值引用即绑定到右值的引用，右值短暂，要么是字面常量，要么是表达式求值过程中创建的临时变量。

### Q46

```c++
int f();
vector<int> vi(100);
int&& r1 = f();
int& r2 = vi[0];
int& r3 = r1;
int&& r4 = vi[0] * f();
```

### Q47

```c++
#ifndef STRING_H
#define STRING_H

#include <algorithm>
#include <memory>
#include <cstring>

using namespace std;

class String {
public:
    String(): elements(nullptr), first_free(nullptr) {}
    String(const char *);
    String(const String&);
    String& operator=(const String&);
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
    cout << "String(const String &s)" << endl;
}

String& String::operator=(const String &s) {
    auto newdata = alloc_n_copy(s.begin(), s.end());
    free();
    elements = newdata.first;
    first_free = newdata.second;
    cout << "=" << endl;
    return *this;
}

String::~String() {
    free();
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

#endif
```

### Q48

// 执行两次拷贝（多出的一次是因为扩容）

```c++
// String.h
#ifndef STRING_H
#define STRING_H

#include <iostream>
#include <algorithm>
#include <memory>
#include <cstring>

using namespace std;

class String {
public:
    String(): elements(nullptr), first_free(nullptr) {}
    String(const char *);
    String(const String&);
    String& operator=(const String&);
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
    cout << "String(const String &s)" << endl;
}

String& String::operator=(const String &s) {
    auto newdata = alloc_n_copy(s.begin(), s.end());
    free();
    elements = newdata.first;
    first_free = newdata.second;
    cout << "=" << endl;
    return *this;
}

String::~String() {
    free();
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

#endif
```

```c++
#include <iostream>
#include <vector>
#include "String.h"

int main() {
    vector<String> v;
    v.push_back("a");
    v.push_back("b");
    return 0;
}
```

### Q49

```c++
// StrVec.h
#ifndef STRVEC_H
#define STRVEC_H

#include <utility>
#include <memory>
#include <algorithm>
#include <string>
#include <initializer_list>

using namespace std;

class StrVec {
public:
    StrVec() : elements(nullptr), first_free(nullptr), cap(nullptr) {};
    StrVec(initializer_list<string>);
    StrVec(const StrVec&);
    StrVec(StrVec&&) noexcept;
    StrVec& operator=(const StrVec&);
    StrVec& operator=(StrVec&&) noexcept;
    ~StrVec();
    void push_back(const string&);
    size_t size() const { return first_free - elements; }
    size_t capacity() const { return cap - elements; }
    string *begin() const { return elements; }
    string *end() const { return first_free; }
    void reserve(size_t);
    void resize(size_t);
    void resize(size_t, const string&);
private:
    allocator<string> alloc;
    void chk_n_alloc() { if (size() == capacity()) reallocate(); }
    pair<string*, string*> alloc_n_copy(const string*, const string*);
    void free();
    void reallocate();
    string *elements;
    string *first_free;
    string *cap;
};

StrVec::StrVec(initializer_list<string> l) {
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

void StrVec::push_back(const string& s) {
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
    resize(n, string());
}

void StrVec::resize(size_t n, const string& s) {
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

pair<string*, string*> StrVec::alloc_n_copy
        (const string *b, const string *e) {
    auto data = alloc.allocate(e-b);
    return {data, uninitialized_copy(b, e, data)};
}

void StrVec::free() {

    if (elements) {
        for_each(elements, first_free, [this](string &p) { alloc.destroy(&p); });
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

#endif
```

```c++
// String.h
#ifndef STRING_H
#define STRING_H

#include <iostream>
#include <algorithm>
#include <memory>
#include <cstring>

class String {
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
    alloc = move(s.alloc);
    elements = move(s.elements);
    first_free = move(s.first_free);
    s.elements = s.first_free = nullptr;
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
        alloc = move(s.alloc);
        elements = move(s.elements);
        first_free = move(s.first_free);
        s.elements = s.first_free = nullptr;
    }
    return *this;
}

String::~String() {
    free();
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

#endif
```

```c++
// Message.h
#ifndef MESSAGE_H
#define MESSAGE_H

#include <string>
#include <set>

using namespace std;

class Folder;

class Message {
    friend class Folder;
    friend void swap(Message &, Message &);
public:
    explicit Message(const string &str = "") : contents(str) {}
    Message(const Message&);
    Message(Message &&);
    Message& operator=(const Message&);
    Message& operator=(Message&&);
    ~Message();
    void save(Folder&);
    void remove(Folder&);
    void move_Folders(Message*);
private:
    string contents;
    set<Folder*> folders;
    void add_to_Folders(const Message&);
    void remove_from_Folders();
	void addFldr(Folder *f) { folders.insert(f); }
	void remFldr(Folder *f) { folders.erase(f); }
};

class Folder {
    friend class Message;
    friend void swap(Message &, Message &);
public:
    Folder() = default;
    Folder(const Folder&);
    Folder& operator=(const Folder&);
    ~Folder();
private:
    set<Message*> messages;
    void add_to_Messages(const Folder&);
    void remove_from_Messages();
    void addMsg(Message *m) { messages.insert(m); };
    void remMsg(Message *m) { messages.erase(m); };
};

void Message::add_to_Folders(const Message &m) {
    for (auto f : m.folders) {
        f->addMsg(this);
    }
}

void Message::remove_from_Folders() {
    for (auto f : folders) {
        f->remMsg(this);
    }
}

void Message::save(Folder &f) {
    folders.insert(&f);
    f.addMsg(this);
}

void Message::remove(Folder &f) {
    folders.erase(&f);
    f.remMsg(this);
}

void Message::move_Folders(Message *m) {
    folders = std::move(m->folders);
    for (auto f : folders) {
        f->remMsg(m);
        f->addMsg(this);
    }
    m->folders.clear();
}

Message::Message(const Message &m) : contents(m.contents), folders(m.folders) {
    add_to_Folders(m);
}

Message::Message(Message &&m) : contents(std::move(m.contents)) {
    move_Folders(&m);
}

Message& Message::operator=(const Message &rhs) {
    remove_from_Folders();
    contents = rhs.contents;
    folders = rhs.folders;
    add_to_Folders(rhs);
    return *this;
}

Message& Message::operator=(Message &&rhs) {
    if (this != &rhs) {
        remove_from_Folders();
        contents = std::move(rhs.contents);
        move_Folders(&rhs);
    }
    return *this;
}

Message::~Message() {
    remove_from_Folders();
}

void Folder::add_to_Messages(const Folder &f) {
    for (auto m : f.messages) {
        m->addFldr(this);
    }
}

void Folder::remove_from_Messages() {
    for (auto m : messages) {
        m->remFldr(this);
    }
}

Folder::Folder(const Folder& f) : messages(f.messages) {
    add_to_Messages(f);
}

Folder& Folder::operator=(const Folder& rhs) {
    remove_from_Messages();
    messages = rhs.messages;
    add_to_Messages(rhs);
    return *this;
}

Folder::~Folder() {
    remove_from_Messages();
}

void swap(Message &lhs, Message &rhs) {
    using std::swap;
    for (auto f : lhs.folders) {
        f->remMsg(&lhs);
    }
    for (auto f : rhs.folders) {
        f->remMsg(&rhs);
    }
    swap(lhs.contents, rhs.contents);
    swap(lhs.folders, rhs.folders);
    for (auto f : lhs.folders) {
        f->addMsg(&lhs);
    }
    for (auto f : rhs.folders) {
        f->addMsg(&rhs);
    }
}

#endif
```

### Q50

```c++
// String.h
#ifndef STRING_H
#define STRING_H

#include <iostream>
#include <algorithm>
#include <memory>
#include <cstring>

using namespace std;

class String {
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

#endif
```

```c++
#include <iostream>
#include <vector>
#include "String.h"

int main() {
    vector<String> v;
    v.push_back("a");
    String a;
    v.push_back(a);
    return 0;
}
```

### Q51

这里是移动操作而非拷贝操作，所以是合法的。

### Q52

第一个为左值，进行拷贝操作；第二个为右值，进行移动操作。

### Q53

交换需要给另一个不需要用到的变量赋值

```c++
#ifndef HASPTR_H
#define HASPTR_H

#include <string>

using namespace std;

class HasPtr {
friend void swap(HasPtr&, HasPtr&);
friend bool operator<(const HasPtr&, const HasPtr&);
public:
    HasPtr(const string &s = string()) : ps(new string(s)), i(0), use(new size_t(1)) {}
    HasPtr(const HasPtr &hp) : ps(hp.ps), i(hp.i), use(hp.use) { ++*use; }
    HasPtr(HasPtr &&hp) noexcept : ps(hp.ps), i(hp.i), use(hp.use) { hp.ps = nullptr; hp.use = nullptr; }
    HasPtr& operator=(HasPtr &hp) { swap(*this, hp); return *this; };
    HasPtr& operator=(HasPtr&&) noexcept;
    ~HasPtr();
private:
    string *ps;
    int i;
    size_t *use;
};

HasPtr& HasPtr::operator=(HasPtr&& hp) noexcept {
    if (this != &hp) {
        delete ps;
        delete use;
        ps = std::move(hp.ps);
        i = std::move(hp.i);
        use = std::move(hp.use);
    }
    return *this;
}

bool operator<(const HasPtr &lhs, const HasPtr &rhs) {
    return *lhs.ps < *rhs.ps;
}

HasPtr::~HasPtr() {
    if (--*use == 0) {
        delete ps;
        delete use;
    }
}

void swap(HasPtr &lhs, HasPtr &rhs) {
    using std::swap;
    swap(lhs.ps, rhs.ps);
    swap(lhs.i, rhs.i);
    swap(lhs.use, rhs.use);
}

#endif
```

### Q54

二义性调用，函数无法匹配

```c++
// HasPtr.h
#ifndef HASPTR_H
#define HASPTR_H

#include <string>

using namespace std;

class HasPtr {
friend void swap(HasPtr&, HasPtr&);
friend bool operator<(const HasPtr&, const HasPtr&);
public:
    HasPtr(const string &s = string()) : ps(new string(s)), i(0), use(new size_t(1)) {}
    HasPtr(const HasPtr &hp) : ps(hp.ps), i(hp.i), use(hp.use) { ++*use; }
    HasPtr(HasPtr &&hp) noexcept : ps(hp.ps), i(hp.i), use(hp.use) { hp.ps = nullptr; hp.use = nullptr; }
    HasPtr& operator=(HasPtr&hp) { swap(*this, hp); return *this; };
    HasPtr& operator=(HasPtr&&) noexcept;
    ~HasPtr();
private:
    string *ps;
    int i;
    size_t *use;
};

HasPtr& HasPtr::operator=(HasPtr&& hp) noexcept {
    if (this != &hp) {
        delete ps;
        delete use;
        ps = std::move(hp.ps);
        i = std::move(hp.i);
        use = std::move(hp.use);
    }
    return *this;
}

bool operator<(const HasPtr &lhs, const HasPtr &rhs) {
    return *lhs.ps < *rhs.ps;
}

HasPtr::~HasPtr() {
    if (--*use == 0) {
        delete ps;
        delete use;
    }
}

void swap(HasPtr &lhs, HasPtr &rhs) {
    using std::swap;
    swap(lhs.ps, rhs.ps);
    swap(lhs.i, rhs.i);
    swap(lhs.use, rhs.use);
}

#endif
```

### Q55

```c++
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
    StrBlob(const StrBlob&);
    StrBlob& operator=(const StrBlob&);
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

#endif
```

### Q56

会递归调用自身直至堆栈溢出

### Q57

可以，Foo(*this)是右值，会调用右值引用版本的sorted

### Q58

```c++
// Foo.h
#ifndef FOO_H
#define FOO_H

#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

class Foo {
public:
    Foo sorted() &&;
    Foo sorted() const &;
private:
    vector<int> data;
};

Foo Foo::sorted() && {
    cout << "Foo::sorted() &&" << endl;
    sort(data.begin(), data.end());
    return *this;
}

Foo Foo::sorted() const & {
    cout << "Foo::sorted() const &" << endl;
    // Foo ret(*this);
    // return ret.sorted();
    return Foo(*this).sorted();
}

#endif
```

```c++
#include <iostream>
#include "Foo.h"

int main() {
    Foo f;
    f.sorted();
    return 0;
}
```