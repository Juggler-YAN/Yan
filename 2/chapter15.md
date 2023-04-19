### Q1

虚成员函数是基类希望其派生类进行覆盖的成员函数。

### Q2

派生类的成员函数可以访问基类中protected访问运算符的成员，而不能访问private的。

### Q3

```c++
#ifndef QUOTE_H
#define QUOTE_H

#include <iostream>
#include <string>

using namespace std;

class Quote {
public:
    Quote() = default;
    Quote(const string &book, double sales_price) : bookNo(book), price(sales_price) {}
    string isbn() const { return bookNo; }
    virtual double net_price(size_t n) const { return n * price; }
    virtual ~Quote() = default;
private:
    string bookNo;
protected:
    double price = 0.0;
};

double print_total(ostream &os, const Quote &item, size_t n) {
    double ret = item.net_price(n);
    os << "ISBN: " << item.isbn() << " # sold: " << n << " total due: " << ret << endl;
    return ret;
}

#endif
```

### Q4

不正确，是定义而不是声明，而且一个类不能继承自身
不正确，是定义而不是声明
不正确，声明中不能包含派生列表

### Q5

```c++
// Quote.h
#ifndef QUOTE_H
#define QUOTE_H

#include <iostream>
#include <string>

using namespace std;

class Quote {
    friend double print_total(ostream&, const Quote&, size_t);
public:
    Quote() = default;
    Quote(const string &book, double sales_price) : bookNo(book), price(sales_price) {}
    string isbn() const { return bookNo; }
    virtual double net_price(size_t n) const { return n * price; }
    virtual ~Quote() = default;
private:
    string bookNo;
protected:
    double price = 0.0;
};

double print_total(ostream &os, const Quote &item, size_t n) {
    double ret = item.net_price(n);
    os << "ISBN: " << item.isbn() << " # sold: " << n << " total due: " << ret << endl;
    return ret;
}

#endif
```

```c++
// Bulk_quote.h
#ifndef BULK_QUOTE_H
#define BULK_QUOTE_H

#include "Quote.h"

using namespace std;

class Bulk_quote : public Quote {
public:
    Bulk_quote() = default;
    Bulk_quote(const string& book, double p, size_t qty, double disc) : 
               Quote(book, p), min_qty(qty), discount(disc) {}
    double net_price(size_t) const override;
private:
    size_t min_qty = 0;
    double discount = 0.0;
};

double Bulk_quote::net_price(size_t cnt) const {
    if (cnt >= min_qty) {
        return cnt * (1 - discount) * price;
    }
    else {
        return cnt * price;
    }
}

#endif
```

### Q6

```c++
// Quote.h
#ifndef QUOTE_H
#define QUOTE_H

#include <iostream>
#include <string>

using namespace std;

class Quote {
    friend double print_total(ostream&, const Quote&, size_t);
public:
    Quote() = default;
    Quote(const string &book, double sales_price) : bookNo(book), price(sales_price) {}
    string isbn() const { return bookNo; }
    virtual double net_price(size_t n) const { return n * price; }
    virtual ~Quote() = default;
private:
    string bookNo;
protected:
    double price = 0.0;
};

double print_total(ostream &os, const Quote &item, size_t n) {
    double ret = item.net_price(n);
    os << "ISBN: " << item.isbn() << " # sold: " << n << " total due: " << ret << endl;
    return ret;
}

#endif
```

```c++
// Bulk_quote.h
#ifndef BULK_QUOTE_H
#define BULK_QUOTE_H

#include "Quote.h"

using namespace std;

class Bulk_quote : public Quote {
public:
    Bulk_quote() = default;
    Bulk_quote(const string& book, double p, size_t qty, double disc) : 
               Quote(book, p), min_qty(qty), discount(disc) {}
    double net_price(size_t) const override;
private:
    size_t min_qty = 0;
    double discount = 0.0;
};

double Bulk_quote::net_price(size_t cnt) const {
    if (cnt >= min_qty) {
        return cnt * (1 - discount) * price;
    }
    else {
        return cnt * price;
    }
}

#endif
```

```c++
#include <iostream>
#include "Quote.h"
#include "Bulk_quote.h"

int main() {
    Quote a("A", 10);
    Bulk_quote b("A", 10, 5, 0.1);
    print_total(cout, a, 10);
    print_total(cout, b, 10);
    return 0;
}
```

### Q7

```c++
// Quote.h
#ifndef QUOTE_H
#define QUOTE_H

#include <iostream>
#include <string>

using namespace std;

class Quote {
    friend double print_total(ostream&, const Quote&, size_t);
public:
    Quote() = default;
    Quote(const string &book, double sales_price) : bookNo(book), price(sales_price) {}
    string isbn() const { return bookNo; }
    virtual double net_price(size_t n) const { return n * price; }
    virtual ~Quote() = default;
private:
    string bookNo;
protected:
    double price = 0.0;
};

double print_total(ostream &os, const Quote &item, size_t n) {
    double ret = item.net_price(n);
    os << "ISBN: " << item.isbn() << " # sold: " << n << " total due: " << ret << endl;
    return ret;
}

#endif
```

```c++
// Few_quote.h
#ifndef FEW_QUOTE_H
#define FEW_QUOTE_H

#include "Quote.h"

using namespace std;

class Few_quote : public Quote {
public:
    Few_quote() = default;
    Few_quote(const string& book, double p, size_t qty, double disc) : 
               Quote(book, p), max_qty(qty), discount(disc) {}
    double net_price(size_t) const override;
private:
    size_t max_qty = 0;
    double discount = 0.0;
};

double Few_quote::net_price(size_t cnt) const {
    if (cnt <= max_qty) {
        return cnt * (1 - discount) * price;
    }
    else {
        return max_qty * (1 - discount) * price + (cnt - max_qty) * price;
    }
}

#endif
```

```c++
#include <iostream>
#include "Quote.h"
#include "Few_quote.h"

using namespace std;

int main() {
    Quote a("A", 10);
    Few_quote b("A", 10, 5, 0.1);
    print_total(cout, a, 10);
    print_total(cout, b, 10);
    return 0;
}
```

### Q8

表达式的静态类型在编译时总是已知的，它是变量声明时的类型或表达式生成的类型；动态类型则是变量或表达式表示的内存中的对象类型。动态类型直到运行时才可知。

### Q9

基类的指针或引用的静态类型可能与其动态类型不一致

```c++
// Quote.h
#ifndef QUOTE_H
#define QUOTE_H

#include <iostream>
#include <string>

using namespace std;

class Quote {
    friend double print_total(ostream&, const Quote&, size_t);
public:
    Quote() = default;
    Quote(const string &book, double sales_price) : bookNo(book), price(sales_price) {}
    string isbn() const { return bookNo; }
    virtual double net_price(size_t n) const { return n * price; }
    virtual ~Quote() = default;
private:
    string bookNo;
protected:
    double price = 0.0;
};

double print_total(ostream &os, const Quote &item, size_t n) {
    double ret = item.net_price(n);
    os << "ISBN: " << item.isbn() << " # sold: " << n << " total due: " << ret << endl;
    return ret;
}

#endif
```

```c++
// Few_quote.h
#ifndef FEW_QUOTE_H
#define FEW_QUOTE_H

#include "Quote.h"

using namespace std;

class Few_quote : public Quote {
public:
    Few_quote() = default;
    Few_quote(const string& book, double p, size_t qty, double disc) : 
               Quote(book, p), max_qty(qty), discount(disc) {}
    double net_price(size_t) const override;
private:
    size_t max_qty = 0;
    double discount = 0.0;
};

double Few_quote::net_price(size_t cnt) const {
    if (cnt <= max_qty) {
        return cnt * (1 - discount) * price;
    }
    else {
        return max_qty * (1 - discount) * price + (cnt - max_qty) * price;
    }
}

#endif
```

```c++
#include <iostream>
#include "Quote.h"
#include "Few_quote.h"

using namespace std;

int main() {
    Few_quote a("A", 10, 5, 0.1);
    // 1
    Quote *p = &a;
    cout << p->net_price(10) << endl;
    // 2
    Quote &r = a;
    cout << r.net_price(10) << endl;
    return 0;
}
```

### Q10

ifstream是istream的派生类，所以可以使用

### Q11

```c++
// Quote.h
#ifndef QUOTE_H
#define QUOTE_H

#include <iostream>
#include <string>

using namespace std;

class Quote {
    friend double print_total(ostream&, const Quote&, size_t);
public:
    Quote() = default;
    Quote(const string &book, double sales_price) : bookNo(book), price(sales_price) {}
    string isbn() const { return bookNo; }
    virtual double net_price(size_t n) const { return n * price; }
    virtual ~Quote() = default;
    virtual void debug() const;
private:
    string bookNo;
protected:
    double price = 0.0;
};

double print_total(ostream &os, const Quote &item, size_t n) {
    double ret = item.net_price(n);
    os << "ISBN: " << item.isbn() << " # sold: " << n << " total due: " << ret << endl;
    return ret;
}

void Quote::debug() const {
    cout << bookNo << " " << price;
}

#endif
```

```c++
// Bulk_quote.h
#ifndef BULK_QUOTE_H
#define BULK_QUOTE_H

#include <iostream>
#include "Quote.h"

using namespace std;

class Bulk_quote : public Quote {
public:
    Bulk_quote() = default;
    Bulk_quote(const string& book, double p, size_t qty, double disc) : 
               Quote(book, p), min_qty(qty), discount(disc) {}
    double net_price(size_t) const override;
    void debug() const override;
private:
    size_t min_qty = 0;
    double discount = 0.0;
};

double Bulk_quote::net_price(size_t cnt) const {
    if (cnt >= min_qty) {
        return cnt * (1 - discount) * price;
    }
    else {
        return cnt * price;
    }
}

void Bulk_quote::debug() const {
    Quote::debug();
    cout << " " << min_qty << " " << discount;
}

#endif
```

```c++
#include <iostream>
#include "Quote.h"
#include "Bulk_quote.h"

int main() {
    Quote a("A", 10);
    Bulk_quote b("A", 10, 5, 0.1);
    a.debug();
    cout << endl;
    b.debug();
    cout << endl;
    return 0;
}
```

### Q12

有必要。override是重写基类中相同名称的虚函数，final是阻止它的派生类重写当前虚函数，两者不冲突。

### Q13

有。派生类中的print会循环调用自身直至内存耗尽

```c++
class base {
public:
	string name() { return basename;}
	virtual void print(ostream &os) { os << basename; }
private:
	string basename;
};
class derived : public base {
public:
	void print(ostream &os) { base::print(os); os << " " << i; }
private:
	int i;
};

```

### Q14

```c++
base::print()
derived::print()
base::name()
base::name()
base::print()
derived::print()
```

### Q15

```c++
// Quote.h
#ifndef QUOTE_H
#define QUOTE_H

#include <iostream>
#include <string>

using namespace std;

class Quote {
    friend double print_total(ostream&, const Quote&, size_t);
public:
    Quote() = default;
    Quote(const string &book, double sales_price) : bookNo(book), price(sales_price) {}
    string isbn() const { return bookNo; }
    virtual double net_price(size_t n) const { return n * price; }
    virtual ~Quote() = default;
    virtual void debug() const;
private:
    string bookNo;
protected:
    double price = 0.0;
};

double print_total(ostream &os, const Quote &item, size_t n) {
    double ret = item.net_price(n);
    os << "ISBN: " << item.isbn() << " # sold: " << n << " total due: " << ret << endl;
    return ret;
}

void Quote::debug() const {
    cout << bookNo << " " << price;
}

#endif
```

```c++
// Disc_quote.h
#ifndef DISC_QUOTE_H
#define DISC_QUOTE_H

#include <string>
#include "Quote.h"

using namespace std;

class Disc_quote : public Quote {
public:
    Disc_quote() = default;
    Disc_quote(const string &book, double p, size_t qty, double disc) : 
               Quote(book, p), quantity(qty), discount(disc) {}
    double net_price(size_t) const = 0;
protected:
    size_t quantity = 0;
    double discount = 0.0;
};

#endif
```

```c++
// Bulk_quote.h
#ifndef BULK_QUOTE_H
#define BULK_QUOTE_H

#include <iostream>
#include <string>
#include "Disc_quote.h"

using namespace std;

class Bulk_quote : public Disc_quote {
public:
    Bulk_quote() = default;
    Bulk_quote(const string &book, double p, size_t qty, double disc) : 
               Disc_quote(book, p, qty, disc) {}
    double net_price(size_t) const override;
    void debug() const override;
};

double Bulk_quote::net_price(size_t cnt) const {
    if (cnt >= quantity) {
        return cnt * (1 - discount) * price;
    }
    else {
        return cnt * price;
    }
}

void Bulk_quote::debug() const {
    Quote::debug();
    cout << " " << quantity << " " << discount;
}

#endif
```

```c++
#include <iostream>
#include "Quote.h"
#include "Disc_quote.h"
#include "Bulk_quote.h"

int main() {
    Quote a("A", 10);
    Bulk_quote b("A", 10, 5, 0.1);
    print_total(cout, a, 10);
    print_total(cout, b, 10);
    a.debug();
    cout << endl;
    b.debug();
    cout << endl;
    return 0;
}
```

### Q16

```c++
// Quote.h
#ifndef QUOTE_H
#define QUOTE_H

#include <iostream>
#include <string>

using namespace std;

class Quote {
    friend double print_total(ostream&, const Quote&, size_t);
public:
    Quote() = default;
    Quote(const string &book, double sales_price) : bookNo(book), price(sales_price) {}
    string isbn() const { return bookNo; }
    virtual double net_price(size_t n) const { return n * price; }
    virtual ~Quote() = default;
    virtual void debug() const;
private:
    string bookNo;
protected:
    double price = 0.0;
};

double print_total(ostream &os, const Quote &item, size_t n) {
    double ret = item.net_price(n);
    os << "ISBN: " << item.isbn() << " # sold: " << n << " total due: " << ret << endl;
    return ret;
}

void Quote::debug() const {
    cout << bookNo << " " << price;
}

#endif
```

```c++
// Disc_quote.h
#ifndef DISC_QUOTE_H
#define DISC_QUOTE_H

#include <string>
#include "Quote.h"

using namespace std;

class Disc_quote : public Quote {
public:
    Disc_quote() = default;
    Disc_quote(const string &book, double p, size_t qty, double disc) : 
               Quote(book, p), quantity(qty), discount(disc) {}
    double net_price(size_t) const = 0;
protected:
    size_t quantity = 0;
    double discount = 0.0;
};

#endif
```

```c++
// Few_quote.h
#ifndef FEW_QUOTE_H
#define FEW_QUOTE_H

#include <string>
#include "Disc_quote.h"

using namespace std;

class Few_quote : public Disc_quote {
public:
    Few_quote() = default;
    Few_quote(const string& book, double p, size_t qty, double disc) : 
               Disc_quote(book, p, qty, disc) {}
    double net_price(size_t) const override;
};

double Few_quote::net_price(size_t cnt) const {
    if (cnt <= quantity) {
        return cnt * (1 - discount) * price;
    }
    else {
        return quantity * (1 - discount) * price + (cnt - quantity) * price;
    }
}

#endif
```

```c++
#include <iostream>
#include "Quote.h"
#include "Few_quote.h"

int main() {
    Quote a("A", 10);
    Few_quote b("A", 10, 5, 0.1);
    print_total(cout, a, 10);
    print_total(cout, b, 10);
    return 0;
}
```

### Q17

```c++
// Quote.h
#ifndef QUOTE_H
#define QUOTE_H

#include <iostream>
#include <string>

using namespace std;

class Quote {
    friend double print_total(ostream&, const Quote&, size_t);
public:
    Quote() = default;
    Quote(const string &book, double sales_price) : bookNo(book), price(sales_price) {}
    string isbn() const { return bookNo; }
    virtual double net_price(size_t n) const { return n * price; }
    virtual ~Quote() = default;
    virtual void debug() const;
private:
    string bookNo;
protected:
    double price = 0.0;
};

double print_total(ostream &os, const Quote &item, size_t n) {
    double ret = item.net_price(n);
    os << "ISBN: " << item.isbn() << " # sold: " << n << " total due: " << ret << endl;
    return ret;
}

void Quote::debug() const {
    cout << bookNo << " " << price;
}

#endif
```

```c++
// Disc_quote.h
#ifndef DISC_QUOTE_H
#define DISC_QUOTE_H

#include <string>
#include "Quote.h"

using namespace std;

class Disc_quote : public Quote {
public:
    Disc_quote() = default;
    Disc_quote(const string &book, double p, size_t qty, double disc) : 
               Quote(book, p), quantity(qty), discount(disc) {}
    double net_price(size_t) const = 0;
protected:
    size_t quantity = 0;
    double discount = 0.0;
};

#endif
```


```c++
#include <iostream>
#include "Quote.h"
#include "Disc_quote.h"

int main() {
    Disc_quote a;
    return 0;
}
```

### Q18

1. 合法
2. 非法
3. 非法
4. 合法
5. 非法
6. 非法

只有当派生类公有地继承基类时，用户代码才能使用派生类向基类的转换。如果派生类继承基类的方式是受保护的或者私有的，则用户代码不能使用该转换。

### Q19

Base类：合法；
Pub_Derv类：合法；
Priv_Derv类：合法；
Prot_Derv类：合法；
Derived_from_Public类：合法；
Derived_from_Private类：非法；
Derived_from_Protected类：合法。

无论派生类以什么方式继承基类，派生类的成员函数和友元都能使用派生类向基类的转换；派生类向其直接基类的类型转换对于派生类的成员来说永远是可访问的。如果派生类继承基类的方式是公有的或者受保护的，则派生类的成员和友元可以使用派生类向基类的类型转换；反之，如果派生类继承基类的方式是私有的，则不能使用。

### Q20

```c++
#include <iostream>

class Base {
public:
    void memfcn(Base &b) { b = *this; };
protected:
    int prot_mem;
private:
    char priv_mem;
};

struct Pub_Derv : public Base {
public:
    void memfcn(Base &b) { b = *this; };
};

struct Priv_Derv : private Base {
public:
    void memfcn(Base &b) { b = *this; };
};

struct Prot_Derv : protected Base {
public:
    void memfcn(Base &b) { b = *this; };
};

struct Derived_from_Public : public Pub_Derv {
public:
    void memfcn(Base &b) { b = *this; };
};

struct Derived_from_Private : private Priv_Derv {
public:
    // void memfcn(Base &b) { b = *this; };
};

struct Derived_from_Protected : protected Prot_Derv {
public:
    void memfcn(Base &b) { b = *this; };
};

int main() {
    Pub_Derv d1;
    Priv_Derv d2;
    Prot_Derv d3;
    Derived_from_Public dd1;
    Derived_from_Private dd2;
    Derived_from_Protected dd3;
    Base * p = &d1;
    // p = &d2;
    // p = &d3;
    p = &dd1;
    // p = &dd2;
    // p = &dd3;
    return 0;
}
```

### Q21

```c++
// Graph.h
#ifndef GRAPH_H
#define GRAPH_H

#include <string>

using namespace std;

static const double PI = 3.1415926;

class Shape {
public:
    virtual string name() const = 0;
    virtual ~Shape() {}
};

class Shape_2D : public Shape {
public:
    virtual double perimeter() const = 0;
    virtual double area() const = 0;
    ~Shape_2D() override {}
};

class Shape_3D : public Shape {
public:
    virtual double volume() const = 0;
    ~Shape_3D() override {}
};

class Box : public Shape_2D {
public:
    Box() = default;
    Box(double x, double y) : len_x(x), len_y(y) {}
    string name() const override { return string("Box"); }
    double perimeter() const override { return (len_x + len_y) * 2; }
    double area() const override { return len_x * len_y; }
    ~Box() override {}
private:
    double len_x;
    double len_y;
};

class Circle : public Shape_2D {
public:
    Circle() = default;
    Circle(double r) : radius(r) {}
    string name() const override { return string("Circle"); }
    double perimeter() const override { return 2 * PI * radius; }
    double area() const override { return PI * radius * radius; }
    ~Circle() override {}
private:
    double radius;
};

class Sphere : public Shape_3D {
public:
    Sphere() = default;
    Sphere(double r) : radius(r) {}
    string name() const override { return string("Sphere"); }
    double volume() const override { return 4.0 / 3 * PI * radius * radius * radius; }
    ~Sphere() override {}
private:
    double radius;
};

class Cone : public Shape_3D {
public:
    Cone() = default;
    Cone(double r, double h) : bottomradius(r), height(h) {}
    string name() const override { return string("Cone"); }
    double volume() const override { return 1.0 / 3 * PI * bottomradius * bottomradius * height; }
    ~Cone() override {}
private:
    double bottomradius;
    double height;
};

#endif
```

```c++
#include <iostream>
#include "Graph.h"

int main() {
    Box box(2,3);
    cout << box.name() << endl;
    cout << box.perimeter() << endl;
    cout << box.area() << endl;
    Circle circle(2);
    cout << circle.name() << endl;
    cout << circle.perimeter() << endl;
    cout << circle.area() << endl;
    Sphere sphere(2);
    cout << sphere.name() << endl;
    cout << sphere.volume() << endl;
    Cone cone(2,2);
    cout << cone.name() << endl;
    cout << cone.volume() << endl;
    return 0;
}
```

### Q22

见Q21

### Q23

```c++
#include <iostream>
#include <string>

using namespace std;

class Base {
public:
    virtual int fcn() {
        cout << "Base::fcn()" << endl;
        return 0;
    }
};

class D1 : public Base {
public:
    int fcn(int) {
        cout << "D1::fcn(int)" << endl;
        return 0;
    }
    int fcn() {
        cout << "D1::fcn()\n";
        return 0;
    }

    virtual void f2() {
        cout << "D1::f2()" << endl;
    }
};

class D2 : public D1 {
public:
    int fcn(int) {
        cout << "D2::fcn(int)" << endl;
        return 0;
    }

    int fcn() override {
        cout << "D2::fcn()" << endl;
        return 0;
    }

    void f2() override {
        cout << "D2::f2()" << endl;
    }
};

int main() {
    Base bobj; D1 d1obj; D2 d2obj;

    Base *bp1 = &bobj, *bp2 = &d1obj, *bp3 = &d2obj;
    bp1->fcn();
    bp2->fcn();
    bp3->fcn();

    D1 *d1p = &d1obj; D2 *d2p = &d2obj;
    // bp2->f2();
    d1p->f2();
    d2p->f2();

    Base *p = &d2obj; D1 * p2 = &d2obj; D2 * p3 = &d2obj;
    // p1->fcn(42);
    p2->fcn(42);
    p3->fcn(42);

    return 0;
}
```

### Q24

基类需要定义一个虚析构函数。这样我们就能动态分配继承体系中的对象了。

### Q25

Disc_quote中定义了一个构造函数，所以默认构造函数是被删除的，需要显式地定义。如果去掉了Disc_quote的默认构造函数，Bulk_quote的默认构造函数是被删除的。

### Q26

```c++
// Quote.h
#ifndef QUOTE_H
#define QUOTE_H

#include <iostream>
#include <string>

using namespace std;

class Quote {
    friend double print_total(ostream&, const Quote&, size_t);
public:
    // Quote() = default;
    Quote() {
        cout << "Quote()" << endl;
    };
    Quote(const string &book, double sales_price) : bookNo(book), price(sales_price) {
        cout << "Quote(const string&, double)" << endl;
    }
    Quote(const Quote &rhs) : bookNo(rhs.bookNo), price(rhs.price) {
        cout << "Quote(const Quote&)" << endl;
    }
    Quote(Quote &&rhs) noexcept : bookNo(std::move(rhs.bookNo)), price(std::move(rhs.price)) {
        cout << "Quote(Quote&&)" << endl;
    }
    Quote& operator=(const Quote &rhs) {
        cout << "Quote& operator=(const Quote &)" << endl;
        bookNo = rhs.bookNo;
        price = rhs.price;
        return *this;
    }
    Quote& operator=(Quote &&rhs) noexcept {
        cout << "Quote& operator=(Quote &&)" << endl;
        if (this != &rhs) {
            bookNo = std::move(rhs.bookNo);
            price = std::move(rhs.price);
        }
        return *this;
    }
    // virtual ~Quote() = default;
    virtual ~Quote() {
        cout << "~Quote()" << endl;
    };
    string isbn() const { return bookNo; }
    virtual double net_price(size_t n) const { return n * price; }
private:
    string bookNo;
protected:
    double price = 0.0;
};

double print_total(ostream &os, const Quote &item, size_t n) {
    double ret = item.net_price(n);
    os << "ISBN: " << item.isbn() << " # sold: " << n << " total due: " << ret << endl;
    return ret;
}

#endif
```

```c++
// Bulk_quote.h
#ifndef BULK_QUOTE_H
#define BULK_QUOTE_H

#include <iostream>
#include "Quote.h"

using namespace std;

class Bulk_quote : public Quote {
public:
    // Bulk_quote() = default;
    Bulk_quote() {
        cout << "Bulk_quote()" << endl;
    }
    Bulk_quote(const string& book, double p, size_t qty, double disc) : 
               Quote(book, p), min_qty(qty), discount(disc) {
        cout << "Bulk_quote(const string&, double, size_t, double)" << endl;
    }
    Bulk_quote(const Bulk_quote &rhs) : Quote(rhs), min_qty(rhs.min_qty), discount(rhs.discount) {
        cout << "Bulk_quote(const Bulk_quote&)" << endl;
    }
    Bulk_quote(Bulk_quote &&rhs) noexcept : Quote(std::move(rhs)), min_qty(std::move(rhs.min_qty)), 
        discount(std::move(rhs.discount)) {
        cout << "Bulk_quote(Bulk_quote&&)" << endl;
    }
    Bulk_quote& operator=(const Bulk_quote &rhs) {
        Quote::operator=(rhs);
        cout << "Bulk_quote& operator=(const Bulk_quote&)" << endl;
        min_qty = rhs.min_qty;
        discount = rhs.discount;
        return *this;
    }
    Bulk_quote& operator=(Bulk_quote &&rhs) noexcept {
        Quote::operator=(std::move(rhs));
        cout << "Bulk_quote& operator=(Bulk_quote &&)" << endl;
        if (this != &rhs) {
            min_qty = rhs.min_qty;
            discount = rhs.discount;
        }
        return *this;
    }
    virtual ~Bulk_quote() {
        cout << "~Bulk_quote()" << endl;
    }
    double net_price(size_t) const override;
private:
    size_t min_qty = 0;
    double discount = 0.0;
};

double Bulk_quote::net_price(size_t cnt) const {
    if (cnt >= min_qty) {
        return cnt * (1 - discount) * price;
    }
    else {
        return cnt * price;
    }
}

#endif
```

```c++
#include <iostream>
#include "Quote.h"
#include "Bulk_quote.h"

int main() {
    cout << "********************" << endl;
    Bulk_quote a("a", 10, 5, 0.1);
    cout << "********************" << endl;
    Bulk_quote b1(a);
    cout << "********************" << endl;
    Bulk_quote c1(std::move(a));
    cout << "********************" << endl;
    Bulk_quote d1;
    d1 = a;
    cout << "********************" << endl;
    Bulk_quote e1;
    e1 = std::move(a);
    cout << "********************" << endl;
    Quote b2(a);
    cout << "********************" << endl;
    Quote c2(std::move(a));
    cout << "********************" << endl;
    Quote d2;
    d2 = a;
    cout << "********************" << endl;
    Quote e2;
    e2 = std::move(a);
    cout << "********************" << endl;
    
    return 0;
}
```

### Q27

```c++
// Quote.h
#ifndef QUOTE_H
#define QUOTE_H

#include <iostream>
#include <string>

using namespace std;

class Quote {
    friend double print_total(ostream&, const Quote&, size_t);
public:
    Quote() = default;
    Quote(const string &book, double sales_price) : bookNo(book), price(sales_price) {}
    string isbn() const { return bookNo; }
    virtual double net_price(size_t n) const { return n * price; }
    virtual ~Quote() = default;
    virtual void debug() const;
private:
    string bookNo;
protected:
    double price = 0.0;
};

double print_total(ostream &os, const Quote &item, size_t n) {
    double ret = item.net_price(n);
    os << "ISBN: " << item.isbn() << " # sold: " << n << " total due: " << ret << endl;
    return ret;
}

void Quote::debug() const {
    cout << bookNo << " " << price;
}

#endif
```

```c++
// Disc_quote.h
#ifndef DISC_QUOTE_H
#define DISC_QUOTE_H

#include <string>
#include "Quote.h"

using namespace std;

class Disc_quote : public Quote {
public:
    Disc_quote() = default;
    Disc_quote(const string &book, double p, size_t qty, double disc) : 
               Quote(book, p), quantity(qty), discount(disc) {}
    double net_price(size_t) const = 0;
protected:
    size_t quantity = 0;
    double discount = 0.0;
};

#endif
```

```c++
// Bulk_quote.h
#ifndef BULK_QUOTE_H
#define BULK_QUOTE_H

#include <iostream>
#include "Quote.h"

class Bulk_quote : public Disc_quote {
public:
    // Bulk_quote() = default;
    // Bulk_quote(const string& book, double p, size_t qty, double disc) : 
    //            Quote(book, p), min_qty(qty), discount(disc) {}
    using Disc_quote::Disc_quote;
    double net_price(size_t) const override;
    void debug() const override;
};

double Bulk_quote::net_price(size_t cnt) const {
    if (cnt >= quantity) {
        return cnt * (1 - discount) * price;
    }
    else {
        return cnt * price;
    }
}

void Bulk_quote::debug() const {
    Quote::debug();
    cout << " " << quantity << " " << discount;
}

#endif
```

```c++
#include <iostream>
#include "Quote.h"
#include "Disc_quote.h"
#include "Bulk_quote.h"

int main() {
    Quote a("A", 10);
    Bulk_quote b("A", 10, 5, 0.1);
    print_total(cout, a, 10);
    print_total(cout, b, 10);
    a.debug();
    cout << endl;
    b.debug();
    cout << endl;
    return 0;
}
```

### Q28

```c++
// Quote.h
#ifndef QUOTE_H
#define QUOTE_H

#include <iostream>
#include <string>

using namespace std;

class Quote {
    friend double print_total(ostream&, const Quote&, size_t);
public:
    Quote() = default;
    Quote(const string &book, double sales_price) : bookNo(book), price(sales_price) {}
    string isbn() const { return bookNo; }
    virtual double net_price(size_t n) const { return n * price; }
    virtual ~Quote() = default;
    virtual void debug() const;
private:
    string bookNo;
protected:
    double price = 0.0;
};

double print_total(ostream &os, const Quote &item, size_t n) {
    double ret = item.net_price(n);
    os << "ISBN: " << item.isbn() << " # sold: " << n << " total due: " << ret << endl;
    return ret;
}

void Quote::debug() const {
    cout << bookNo << " " << price;
}

#endif
```

```c++
// Disc_quote.h
#ifndef DISC_QUOTE_H
#define DISC_QUOTE_H

#include <string>
#include "Quote.h"

using namespace std;

class Disc_quote : public Quote {
public:
    Disc_quote() = default;
    Disc_quote(const string &book, double p, size_t qty, double disc) : 
               Quote(book, p), quantity(qty), discount(disc) {}
    double net_price(size_t) const = 0;
protected:
    size_t quantity = 0;
    double discount = 0.0;
};

#endif
```

```c++
// Bulk_quote.h
#ifndef BULK_QUOTE_H
#define BULK_QUOTE_H

#include <iostream>
#include "Quote.h"

using namespace std;

class Bulk_quote : public Disc_quote {
public:
    // Bulk_quote() = default;
    // Bulk_quote(const string& book, double p, size_t qty, double disc) : 
    //            Quote(book, p), min_qty(qty), discount(disc) {}
    using Disc_quote::Disc_quote;
    double net_price(size_t) const override;
    void debug() const override;
};

double Bulk_quote::net_price(size_t cnt) const {
    if (cnt >= quantity) {
        return cnt * (1 - discount) * price;
    }
    else {
        return cnt * price;
    }
}

void Bulk_quote::debug() const {
    Quote::debug();
    cout << " " << quantity << " " << discount;
}

#endif
```

```c++
#include <iostream>
#include <algorithm>
#include <vector>
#include "Quote.h"
#include "Disc_quote.h"
#include "Bulk_quote.h"

int main() {
    vector<Quote> basket;
    basket.push_back(Quote("a", 10));
    basket.push_back(Bulk_quote("a", 10, 5, 0.1));
    double res(0.0);
    for_each(basket.begin(), basket.end(), [&res](Quote &p) {
        res += p.net_price(10);
    });
    cout << res << endl;
    return 0;
}
```

### Q29

当派生类对象被赋值给基类对象时，其中的派生类部分将被“切掉”，因此容器和存在继承关系的类型无法兼容；
当我们希望在容器中存放具有继承关系的对象时，我们实际上存放的通常是基类的指针。

```c++
// Quote.h
#ifndef QUOTE_H
#define QUOTE_H

#include <iostream>
#include <string>

using namespace std;

class Quote {
    friend double print_total(ostream&, const Quote&, size_t);
public:
    Quote() = default;
    Quote(const string &book, double sales_price) : bookNo(book), price(sales_price) {}
    string isbn() const { return bookNo; }
    virtual double net_price(size_t n) const { return n * price; }
    virtual ~Quote() = default;
    virtual void debug() const;
private:
    string bookNo;
protected:
    double price = 0.0;
};

double print_total(ostream &os, const Quote &item, size_t n) {
    double ret = item.net_price(n);
    os << "ISBN: " << item.isbn() << " # sold: " << n << " total due: " << ret << endl;
    return ret;
}

void Quote::debug() const {
    cout << bookNo << " " << price;
}

#endif
```

```c++
// Disc_quote.h
#ifndef DISC_QUOTE_H
#define DISC_QUOTE_H

#include <string>
#include "Quote.h"

using namespace std;

class Disc_quote : public Quote {
public:
    Disc_quote() = default;
    Disc_quote(const string &book, double p, size_t qty, double disc) : 
               Quote(book, p), quantity(qty), discount(disc) {}
    double net_price(size_t) const = 0;
protected:
    size_t quantity = 0;
    double discount = 0.0;
};

#endif
```

```c++
// Bulk_quote.h
#ifndef BULK_QUOTE_H
#define BULK_QUOTE_H

#include <iostream>
#include "Quote.h"

using namespace std;

class Bulk_quote : public Disc_quote {
public:
    // Bulk_quote() = default;
    // Bulk_quote(const string& book, double p, size_t qty, double disc) : 
    //            Quote(book, p), min_qty(qty), discount(disc) {}
    using Disc_quote::Disc_quote;
    double net_price(size_t) const override;
    void debug() const override;
};

double Bulk_quote::net_price(size_t cnt) const {
    if (cnt >= quantity) {
        return cnt * (1 - discount) * price;
    }
    else {
        return cnt * price;
    }
}

void Bulk_quote::debug() const {
    Quote::debug();
    cout << " " << quantity << " " << discount;
}

#endif
```

```c++
#include <iostream>
#include <algorithm>
#include <memory>
#include <vector>
#include "Quote.h"
#include "Disc_quote.h"
#include "Bulk_quote.h"

int main() {
    vector<shared_ptr<Quote>> basket;
    basket.push_back(make_shared<Quote>("a", 10));
    basket.push_back(make_shared<Bulk_quote>("a", 10, 5, 0.1));
    double res(0.0);
    for_each(basket.begin(), basket.end(), [&res](shared_ptr<Quote> p) {
        res += p->net_price(10);
    });
    cout << res << endl;
    return 0;
}
```

### Q30

```c++
// Quote.h
#ifndef QUOTE_H
#define QUOTE_H

#include <iostream>
#include <string>

using namespace std;

class Quote {
    friend double print_total(ostream&, const Quote&, size_t);
public:
    virtual Quote* clone() const & { return new Quote(*this); }
    virtual Quote* clone() && { return new Quote(std::move(*this)); }
    Quote() = default;
    Quote(const string &book, double sales_price) : bookNo(book), price(sales_price) {}
    string isbn() const { return bookNo; }
    virtual double net_price(size_t n) const { return n * price; }
    virtual ~Quote() = default;
    virtual void debug() const;
private:
    string bookNo;
protected:
    double price = 0.0;
};

double print_total(ostream &os, const Quote &item, size_t n) {
    double ret = item.net_price(n);
    os << "ISBN: " << item.isbn() << " # sold: " << n << " total due: " << ret << endl;
    return ret;
}

void Quote::debug() const {
    cout << bookNo << " " << price;
}

#endif
```

```c++
// Bulk_quote.h
#ifndef BULK_QUOTE_H
#define BULK_QUOTE_H

#include <iostream>
#include "Quote.h"

using namespace std;

class Bulk_quote : public Quote {
public:
    Bulk_quote* clone() const & { return new Bulk_quote(*this); }
    Bulk_quote* clone() && { return new Bulk_quote(std::move(*this)); }
    Bulk_quote() = default;
    Bulk_quote(const string& book, double p, size_t qty, double disc) : 
               Quote(book, p), min_qty(qty), discount(disc) {}
    double net_price(size_t) const override;
    void debug() const override;
private:
    size_t min_qty = 0;
    double discount = 0.0;
};

double Bulk_quote::net_price(size_t cnt) const {
    if (cnt >= min_qty) {
        return cnt * (1 - discount) * price;
    }
    else {
        return cnt * price;
    }
}

void Bulk_quote::debug() const {
    Quote::debug();
    cout << " " << min_qty << " " << discount;
}

#endif
```

```c++
// Basket.h
#ifndef BASKET_H
#define BASKET_H

#include <iostream>
#include <memory>
#include <set>
#include "Quote.h"

using namespace std;

class Basket {
public:
    void add_item(const Quote &sale) { items.insert(shared_ptr<Quote>(sale.clone())); }
    void add_item(Quote &&sale) { items.insert(shared_ptr<Quote>(std::move(sale).clone())); }
    double total_receipt(ostream&) const;
private:
    static bool compare(const shared_ptr<Quote> &lhs, const shared_ptr<Quote> &rhs) {
        return lhs->isbn() < rhs->isbn();
    }
    multiset<shared_ptr<Quote>, decltype(compare)*> items{compare};
};

double Basket::total_receipt(ostream &os) const {
    double sum = 0.0;
    for (auto iter = items.cbegin(); iter != items.cend(); iter = items.upper_bound(*iter)) {
        sum += print_total(os, **iter, items.count(*iter));
    }
    return sum;
}

#endif
```

```c++
#include <iostream>
#include "Quote.h"
#include "Bulk_quote.h"
#include "Basket.h"

int main() {
    Basket basket;
    basket.add_item(Quote{"a", 10});
    basket.add_item(Bulk_quote{"a", 10, 5, 0.1});
    basket.add_item(Quote{"b", 10});
    basket.add_item(Bulk_quote{"b", 10, 5, 0.1});
    basket.total_receipt(cout);
    return 0;
}
```

### Q31

(a) OrQuery, AndQuery, NotQuery, WordQuery
(b) OrQuery, AndQuery, NotQuery, WordQuery
(c) OrQuery, AndQuery, WordQuery

### Q32

Query类未定义自己的拷贝、移动、赋值或销毁操作，所以会执行相应的合成拷贝、移动、赋值或销毁操作；又因为只含有唯一数据成员Query_base的shared_ptr，所以最终会执行shared_ptr相应的合成拷贝、移动、赋值或销毁操作。

### Q33

Query_base是一个抽象类，所以这种类型的对象本质上是派生类的子对象。当派生类对象进行拷贝、移动、赋值或销毁操作时，会执行Query_base中的拷贝、移动、赋值或销毁操作，但因为Query_base中没定义这些操作且不存在数据成员，所以什么也不会发生。

### Q34

```c++
(a)
WordQuery(const string &) wind
Query(const string &) wind
WordQuery(const string &) bird
Query(const string &) bird
WordQuery(const string &) fiery
Query(const string &) fiery
BinaryQuery(const Query &, const Query &, string) (fiery, bird, &)
AndQuery(const Query &, const Query &, string) (fiery, bird)
Query(shared_ptr<Query_base>)
BinaryQuery(const Query &, const Query &, string) ((fiery, bird, &), wind, |)
OrQuery(const Query &, const Query &, string) ((fiery, bird, &), wind)
Query(shared_ptr<Query_base>)
(b)
Query::rep()
BinaryQuery::rep()
Query::rep()
WordQuery::rep()
Query::rep()
BinaryQuery::rep()
Query::rep()
WordQuery::rep()
Query::rep()
WordQuery::rep()
((fiery & bird) | wind)
(c)
Query::eval()
OrQuery::eval()
Query::eval()
Query::eval()
AndQuery::eval()
Query::eval()
WordQuery::eval()
Query::eval()
WordQuery::eval()
```

### Q35

```c++
#ifndef QUERY_H
#define QUERY_H

#include <iostream>
#include <memory>
#include <set>
#include <string>
#include "TextQuery.h"

using namespace std;

class Query_base {
    friend class Query;
protected:
    using line_no = TextQuery::line_no;
    virtual ~Query_base() = default;
private:
    virtual QueryResult eval(const TextQuery&) const = 0;
    virtual string rep() const = 0;
};

class Query {
    friend Query operator~(const Query&);
    friend Query operator|(const Query&, const Query&);
    friend Query operator&(const Query&, const Query&);
public:
    Query(const string&);
    QueryResult eval(const TextQuery &t) const { return q->eval(t); }
    string rep() const { return q->rep(); }
private:
    Query(shared_ptr<Query_base> query): q(query) {}
    shared_ptr<Query_base> q;
};
inline ostream& operator<<(ostream &os, const Query &query) {
    return os << query.rep();
}

class WordQuery : public Query_base {
    friend class Query;
    WordQuery(const string &s) : query_word(s) {}
    QueryResult eval(const TextQuery &t) const { return t.query(query_word); }
    string rep() const { return query_word; }
    string query_word;
};
inline Query::Query(const string &s) : q(new WordQuery(s)) {}

class NotQuery : public Query_base {
    friend Query operator~(const Query&);
    NotQuery(const Query &q) : query(q) {}
    string rep() const { return "~(" + query.rep() + ")"; }
    QueryResult eval(const TextQuery&) const;
    Query query;
};
inline Query operator~(const Query &operand) {
    return shared_ptr<Query_base>(new NotQuery(operand));
}
QueryResult NotQuery::eval(const TextQuery &text) const {
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
    return QueryResult(rep(), ret_lines, result.get_file());
}

class BinaryQuery : public Query_base {
protected:
    BinaryQuery(const Query &l, const Query &r, string s) : lhs(l), rhs(r), opSym(s) {}
    string rep() const { return "(" + lhs.rep() + " " + opSym + " " + rhs.rep() + ")"; }
    Query lhs, rhs;
    string opSym;
};

class AndQuery : public BinaryQuery {
    friend Query operator&(const Query&, const Query&);
    AndQuery(const Query &left, const Query &right) : BinaryQuery(left, right, "&") {}
    QueryResult eval(const TextQuery&) const;
};
inline Query operator&(const Query &lhs, const Query &rhs) {
    return shared_ptr<Query_base>(new AndQuery(lhs, rhs));
}
QueryResult AndQuery::eval(const TextQuery &text) const {
    auto right = rhs.eval(text), left = lhs.eval(text);
    auto ret_lines = make_shared<set<line_no>>();
    set_intersection(left.begin(), left.end(), right.begin(), right.end(), inserter(*ret_lines, ret_lines->begin()));
    return QueryResult(rep(), ret_lines, left.get_file());
}

class OrQuery : public BinaryQuery {
    friend Query operator|(const Query&, const Query&);
    OrQuery(const Query &left, const Query &right) : BinaryQuery(left, right, "|") {}
    QueryResult eval(const TextQuery&) const;
};
inline Query operator|(const Query &lhs, const Query &rhs) {
    return shared_ptr<Query_base>(new OrQuery(lhs, rhs));
}
QueryResult OrQuery::eval(const TextQuery &text) const {
    auto right = rhs.eval(text), left = lhs.eval(text);
    auto ret_lines = make_shared<set<line_no>>(left.begin(), left.end());
    ret_lines->insert(right.begin(), right.end());
    return QueryResult(rep(), ret_lines, left.get_file());
}

#endif
```

### Q36

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

#endif
```

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

class Query_base {
    friend class Query;
protected:
    using line_no = TextQuery::line_no;
    virtual ~Query_base() = default;
private:
    virtual QueryResult eval(const TextQuery&) const = 0;
    virtual string rep() const = 0;
};

class Query {
    friend Query operator~(const Query&);
    friend Query operator|(const Query&, const Query&);
    friend Query operator&(const Query&, const Query&);
public:
    Query(const string&);
    QueryResult eval(const TextQuery &t) const {
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
    QueryResult eval(const TextQuery &t) const {
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
    QueryResult eval(const TextQuery&) const;
    Query query;
};
inline Query operator~(const Query &operand) {
    return shared_ptr<Query_base>(new NotQuery(operand));
}
QueryResult NotQuery::eval(const TextQuery &text) const {
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
    return QueryResult(rep(), ret_lines, result.get_file());
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
    QueryResult eval(const TextQuery&) const;
};
inline Query operator&(const Query &lhs, const Query &rhs) {
    return shared_ptr<Query_base>(new AndQuery(lhs, rhs));
}
QueryResult AndQuery::eval(const TextQuery &text) const {
    auto right = rhs.eval(text), left = lhs.eval(text);
    auto ret_lines = make_shared<set<line_no>>();
    set_intersection(left.begin(), left.end(), right.begin(), right.end(), inserter(*ret_lines, ret_lines->begin()));
    return QueryResult(rep(), ret_lines, left.get_file());
}

class OrQuery : public BinaryQuery {
    friend Query operator|(const Query&, const Query&);
    OrQuery(const Query &left, const Query &right) : BinaryQuery(left, right, "|") {
        cout << "OrQuery(const Query &, const Query &, string)" << endl;
    }
    QueryResult eval(const TextQuery&) const;
};
inline Query operator|(const Query &lhs, const Query &rhs) {
    return shared_ptr<Query_base>(new OrQuery(lhs, rhs));
}
QueryResult OrQuery::eval(const TextQuery &text) const {
    auto right = rhs.eval(text), left = lhs.eval(text);
    auto ret_lines = make_shared<set<line_no>>(left.begin(), left.end());
    ret_lines->insert(right.begin(), right.end());
    return QueryResult(rep(), ret_lines, left.get_file());
}

#endif
```

```c++
#include <iostream>
#include <fstream>
#include "TextQuery.h"
#include "Query.h"

int main() {
    Query q = Query("fiery") & Query("bird") | Query("wind");
    cout << q << endl;
    return 0;
}
```

### Q37

涉及Query的地方均要改成Query_base指针

### Q38

非法。BinaryQuery是抽象类，不能定义对象。
非法。返回的为Query类型，不能转换为AndQuery。
非法。返回的为Query类型，不能转换为OrQuery。

### Q39

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

#endif
```

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

class Query_base {
    friend class Query;
protected:
    using line_no = TextQuery::line_no;
    virtual ~Query_base() = default;
private:
    virtual QueryResult eval(const TextQuery&) const = 0;
    virtual string rep() const = 0;
};

class Query {
    friend Query operator~(const Query&);
    friend Query operator|(const Query&, const Query&);
    friend Query operator&(const Query&, const Query&);
public:
    Query(const string&);
    QueryResult eval(const TextQuery &t) const {
        cout << "Query::eval()" << endl;
        return q->eval(t);
    }
    string rep() const { 
        return q->rep();
    }
private:
    Query(shared_ptr<Query_base> query): q(query) {
    }
    shared_ptr<Query_base> q;
};
inline ostream& operator<<(ostream &os, const Query &query) {
    return os << query.rep();
}

class WordQuery : public Query_base {
    friend class Query;
    WordQuery(const string &s) : query_word(s) {
    }
    QueryResult eval(const TextQuery &t) const {
        cout << "WordQuery::eval()" << endl;
        return t.query(query_word);
    }
    string rep() const {
        return query_word;
    }
    string query_word;
};
inline Query::Query(const string &s) : q(new WordQuery(s)) {
}

class NotQuery : public Query_base {
    friend Query operator~(const Query&);
    NotQuery(const Query &q) : query(q) {
    }
    string rep() const {
        return "~(" + query.rep() + ")";
    }
    QueryResult eval(const TextQuery&) const;
    Query query;
};
inline Query operator~(const Query &operand) {
    return shared_ptr<Query_base>(new NotQuery(operand));
}
QueryResult NotQuery::eval(const TextQuery &text) const {
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
    return QueryResult(rep(), ret_lines, result.get_file());
}

class BinaryQuery : public Query_base {
protected:
    BinaryQuery(const Query &l, const Query &r, string s) : lhs(l), rhs(r), opSym(s) {
    }
    string rep() const {
        return "(" + lhs.rep() + " " + opSym + " " + rhs.rep() + ")";
    }
    Query lhs, rhs;
    string opSym;
};

class AndQuery : public BinaryQuery {
    friend Query operator&(const Query&, const Query&);
    AndQuery(const Query &left, const Query &right) : BinaryQuery(left, right, "&") {
    }
    QueryResult eval(const TextQuery&) const;
};
inline Query operator&(const Query &lhs, const Query &rhs) {
    return shared_ptr<Query_base>(new AndQuery(lhs, rhs));
}
QueryResult AndQuery::eval(const TextQuery &text) const {
    cout << "AndQuery::eval()" << endl;
    auto right = rhs.eval(text), left = lhs.eval(text);
    auto ret_lines = make_shared<set<line_no>>();
    set_intersection(left.begin(), left.end(), right.begin(), right.end(), inserter(*ret_lines, ret_lines->begin()));
    return QueryResult(rep(), ret_lines, left.get_file());
}

class OrQuery : public BinaryQuery {
    friend Query operator|(const Query&, const Query&);
    OrQuery(const Query &left, const Query &right) : BinaryQuery(left, right, "|") {
    }
    QueryResult eval(const TextQuery&) const;
};
inline Query operator|(const Query &lhs, const Query &rhs) {
    return shared_ptr<Query_base>(new OrQuery(lhs, rhs));
}
QueryResult OrQuery::eval(const TextQuery &text) const {
    cout << "OrQuery::eval()" << endl;
    auto right = rhs.eval(text), left = lhs.eval(text);
    auto ret_lines = make_shared<set<line_no>>(left.begin(), left.end());
    ret_lines->insert(right.begin(), right.end());
    return QueryResult(rep(), ret_lines, left.get_file());
}

#endif
```

```c++
#include <iostream>
#include <fstream>
#include "TextQuery.h"
#include "Query.h"

int main() {
    Query q = Query("fiery") & Query("bird") | Query("wind");
    ifstream in("test.txt");
    q.eval(TextQuery(in));
    return 0;
}
```

### Q40

空集无影响。

### Q41

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

#endif
```

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

class Query_base {
    friend class Query;
protected:
    using line_no = TextQuery::line_no;
    virtual ~Query_base() = default;
private:
    virtual QueryResult eval(const TextQuery&) const = 0;
    virtual string rep() const = 0;
};

class Query {
    friend Query operator~(const Query&);
    friend Query operator|(const Query&, const Query&);
    friend Query operator&(const Query&, const Query&);
public:
    Query(const string&);
    Query(const Query &query) : q(query.q), use(query.use) { ++*use; }
    Query& operator=(const Query&);
    ~Query();
    QueryResult eval(const TextQuery &t) const { return q->eval(t); }
    string rep() const { return q->rep(); }
private:
    Query(Query_base *query): q(query), use(new size_t(1)) {}
    Query_base *q;
    size_t *use;
};
inline ostream& operator<<(ostream &os, const Query &query) {
    return os << query.rep();
}
inline Query& Query::operator=(const Query &query) {
    ++*query.use;
    if (--*use == 0) {
        delete q;
        delete use;
    }
    q = query.q;
    use = query.use;
    return *this;
}
inline Query::~Query() {
    if (--*use == 0) {
        delete q;
        delete use;
    }
}

class WordQuery : public Query_base {
    friend class Query;
    WordQuery(const string &s) : query_word(s) {}
    QueryResult eval(const TextQuery &t) const { return t.query(query_word); }
    string rep() const { return query_word; }
    string query_word;
};
inline Query::Query(const string &s) : q(new WordQuery(s)), use(new size_t(1)) {}

class NotQuery : public Query_base {
    friend Query operator~(const Query&);
    NotQuery(const Query &q) : query(q) {}
    string rep() const { return "~(" + query.rep() + ")"; }
    QueryResult eval(const TextQuery&) const;
    Query query;
};
inline Query operator~(const Query &operand) {
    return new NotQuery(operand);
}
QueryResult NotQuery::eval(const TextQuery &text) const {
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
    return QueryResult(rep(), ret_lines, result.get_file());
}

class BinaryQuery : public Query_base {
protected:
    BinaryQuery(const Query &l, const Query &r, string s) : lhs(l), rhs(r), opSym(s) {}
    string rep() const { return "(" + lhs.rep() + " " + opSym + " " + rhs.rep() + ")"; }
    Query lhs, rhs;
    string opSym;
};

class AndQuery : public BinaryQuery {
    friend Query operator&(const Query&, const Query&);
    AndQuery(const Query &left, const Query &right) : BinaryQuery(left, right, "&") {}
    QueryResult eval(const TextQuery&) const;
};
inline Query operator&(const Query &lhs, const Query &rhs) {
    return new AndQuery(lhs, rhs);
}
QueryResult AndQuery::eval(const TextQuery &text) const {
    auto right = rhs.eval(text), left = lhs.eval(text);
    auto ret_lines = make_shared<set<line_no>>();
    set_intersection(left.begin(), left.end(), right.begin(), right.end(), inserter(*ret_lines, ret_lines->begin()));
    return QueryResult(rep(), ret_lines, left.get_file());
}

class OrQuery : public BinaryQuery {
    friend Query operator|(const Query&, const Query&);
    OrQuery(const Query &left, const Query &right) : BinaryQuery(left, right, "|") {}
    QueryResult eval(const TextQuery&) const;
};
inline Query operator|(const Query &lhs, const Query &rhs) {
    return new OrQuery(lhs, rhs);
}
QueryResult OrQuery::eval(const TextQuery &text) const {
    auto right = rhs.eval(text), left = lhs.eval(text);
    auto ret_lines = make_shared<set<line_no>>(left.begin(), left.end());
    ret_lines->insert(right.begin(), right.end());
    return QueryResult(rep(), ret_lines, left.get_file());
}

#endif
```

```c++
#include <iostream>
#include <fstream>
#include "TextQuery.h"
#include "Query.h"

int main() {
    Query q = Query("fiery") & Query("bird") | Query("wind");
    ifstream in("test.txt");
    print(cout, q.eval(TextQuery(in)));
    return 0;
}
```

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
#include <cctype>

using namespace std;

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
    string sentence;
    char ch, prech;
    while (is.get(ch)) {
        sentence += ch;
        if (ch == '.' ||  (ch == '\"' && prech == '?')) {
            file->push_back(sentence);
            int n = file->size() - 1;
            istringstream line(sentence);
            string word;
            while (line >> word) {
                word = cleanup_str(word);
                auto &lines = wm[word];
                if (!lines)
                    lines.reset(new set<line_no>);
                lines->insert(n);
            }
            prech = ch;
            sentence = "";
        }
    }
    file->push_back(sentence);
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
        os << "\t(sentence " << num+1 << ") " << *(qr.file->begin()+num) << endl;
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

#endif
```

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

class Query_base {
    friend class Query;
protected:
    using line_no = TextQuery::line_no;
    virtual ~Query_base() = default;
private:
    virtual QueryResult eval(const TextQuery&) const = 0;
    virtual string rep() const = 0;
};

class Query {
    friend Query operator~(const Query&);
    friend Query operator|(const Query&, const Query&);
    friend Query operator&(const Query&, const Query&);
public:
    Query(const string&);
    Query(const Query &query) : q(query.q), use(query.use) { ++*use; }
    Query& operator=(const Query&);
    ~Query();
    QueryResult eval(const TextQuery &t) const { return q->eval(t); }
    string rep() const { return q->rep(); }
private:
    Query(Query_base *query): q(query), use(new size_t(1)) {}
    Query_base *q;
    size_t *use;
};
inline ostream& operator<<(ostream &os, const Query &query) {
    return os << query.rep();
}
inline Query& Query::operator=(const Query &query) {
    ++*query.use;
    if (--*use == 0) {
        delete q;
        delete use;
    }
    q = query.q;
    use = query.use;
    return *this;
}
inline Query::~Query() {
    if (--*use == 0) {
        delete q;
        delete use;
    }
}

class WordQuery : public Query_base {
    friend class Query;
    WordQuery(const string &s) : query_word(s) {}
    QueryResult eval(const TextQuery &t) const { return t.query(query_word); }
    string rep() const { return query_word; }
    string query_word;
};
inline Query::Query(const string &s) : q(new WordQuery(s)), use(new size_t(1)) {}

class NotQuery : public Query_base {
    friend Query operator~(const Query&);
    NotQuery(const Query &q) : query(q) {}
    string rep() const { return "~(" + query.rep() + ")"; }
    QueryResult eval(const TextQuery&) const;
    Query query;
};
inline Query operator~(const Query &operand) {
    return new NotQuery(operand);
}
QueryResult NotQuery::eval(const TextQuery &text) const {
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
    return QueryResult(rep(), ret_lines, result.get_file());
}

class BinaryQuery : public Query_base {
protected:
    BinaryQuery(const Query &l, const Query &r, string s) : lhs(l), rhs(r), opSym(s) {}
    string rep() const { return "(" + lhs.rep() + " " + opSym + " " + rhs.rep() + ")"; }
    Query lhs, rhs;
    string opSym;
};

class AndQuery : public BinaryQuery {
    friend Query operator&(const Query&, const Query&);
    AndQuery(const Query &left, const Query &right) : BinaryQuery(left, right, "&") {}
    QueryResult eval(const TextQuery&) const;
};
inline Query operator&(const Query &lhs, const Query &rhs) {
    return new AndQuery(lhs, rhs);
}
QueryResult AndQuery::eval(const TextQuery &text) const {
    auto right = rhs.eval(text), left = lhs.eval(text);
    auto ret_lines = make_shared<set<line_no>>();
    set_intersection(left.begin(), left.end(), right.begin(), right.end(), inserter(*ret_lines, ret_lines->begin()));
    return QueryResult(rep(), ret_lines, left.get_file());
}

class OrQuery : public BinaryQuery {
    friend Query operator|(const Query&, const Query&);
    OrQuery(const Query &left, const Query &right) : BinaryQuery(left, right, "|") {}
    QueryResult eval(const TextQuery&) const;
};
inline Query operator|(const Query &lhs, const Query &rhs) {
    return new OrQuery(lhs, rhs);
}
QueryResult OrQuery::eval(const TextQuery &text) const {
    auto right = rhs.eval(text), left = lhs.eval(text);
    auto ret_lines = make_shared<set<line_no>>(left.begin(), left.end());
    ret_lines->insert(right.begin(), right.end());
    return QueryResult(rep(), ret_lines, left.get_file());
}

#endif
```

```c++
#include <iostream>
#include <fstream>
#include "TextQuery.h"
#include "Query.h"

int main() {
    Query q = Query("fiery") & Query("bird") | Query("wind");
    ifstream in("test.txt");
    print(cout, q.eval(TextQuery(in)));
    return 0;
}
```