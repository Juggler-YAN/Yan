# Chapter 7

### Q1

```
0-201-70353-X 4 24.99
0-201-82470-1 4 45.39
0-201-88954-4 2 15.00 
0-201-88954-4 5 12.00 
0-201-88954-4 7 12.00 
0-201-88954-4 2 12.00 
0-399-82477-1 2 45.39
0-399-82477-1 3 45.39
0-201-78345-X 3 20.00
0-201-78345-X 2 25.00
```

```c++
// Sales_data.h
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <string>

using namespace std;

struct Sales_data {
    string bookNo;
    unsigned units_sold;
    double revenue;
};
#endif
```

```c++
#include <iostream>
#include <string>
#include "Sales_data.h"

using namespace std;

int main() {
    double allprice, num;
    Sales_data sum;
    if (cin >> sum.bookNo >> sum.units_sold >> sum.revenue) {
        Sales_data item;
        allprice = sum.units_sold * sum.revenue;
        num = sum.units_sold;
        while (cin >> item.bookNo >> item.units_sold >> item.revenue) {
            if (sum.bookNo == item.bookNo) {
                allprice += item.units_sold * item.revenue;
                num += item.units_sold;
            }
            else {
                cout << sum.bookNo << " " << num << " " << allprice << " " << allprice/num << " " << endl;
                sum.bookNo = item.bookNo;
                sum.units_sold = item.units_sold;
                sum.revenue = item.revenue;
                allprice = sum.units_sold * sum.revenue;
                num = sum.units_sold;
            }
        }
        cout << sum.bookNo << " " << num << " " << allprice << " " << allprice/num << " " << endl;
    }
    else {
        cerr << "No data?!" << endl;
        return -1;
    }
    return 0;
}
```

### Q2

```c++
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <string>

using namespace std;

struct Sales_data {

    string bookNo;
    unsigned units_sold;
    double revenue;

    string isbn() const { return bookNo; }
    Sales_data& combine(const Sales_data&);

};

Sales_data& Sales_data::combine(const Sales_data& rhs) {
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}

#endif
```

### Q3

```
0-201-70353-X 4 24.99
0-201-82470-1 4 45.39
0-201-88954-4 2 15.00 
0-201-88954-4 5 12.00 
0-201-88954-4 7 12.00 
0-201-88954-4 2 12.00 
0-399-82477-1 2 45.39
0-399-82477-1 3 45.39
0-201-78345-X 3 20.00
0-201-78345-X 2 25.00
```

```c++
#include <iostream>
#include <string>
#include "Sales_data.h"

using namespace std;

int main() {
    double allprice, num;
    Sales_data sum;
    if (cin >> sum.bookNo >> sum.units_sold >> sum.revenue) {
        Sales_data item;
        allprice = sum.units_sold * sum.revenue;
        num = sum.units_sold;
        while (cin >> item.bookNo >> item.units_sold >> item.revenue) {
            if (sum.isbn() == item.isbn()) {
                sum.combine(item);
            }
            else {
                cout << sum.bookNo << " " << num << " " << allprice << " " << allprice/num << " " << endl;
                sum.bookNo = item.bookNo;
                sum.units_sold = item.units_sold;
                sum.revenue = item.revenue;
                allprice = sum.units_sold * sum.revenue;
                num = sum.units_sold;
            }
        }
        cout << sum.bookNo << " " << num << " " << allprice << " " << allprice/num << " " << endl;
    }
    else {
        cerr << "No data?!" << endl;
        return -1;
    }
    return 0;
}
```

### Q4

```c++
#ifndef PERSON_H
#define PERSON_H

#include <string>

using namespace std;

struct Person {
    string name;
    string address;
};

#endif
```

### Q5

应该是const，因为无需改变成员对象

```c++
#ifndef PERSON_H
#define PERSON_H

#include <string>

using namespace std;

struct Person {
    string name;
    string address;
    string getName() const { return name; }
    string getAddress() const { return address; }
};

#endif
```

### Q6

```c++
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

Sales_data add(const Sales_data &lhs, const Sales_data &rhs) {
	Sales_data sum = lhs;
	sum.combine(rhs);
	return sum;
}
```

### Q7

```
0-201-70353-X 4 24.99
0-201-82470-1 4 45.39
0-201-88954-4 2 15.00 
0-201-88954-4 5 12.00 
0-201-88954-4 7 12.00 
0-201-88954-4 2 12.00 
0-399-82477-1 2 45.39
0-399-82477-1 3 45.39
0-201-78345-X 3 20.00
0-201-78345-X 2 25.00
```

```c++
// Sales_data.h
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <string>

using namespace std;

struct Sales_data {

    string bookNo;
    unsigned units_sold;
    double revenue;

    string isbn() const { return bookNo; }
    Sales_data& combine(const Sales_data&);
    double avg_price() const;

};

Sales_data& Sales_data::combine(const Sales_data& rhs) {
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}

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

Sales_data add(const Sales_data &lhs, const Sales_data &rhs) {
	Sales_data sum = lhs;
	sum.combine(rhs);
	return sum;
}

#endif
```

```c++
#include <iostream>
#include "Sales_data.h"

using namespace std;

int main() {
    Sales_data sum;
    if (read(cin, sum)) {
        Sales_data item;
        while (read(cin, item)) {
            if (sum.isbn() == item.isbn()) {
                sum.combine(item);
            }
            else {
                print(cout, sum) << endl;
                sum = item;
            }
        }
        print(cout, sum) << endl;
    }
    else {
        cerr << "No data?!" << endl;
        return -1;
    }
    return 0;
}
```

### Q8

因为read函数需要改变成员对象；而print不需要。

### Q9

```c++
#ifndef PERSON_H
#define PERSON_H

#include <string>

using namespace std;

struct Person {
    string name;
    string address;
    string getName() const { return name; }
    string getAddress() const { return address; }
};

istream& read(istream& is, Person& item) {
    is >> item.name >> item.address;
    return is;
}

ostream& print(ostream& os, const Person& item) {
    os << item.name << " " << item.address;
    return os;
}

#endif
```

### Q10

判断读入data1和data2是否成功

### Q11

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
    Sales_data& combine(const Sales_data&);
    double avg_price() const;

    Sales_data() = default;
    Sales_data(const string &s) : bookNo(s) {}
    Sales_data(const string &s, unsigned n, double p) : 
               bookNo(s), units_sold(n), revenue(p*n) {}
    Sales_data(istream &);

};

Sales_data& Sales_data::combine(const Sales_data& rhs) {
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}

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

Sales_data add(const Sales_data &lhs, const Sales_data &rhs) {
	Sales_data sum = lhs;
	sum.combine(rhs);
	return sum;
}

Sales_data::Sales_data(istream &is) {
	read(is, *this);
}

#endif
```

```c++
#include <iostream>
#include <string>
#include "Sales_data.h"

using namespace std;

int main() {
	Sales_data sales_data1;
	print(cout, sales_data1) << endl;

	Sales_data sales_data2("A");
	print(cout, sales_data2) << endl;

	Sales_data sales_data3("A", 1, 2);
	print(cout, sales_data3) << endl;

	Sales_data sales_data4(cin);
	print(cout, sales_data4) << endl;

	return 0;
}
```

### Q12

```c++
// Sales_data.h
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <string>

using namespace std;

struct Sales_data;

istream &read(istream &, Sales_data &);
ostream &print(ostream &, const Sales_data &);
Sales_data add(const Sales_data &, const Sales_data &);

struct Sales_data {

    string bookNo;
    unsigned units_sold = 0;
    double revenue = 0.0;

    string isbn() const { return bookNo; }
    Sales_data& combine(const Sales_data&);
    double avg_price() const;

    Sales_data() = default;
    Sales_data(const string &s) : bookNo(s) {}
    Sales_data(const string &s, unsigned n, double p) : 
               bookNo(s), units_sold(n), revenue(p*n) {}
    Sales_data(istream &is) { read(is, *this); };

};

Sales_data& Sales_data::combine(const Sales_data& rhs) {
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}

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

Sales_data add(const Sales_data &lhs, const Sales_data &rhs) {
	Sales_data sum = lhs;
	sum.combine(rhs);
	return sum;
}

#endif
```

```c++
#include <iostream>
#include <string>
#include "Sales_data.h"

using namespace std;

int main() {
	Sales_data sales_data1;
	print(cout, sales_data1) << endl;

	Sales_data sales_data2("A");
	print(cout, sales_data2) << endl;

	Sales_data sales_data3("A", 1, 2);
	print(cout, sales_data3) << endl;

	Sales_data sales_data4(cin);
	print(cout, sales_data4) << endl;

	return 0;
}
```

### Q13

```
0-201-70353-X 4 24.99
0-201-82470-1 4 45.39
0-201-88954-4 2 15.00 
0-201-88954-4 5 12.00 
0-201-88954-4 7 12.00 
0-201-88954-4 2 12.00 
0-399-82477-1 2 45.39
0-399-82477-1 3 45.39
0-201-78345-X 3 20.00
0-201-78345-X 2 25.00
```

```c++
// Sales_data.h
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <string>

struct Sales_data {

    string bookNo;
    unsigned units_sold = 0;
    double revenue = 0.0;

    string isbn() const { return bookNo; }
    Sales_data& combine(const Sales_data&);
    double avg_price() const;

    Sales_data() = default;
    Sales_data(const string &s) : bookNo(s) {}
    Sales_data(const string &s, unsigned n, double p) : 
               bookNo(s), units_sold(n), revenue(p*n) {}
    Sales_data(istream &);

};

Sales_data& Sales_data::combine(const Sales_data& rhs) {
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}

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

Sales_data add(const Sales_data &lhs, const Sales_data &rhs) {
	Sales_data sum = lhs;
	sum.combine(rhs);
	return sum;
}

Sales_data::Sales_data(istream &is) {
	read(is, *this);
}

#endif
```

```c++
#include <iostream>
#include "Sales_data.h"

using namespace std;

int main() {
    Sales_data sum;
    if (read(cin, sum)) {
        Sales_data item;
        while (read(cin, item)) {
            if (sum.isbn() == item.isbn()) {
                sum.combine(item);
            }
            else {
                print(cout, sum) << endl;
                sum = item;
            }
        }
        print(cout, sum) << endl;
    }
    else {
        cerr << "No data?!" << endl;
        return -1;
    }
    return 0;
}
```

### Q14

```c++
Sales_data() : bookNo(""), units_sold(0), revenue(0) {}
```

### Q15

```c++
#ifndef PERSON_H
#define PERSON_H

#include <string>

using namespace std;

struct Person {
    string name{""};
    string address{""};
    string getName() const { return name; }
    string getAddress() const { return address; }
    Person() = default;
    Person(const string & n, const string & a) : name(n), address(a) {}
    Person(istream &is) { read(is, *this); }
};

istream& read(istream& is, Person& item) {
    is >> item.name >> item.address;
    return is;
}

ostream& print(ostream& os, const Person& item) {
    os << item.name << " " << item.address;
    return os;
}

#endif
```

### Q16

一个类可以包含0个或多个访问说明符，而且对于某个访问说明符能出现多少次也没有严格的限定。public：成员在整个程序内可被访问，public成员定义类的接口；private：成员可以被类的成员函数访问，但是不能被使用该类的代码访问，private部分封装了（即隐藏了）类的实现细节。

### Q17

struct默认的访问权限是public；class默认的访问权限是private。

### Q18

封装是实现与接口的分离,隐藏了类型的实现细节。1.确保用户不会无意间破坏封装对象的状态；2.被封装的类的具体实现细节可以随时改变，而无须调整用户级别的代码。

### Q19

接口定义成公共的，数据定义成私密的

```c++
struct Person {
private:
    string name{""};
    string address{""};
public:
    string getName() const { return name; }
    string getAddress() const { return address; }
    Person() = default;
    Person(const string & n, const string & a) : name(n), address(a) {}
    Person(istream &is) { read(is, *this); }
};
```

### Q20

类可以允许其他类或者函数访问它的非公有成员，方法是令其他类或者函数成为它的友元。优点：外部函数可以方便地使用类的成员，而不需要显示地给它们加上类名；可以方便地访问所有非公有成员；有时，对类的用户更容易读懂。缺点：减少封装和可维护性；代码冗长，类内的声明，类外函数声明。

### Q21

```
0-201-70353-X 4 24.99
0-201-82470-1 4 45.39
0-201-88954-4 2 15.00 
0-201-88954-4 5 12.00 
0-201-88954-4 7 12.00 
0-201-88954-4 2 12.00 
0-399-82477-1 2 45.39
0-399-82477-1 3 45.39
0-201-78345-X 3 20.00
0-201-78345-X 2 25.00
```

```c++
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <string>

using namespace std;

struct Sales_data {

    friend istream &read(istream &, Sales_data &);
    friend ostream &print(ostream &, const Sales_data &);
    friend Sales_data add(const Sales_data &, const Sales_data &);

private:
    string bookNo;
    unsigned units_sold = 0;
    double revenue = 0.0;

public:
    string isbn() const { return bookNo; }
    Sales_data& combine(const Sales_data&);
    double avg_price() const;

    Sales_data() = default;
    Sales_data(const string &s) : bookNo(s) {}
    Sales_data(const string &s, unsigned n, double p) : 
               bookNo(s), units_sold(n), revenue(p*n) {}
    Sales_data(istream &);

};

Sales_data& Sales_data::combine(const Sales_data& rhs) {
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}

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

Sales_data add(const Sales_data &lhs, const Sales_data &rhs) {
	Sales_data sum = lhs;
	sum.combine(rhs);
	return sum;
}

Sales_data::Sales_data(istream &is) {
	read(is, *this);
}

#endif
```

```c++
#include <iostream>
#include "Sales_data.h"

using namespace std;

int main() {
    Sales_data sum;
    if (read(cin, sum)) {
        Sales_data item;
        while (read(cin, item)) {
            if (sum.isbn() == item.isbn()) {
                sum.combine(item);
            }
            else {
                print(cout, sum) << endl;
                sum = item;
            }
        }
        print(cout, sum) << endl;
    }
    else {
        cerr << "No data?!" << endl;
        return -1;
    }
    return 0;
}
```

### Q22

```c++
#ifndef PERSON_H
#define PERSON_H

#include <string>

using namespace std;

struct Person {
    friend istream& read(istream&, Person&);
    friend ostream& print(ostream&, const Person&);
private:
    string name{""};
    string address{""};
public:
    string getName() const { return name; }
    string getAddress() const { return address; }
    Person() = default;
    Person(const string & n, const string & a) : name(n), address(a) {}
    Person(istream &is) { read(is, *this); }
};

istream& read(istream& is, Person& item) {
    is >> item.name >> item.address;
    return is;
}

ostream& print(ostream& os, const Person& item) {
    os << item.name << " " << item.address;
    return os;
}

#endif
```

### Q23

```c++
#ifndef SCREEN_H
#define SCREEN_H

#include <string>

using namespace std;

class Screen {
    public:
        using pos = string::size_type;

        Screen() = default;
        Screen(pos ht, pos wd, char c):height(ht), width(wd), contents(ht*wd, c){ }

        char get() const { return contents[cursor]; }
        inline char get(pos ht, pos wt) const;
        Screen &move(pos r, pos c);

    private:
        pos cursor = 0;
        pos height = 0, width = 0;
        string contents;
};


inline Screen &Screen::move(pos r, pos c) {
	pos row = r * width;
	cursor = row + c;
	return *this;
}

inline char Screen::get(pos r, pos c) const {
    pos row = r * width;
    return contents[row + c];
}

#endif
```

### Q24

```c++
#ifndef SCREEN_H
#define SCREEN_H

#include <string>

using namespace std;

class Screen {
    public:
        using pos = string::size_type;

        Screen() = default;
        Screen(pos ht, pos wd) : height(ht), width(wd) {}
        Screen(pos ht, pos wd, char c) : height(ht), width(wd), contents(ht*wd, c) {}

        char get() const { return contents[cursor]; }
        inline char get(pos ht, pos wt) const;
        Screen &move(pos r, pos c);

    private:
        pos cursor = 0;
        pos height = 0, width = 0;
        string contents;
};


inline Screen &Screen::move(pos r, pos c) {
	pos row = r * width;
	cursor = row + c;
	return *this;
}

inline char Screen::get(pos r, pos c) const {
    pos row = r * width;
    return contents[row + c];
}

#endif
```

### Q25

能，Screen类中只有内置类型和string，都可以使用拷贝和赋值操作

### Q26

```c++
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <string>

using namespace std;

struct Sales_data {

    friend istream &read(istream &, Sales_data &);
    friend ostream &print(ostream &, const Sales_data &);
    friend Sales_data add(const Sales_data &, const Sales_data &);

private:
    string bookNo;
    unsigned units_sold = 0;
    double revenue = 0.0;

public:
    string isbn() const { return bookNo; }
    Sales_data& combine(const Sales_data&);
    inline double avg_price() const;

    Sales_data() = default;
    Sales_data(const string &s) : bookNo(s) {}
    Sales_data(const string &s, unsigned n, double p) : 
               bookNo(s), units_sold(n), revenue(p*n) {}
    Sales_data(istream &);

};

Sales_data& Sales_data::combine(const Sales_data& rhs) {
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}

inline double Sales_data::avg_price() const {
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

Sales_data add(const Sales_data &lhs, const Sales_data &rhs) {
	Sales_data sum = lhs;
	sum.combine(rhs);
	return sum;
}

Sales_data::Sales_data(istream &is) {
	read(is, *this);
}

#endif
```

### Q27

```c++
#ifndef SCREEN_H
#define SCREEN_H

#include <string>

using namespace std;

class Screen {
    public:
        using pos = string::size_type;

        Screen() = default;
        Screen(pos ht, pos wd) : height(ht), width(wd) {}
        Screen(pos ht, pos wd, char c) : height(ht), width(wd), contents(ht*wd, c) {}

        char get() const { return contents[cursor]; }
        inline char get(pos ht, pos wt) const;
        Screen &move(pos r, pos c);
        Screen &set(char);
        Screen &set(pos, pos, char);
        Screen &display(ostream &os) { do_display(os); return *this; }
        const Screen &display(ostream &os) const { do_display(os); return *this; }

    private:
        pos cursor = 0;
        pos height = 0, width = 0;
        string contents;
        void do_display(ostream &os) const {os << contents;}
};


inline Screen &Screen::move(pos r, pos c) {
	pos row = r * width;
	cursor = row + c;
	return *this;
}

inline char Screen::get(pos r, pos c) const {
    pos row = r * width;
    return contents[row + c];
}

inline Screen &Screen::set(char c) {
    contents[cursor] = c;
    return *this;
}

inline Screen &Screen::set(pos r, pos col, char ch) {
    contents[r*width + col] = ch;
    return *this;
}

#endif
```

```c++
#include <iostream>
#include "Screen.h"

using namespace std;

int main() {
    Screen myScreen(5, 5, 'X');
    myScreen.move(4,0).set('#').display(cout);
    cout << "\n";
    myScreen.display(cout);
    cout << "\n";
}
```

### Q28

```
XXXXXXXXXXXXXXXXXXXX#XXXX
XXXXXXXXXXXXXXXXXXXX#XXXX
```

```
XXXXXXXXXXXXXXXXXXXX#XXXX
XXXXXXXXXXXXXXXXXXXXXXXXX
```

move、set和display返回的是Screen的临时副本，后续set和display操作并不会改变myScreen。

### Q29

```c++
#ifndef SCREEN_H
#define SCREEN_H

#include <string>

using namespace std;

class Screen {
    public:
        using pos = string::size_type;

        Screen() = default;
        Screen(pos ht, pos wd) : height(ht), width(wd) {}
        Screen(pos ht, pos wd, char c) : height(ht), width(wd), contents(ht*wd, c) {}

        char get() const { return contents[cursor]; }
        inline char get(pos ht, pos wt) const;
        Screen move(pos r, pos c);
        Screen set(char);
        Screen set(pos, pos, char);
        Screen display(ostream &os) { do_display(os); return *this; }
        const Screen display(ostream &os) const { do_display(os); return *this; }

    private:
        pos cursor = 0;
        pos height = 0, width = 0;
        string contents;
        void do_display(ostream &os) const {os << contents;}
};


inline Screen Screen::move(pos r, pos c) {
	pos row = r * width;
	cursor = row + c;
	return *this;
}

inline char Screen::get(pos r, pos c) const {
    pos row = r * width;
    return contents[row + c];
}

inline Screen Screen::set(char c) {
    contents[cursor] = c;
    return *this;
}

inline Screen Screen::set(pos r, pos col, char ch) {
    contents[r*width + col] = ch;
    return *this;
}

#endif
```

```c++
#include <iostream>
#include "Screen.h"

using namespace std;

int main() {
    Screen myScreen(5, 5, 'X');
    myScreen.move(4,0).set('#').display(cout);
    cout << "\n";
    myScreen.display(cout);
    cout << "\n";
}
```

### Q30

优点是意义更加明确，缺点是增加了冗余代码

### Q31

```c++
#ifndef CLASSXY_H
#define CLASSXY_H

class Y;

class X {
    Y *y;
};
class Y {
    X x;
};

#endif
```

### Q32

```c++
#ifndef SCREEN_H
#define SCREEN_H

#include <string>
#include <vector>

using namespace std;

class Screen;

class Window_mgr {
public:
    using ScreenIndex = vector<Screen>::size_type;
    void clear(ScreenIndex);
private:
    vector<Screen> screens;
};

class Screen {

friend void Window_mgr::clear(ScreenIndex);

public:
    using pos = string::size_type;

    Screen() = default;
    Screen(pos ht, pos wd) : height(ht), width(wd) {}
    Screen(pos ht, pos wd, char c) : height(ht), width(wd), contents(ht*wd, c) {}

    char get() const { return contents[cursor]; }
    inline char get(pos ht, pos wt) const;
    Screen &move(pos r, pos c);
    Screen &set(char);
    Screen &set(pos, pos, char);
    Screen &display(ostream &os) { do_display(os); return *this; }
    const Screen &display(ostream &os) const { do_display(os); return *this; }

private:
    pos cursor = 0;
    pos height = 0, width = 0;
    string contents;
    void do_display(ostream &os) const {os << contents;}
};


inline Screen &Screen::move(pos r, pos c) {
	pos row = r * width;
	cursor = row + c;
	return *this;
}

inline char Screen::get(pos r, pos c) const {
    pos row = r * width;
    return contents[row + c];
}

inline Screen &Screen::set(char c) {
    contents[cursor] = c;
    return *this;
}

inline Screen &Screen::set(pos r, pos col, char ch) {
    contents[r*width + col] = ch;
    return *this;
}

void Window_mgr::clear(ScreenIndex i) {
    Screen &s = screens[i];
    s.contents = string(s.height * s.width, ' ');
}

#endif
```

### Q33

```c++
#ifndef SCREEN_H
#define SCREEN_H

#include <string>
#include <vector>

using namespace std;

class Screen;

class Window_mgr {
public:
    using ScreenIndex = vector<Screen>::size_type;
    void clear(ScreenIndex);
private:
    vector<Screen> screens;
};

class Screen {

friend void Window_mgr::clear(ScreenIndex);

public:
    using pos = string::size_type;

    Screen() = default;
    Screen(pos ht, pos wd) : height(ht), width(wd) {}
    Screen(pos ht, pos wd, char c) : height(ht), width(wd), contents(ht*wd, c) {}

    char get() const { return contents[cursor]; }
    inline char get(pos ht, pos wt) const;
    Screen &move(pos r, pos c);
    Screen &set(char);
    Screen &set(pos, pos, char);
    Screen &display(ostream &os) { do_display(os); return *this; }
    const Screen &display(ostream &os) const { do_display(os); return *this; }
    pos size() const;

private:
    pos cursor = 0;
    pos height = 0, width = 0;
    string contents;
    void do_display(ostream &os) const {os << contents;}
};


inline Screen &Screen::move(pos r, pos c) {
	pos row = r * width;
	cursor = row + c;
	return *this;
}

inline char Screen::get(pos r, pos c) const {
    pos row = r * width;
    return contents[row + c];
}

inline Screen &Screen::set(char c) {
    contents[cursor] = c;
    return *this;
}

inline Screen &Screen::set(pos r, pos col, char ch) {
    contents[r*width + col] = ch;
    return *this;
}

inline Screen::pos Screen::size() const {
    return height * width;
}

void Window_mgr::clear(ScreenIndex i) {
    Screen &s = screens[i];
    s.contents = string(s.height * s.width, ' ');
}

#endif
```

### Q34

报错，dummy_fcn(pos height)中的pos未声明

### Q35

```c++
typedef string Type;
Type initVal(); // string
class Exercise {
public:
    typedef double Type;
    Type setVal(Type); // double,double
    Type initVal(); // double
private:
    int val;
};

Type Exercise::setVal(Type parm) {  // string,double
    val = parm + initVal();     // Exercise::initVal()
    return val;
}
```

```c++
typedef string Type;
Type initVal();
class Exercise {
public:
    typedef double Type;
    Type setVal(Type);
    Type initVal();
private:
    int val;
};

Exercise::Type Exercise::setVal(Type parm) {
    val = parm + initVal();
    return val;
}
```

### Q36

成员的初始化顺序与它们在类定义中的出现顺序一致，会先初始化rem再初始化base，而初始化rem时会用到base，所以会出错。

```c++
struct X {
    X (int i, int j): base(i), rem(i % j) {}
    int base, rem;
};
```

### Q37

```c++
Sales_data first_item(cin);   // Sales_data(istream &is): 值取决于你的输入

int main() {
  Sales_data next;  // Sales_data(string s = ""): bookNo = "", cnt = 0, revenue = 0.0
  Sales_data last("9-999-99999-9"); // Sales_data(string s = ""): bookNo = "9-999-99999-9", cnt = 0, revenue = 0.0
}
```

### Q38

```c++
Sales_data(istream &is = cin) { read(is, *this); }
```

### Q39

非法。重载构造函数Sale_data()将不明确。

### Q40

```c++
#ifndef BOOK_H
#define BOOK_H

#include <string>

using namespace std;

class Book {

public:
    Book() = default;
    Book(unsigned int a, string b, string c) : 
        no(a), name(b), author(c) {}
    Book(istream &is) { is >> no >> name >> author; }

private:
    unsigned int no;
    string name;
    string author;

};

#endif
```

### Q41

```c++
// Sales_data.h
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <string>

struct Sales_data {

    friend std::istream &read(std::istream &, Sales_data &);
    friend std::ostream &print(std::ostream &, const Sales_data &);
    friend Sales_data add(const Sales_data &, const Sales_data &);

private:
    std::string bookNo;
    unsigned units_sold = 0;
    double revenue = 0.0;

public:
    std::string isbn() const { return bookNo; }
    Sales_data& combine(const Sales_data&);
    inline double avg_price() const;

    Sales_data(std::string s, unsigned n, double p) :
                bookNo(s), units_sold(n), revenue(p*n) {
                    std::cout << "Sales_data(const std::string &s, unsigned n, double p)" << std::endl;
                };
    Sales_data() : Sales_data("", 0, 0) {
        std::cout << "Sales_data()" << std::endl;
    }
    Sales_data(std::string s) : Sales_data(s, 0, 0) {
        std::cout << "Sales_data(std::string s)" << std::endl;
    }
    Sales_data(std::istream &is) : Sales_data() {
        read(is, *this);
        std::cout << "Sales_data(std::istream &is)" << std::endl;
    }

};

Sales_data& Sales_data::combine(const Sales_data& rhs) {
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}

inline double Sales_data::avg_price() const {
	if (units_sold) {
		return revenue / units_sold;
    }
	else {
		return 0;
    }
}

std::istream &read(std::istream &is, Sales_data &item) {
	double price = 0;
	is >> item.bookNo >> item.units_sold >> price;
	item.revenue = price * item.units_sold;
	return is;
}

std::ostream &print(std::ostream &os, const Sales_data &item) {
	os << item.isbn() << " " << item.units_sold << " " << item.revenue << " " << item.avg_price();
	return os;
}

Sales_data add(const Sales_data &lhs, const Sales_data &rhs) {
	Sales_data sum = lhs;
	sum.combine(rhs);
	return sum;
}

#endif
```

```c++
#include <iostream>
#include "Sales_data.h"

int main() {
    Sales_data item1;
    Sales_data item2("A");
    Sales_data item3("A",0,0);
    Sales_data item4(std::cin);
    return 0;
}
```

### Q42

```c++
#ifndef BOOK_H
#define BOOK_H

#include <string>

class Book {

public:
    Book(unsigned int a, std::string b, std::string c) : 
        no(a), name(b), author(c) {}
    Book() : Book(0,"","") {};
    Book(std::istream &is) : Book() { is >> no >> name >> author; }

private:
    unsigned int no;
    std::string name;
    std::string author;

};

#endif
```

### Q43

```c++
class NoDefault {
public:
    NoDefault(int) {};
};
class C {
public:
    C() : my_mem(0) {}
private:
    NoDefault my_mem;
};
```

### Q44

非法，因为NoDefault没有默认构造函数。

### Q45

合法，C有默认构造函数。

### Q46

1. （a）不正确，没有构造函数时，有时可以生成默认构造函数；
2. （b）不正确，默认构造函数是没有构造函数的情况下，由编译器生成的构造函数；
3. （c）不正确，默认构造函数在一些情况下非常重要；
4. （d）不正确，当类没有显式地定义构造函数时，编译器才会隐式地定义默认构造函数。

### Q47

应该
1. 优点：防止隐式转换
2. 缺点：只对单个参数的构造函数有效

### Q48

初始化均无误

### Q49

1. （a）无误；
2. （b）有误，combine的参数是非常量的引用，所以我们不能将临时参数传递给它，改成Sales_data &combine(const Sales_data&)后正确；
3. （c）有误，函数名后的const不对，this是可变的。

### Q50

```c++
#ifndef PERSON_H
#define PERSON_H

#include <string>

struct Person {
    friend std::istream& read(std::istream&, Person&);
    friend std::ostream& print(std::ostream&, const Person&);
private:
    std::string name{""};
    std::string address{""};
public:
    std::string getName() const { return name; }
    std::string getAddress() const { return address; }
    Person() = default;
    Person(const std::string & n, const std::string & a) : name(n), address(a) {}
    explicit Person(std::istream &is) { read(is, *this); }
};

std::istream& read(std::istream& is, Person& item) {
    is >> item.name >> item.address;
    return is;
}

std::ostream& print(std::ostream& os, const Person& item) {
    os << item.name << " " << item.address;
    return os;
}

#endif
```

### Q51

string接受的单参数是const char*类型，如果我们得到了一个常量指针，则把它看做string对象是自然而然的过程，编译器自动把参数类型转换成类类型也非常符合逻辑，因此我们无须指定为explicit。与string相反，vector接受的单参数是int类型，这个参数的原意是指定vector的容量。如果我们在本来需要vector的地方提供一个int值并且希望这个int值自动转换成vector，则这个过程显得比较牵强，因此把vector的单参数构造函数定义成explicit的更加合理。

### Q52

聚合类初始化

```c++
struct Sales_data {
    string bookNo;
    unsigned units_sold;
    double revenue;
};
```

### Q53

```c++
#ifndef DEBUG_H
#define DEBUG_H

class Debug {
public:
    constexpr Debug(bool b = true) : hw(b), io(b), other(b) {}
    constexpr Debug(bool h, bool i, bool o) : hw(h), io(h), other(o) {}
    constexpr bool any() { return hw || io || other; }
    void set_hw(bool b) { hw = b; }
    void set_io(bool b) { io = b; }
    void set_other(bool b) { other = b; }
private:
    bool hw;
    bool io;
    bool other;
};

#endif
```

### Q54

constexpr函数是隐式const的

### Q55

不是，string不是字面值类型

### Q56

1. 类的静态成员与类本身直接相关，而不是与类的各个对象保持关联。
2. 优点：每个对象不需要存储公共数据，如果数据被改变，则每个对象都可以使用新值。
3. 静态数据成员可以是不完全类型；可以使用静态成员作为默认实参。

### Q57

```c++
#ifndef ACCOUNT_H
#define ACCOUNT_H

#include <string>

class Account {
public:
	void calculate() { amount += amount * interestRate; }
	static double rate() { return interestRate; }
	static void rate(double newRate) { interestRate = newRate; }
private:
	std::string owner;
	double amount;
	static double interestRate;
	static double initRate(){ return 4.0; }
};
double Account::interestRate = initRate();

#endif
```

### Q58

```c++
// example.h
class Example {
public:
    static double rate;
    static const int vecSize = 20;
    static vector<double> vec;
};
```

```c++
// example.C
#include "example.h"
double Example::rate = 6.5;
vector<double> Example::vec(vecSize);

int main() {
    Sales_data item = {"9999",36,15.88};
    return 0;
}
```