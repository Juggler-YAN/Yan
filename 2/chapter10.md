# Chapter 10

### Q1

```c++
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

int main() {
    vector<int> v{1,1,2,2,5,3,1,1};
    cout << count(v.cbegin(), v.cend(), 1) << endl;
    return 0;
}
```

### Q2

```c++
#include <iostream>
#include <algorithm>
#include <list>
#include <string>

using namespace std;

int main() {
    list<string> l = {"a", "b", "c", "d", "a"};
    cout << count(l.cbegin(), l.cend(), "a") << endl;
    return 0;
}
```

### Q3

```c++
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

int main() {
    vector<int> v{1,2,3,4,5};
    cout << accumulate(v.cbegin(), v.cend(), 0) << endl;
    return 0;
}
```

### Q4

结果会转换成int类型

```c++
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

int main() {
    vector<double> v{1.1,2.2,3.3,4.4,5.5};
    cout << accumulate(v.cbegin(), v.cend(), 0) << endl;
    return 0;
}
```

### Q5

equal会比较指针地址，而不是字符串值

### Q6

```c++
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

int main() {
    vector<int> v{1,1,2,2,5,3,1,1};
    fill_n(v.begin(), v.size(), 0);
    for (const auto &i : v) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

### Q7

```c++
vector<int> vec; list<int> lst; int i;
while (cin >> i)
    lst.push_back(i);
copy(lst.cbegin(), lst.cend(), back_inserter(vec));
```

```c++
vector<int> vec;
vec.reserve(10);
fill_n(back_inserter(vec), 10, 0);
```

### Q8

是迭代器改变它们所操作的容器的大小，而不是标准库算法

### Q9

```c++
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;

void elimDups(vector<string>&);

int main() {
    vector<string> words;
    string s;
    while (cin >> s) {
        words.push_back(s);
    }
    elimDups(words);
    return 0;
}

void elimDups(vector<string> &words) {
    for (const auto &s : words) {
        cout << s << " ";
    }
    cout << endl;
    sort(words.begin(), words.end());
    auto end_unique = unique(words.begin(), words.end());
    for (const auto &s : words) {
        cout << s << " ";
    }
    cout << endl;
    words.erase(end_unique, words.end());
    for (const auto &s : words) {
        cout << s << " ";
    }
    cout << endl;
}
```

### Q10

使算法更加通用，改变容器大小可能会导致迭代器失效，而且改变不同容器大小可能需要使用不同的方法

### Q11

```c++
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;

void elimDups(vector<string>&);
bool isShorter(const string &, const string &);

int main() {
    vector<string> words;
    string s;
    while (cin >> s) {
        words.push_back(s);
    }
    elimDups(words);
    stable_sort(words.begin(), words.end(), isShorter);
    for (const auto &s : words) {
        cout << s << " ";
    }
    cout << endl;
    
    return 0;
}

void elimDups(vector<string> &words) {
    sort(words.begin(), words.end());
    auto end_unique = unique(words.begin(), words.end());
    words.erase(end_unique, words.end());
}

bool isShorter(const string &s1, const string &s2) {
    return s1.size() < s2.size();
}
```

### Q12

```
0-201-70353-X 4 24.99
0-201-82470-1 4 45.39
0-201-88954-4 2 15.00
0-399-82477-1 2 45.39
0-201-78345-X 3 20.00
```

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
#include <algorithm>
#include <vector>
#include "Sales_data.h"

using namespace std;

bool compareIsbn(const Sales_data &, const Sales_data &);

int main() {
    vector<Sales_data> sales;
    Sales_data sale;
    while (read(cin, sale)) {
        sales.push_back(sale);
    }
    sort(sales.begin(), sales.end(), compareIsbn);
    for (const auto &i : sales) {
        print(cout, i) << endl;
    }
    return 0;
}

bool compareIsbn(const Sales_data &sale1, const Sales_data &sale2) {
    return sale1.isbn() < sale2.isbn();
}
```

### Q13

```c++
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;

bool isgreater5(const string &);

int main() {
	vector<string> s{"a","aa","aaa","aaaa","aaaaa","aaaaaa"};
    auto iter = partition(s.begin(), s.end(), isgreater5);
    for (auto i = s.begin(); i != iter; i++) {
        cout << *i << " ";
    }
    cout << endl;
	return 0;
}

bool isgreater5(const string &s) {
    return s.size() >= 5;
}
```

### Q14

```c++
#include <iostream>

using namespace std;

int main() {
    auto add = [](int a, int b) { return a+b; };
    cout << add(1,2) << endl;
    return 0;
}
```

### Q15

```c++
#include <iostream>

using namespace std;

int main() {
    int a = 1;
    auto add = [a](int b) { return a+b; };
    cout << add(2) << endl;
    return 0;
}
```

### Q16

```c++
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;

void elimDups(vector<string> &);
void biggies(vector<string> &, vector<string>::size_type);

int main() {
    vector<string> words;
    string s;
    while (cin >> s) {
        words.push_back(s);
    }
    biggies(words, 4);
    return 0;
}

void elimDups(vector<string> &words) {
    sort(words.begin(), words.end());
    auto end_unique = unique(words.begin(), words.end());
    words.erase(end_unique, words.end());
}

void biggies(vector<string> &words, vector<string>::size_type sz) {
    elimDups(words);
    stable_sort(words.begin(), words.end(), 
                [](const string &a, const string &b) { return a.size() < b.size(); });
    auto wc = find_if(words.begin(), words.end(), 
                [sz](const string &a) { return a.size() >= sz; });
    for_each(wc, words.end(), [](const string &s){ cout << s << " "; });
}
```

### Q17

```
0-201-70353-X 4 24.99
0-201-82470-1 4 45.39
0-201-88954-4 2 15.00
0-399-82477-1 2 45.39
0-201-78345-X 3 20.00
```

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
#include <algorithm>
#include <vector>
#include "Sales_data.h"

using namespace std;

int main() {
    vector<Sales_data> sales;
    Sales_data sale;
    while (read(cin, sale)) {
        sales.push_back(sale);
    }
    sort(sales.begin(), sales.end(),
         [](const Sales_data &sale1, const Sales_data &sale2) {return sale1.isbn() < sale2.isbn();});
    for (const auto &i : sales) {
        print(cout, i) << endl;
    }
    return 0;
}
```

### Q18

```c++
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;

void elimDups(vector<string> &);
void biggies(vector<string> &, vector<string>::size_type);

int main() {
    vector<string> words;
    string s;
    while (cin >> s) {
        words.push_back(s);
    }
    biggies(words, 4);
    return 0;
}

void elimDups(vector<string> &words) {
    sort(words.begin(), words.end());
    auto end_unique = unique(words.begin(), words.end());
    words.erase(end_unique, words.end());
}

void biggies(vector<string> &words, vector<string>::size_type sz) {
    elimDups(words);
    stable_sort(words.begin(), words.end(), 
                [](const string &a, const string &b) { return a.size() < b.size(); });
    auto wc = partition(words.begin(), words.end(), 
                        [sz](const string &a) { return a.size() >= sz; });
    for_each(words.begin(), wc, [](const string &s){ cout << s << " "; });
}
```

### Q19

```c++
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;

void elimDups(vector<string> &);
void biggies(vector<string> &, vector<string>::size_type);

int main() {
    vector<string> words;
    string s;
    while (cin >> s) {
        words.push_back(s);
    }
    biggies(words, 4);
    return 0;
}

void elimDups(vector<string> &words) {
    sort(words.begin(), words.end());
    auto end_unique = unique(words.begin(), words.end());
    words.erase(end_unique, words.end());
}

void biggies(vector<string> &words, vector<string>::size_type sz) {
    elimDups(words);
    stable_sort(words.begin(), words.end(), 
                [](const string &a, const string &b) { return a.size() < b.size(); });
    auto wc = stable_partition(words.begin(), words.end(), 
                              [sz](const string &a) { return a.size() >= sz; });
    for_each(words.begin(), wc, [](const string &s){ cout << s << " "; });
}
```

### Q20

```c++
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;

int main() {
	vector<string> s{"a","aa","aaa","aaaa","aaaaa","aaaaaa"};
    string::size_type sz = 6;
    cout << count_if(s.begin(), s.end(), 
                 [sz](const string &s) { return s.size() >= 6; }) << endl;
	return 0;
}
```

### Q21

```c++
#include <iostream>

using namespace std;

int main() {
    int n = 3;
    auto f = [n]() mutable -> bool {
        if (n) {
            --n;
        }
        return n;
    };
    cout << f() << endl;
    cout << f() << endl;
    cout << f() << endl;
    cout << f() << endl;
    cout << f() << endl;
    return 0;
}
```

### Q22

```c++
#include <iostream>
#include <algorithm>
#include <functional>
#include <vector>
#include <string>

using namespace std;

bool issmaller(const string &, string::size_type);

int main() {
	vector<string> s{"a","aa","aaa","aaaa","aaaaa","aaaaaa"};
    string::size_type sz = 6;
    cout << count_if(s.begin(), s.end(), bind(issmaller, placeholders::_1, sz)) << endl;
	return 0;
}

bool issmaller(const string &s, string::size_type sz) {
    return s.size() <= sz;
}
```

### Q23

n+1个，要绑定的函数以及要绑定的函数的n个参数

### Q24

```c++
#include <iostream>
#include <algorithm>
#include <functional>
#include <vector>
#include <string>

using namespace std;

bool fun(int, const string &);

int main() {
    string s{"hello"};
    vector<int> v{1,3,5,8,9,7};
    cout << *(find_if(v.begin(), v.end(), bind(fun, placeholders::_1, cref(s)))) << endl;
    return 0;
}

bool fun(int i, const string &s) {
    return i > s.size();
}
```

### Q25

```c++
#include <iostream>
#include <algorithm>
#include <functional>
#include <vector>
#include <string>

using namespace std;

void elimDups(vector<string> &);
bool check_size(const string &, string::size_type);
void biggies(vector<string> &, vector<string>::size_type);

int main() {
    vector<string> words;
    string s;
    while (cin >> s) {
        words.push_back(s);
    }
    biggies(words, 4);
    return 0;
}

void elimDups(vector<string> &words) {
    sort(words.begin(), words.end());
    auto end_unique = unique(words.begin(), words.end());
    words.erase(end_unique, words.end());
}

bool check_size(const string &s, string::size_type sz) {
    return s.size() >= sz;
}

void biggies(vector<string> &words, vector<string>::size_type sz) {
    elimDups(words);
    stable_sort(words.begin(), words.end(), 
                [](const string &a, const string &b) { return a.size() < b.size(); });
    auto wc = partition(words.begin(), words.end(), bind(check_size, placeholders::_1, sz));
    for_each(words.begin(), wc, [](const string &s){ cout << s << " "; });
}
```

### Q26

1. back_inserter创建一个使用push_back的迭代器；
2. front_inserter创建一个使用push_front的迭代器；
3. inserter创建一个使用insert的迭代器。此函数接受第二个参数，这个参数必须是一个指向给定容器的迭代器。元素将被插入到给定迭代器所表示的元素之前。

### Q27

```c++
#include <iostream>
#include <algorithm>
#include <vector>
#include <list>

using namespace std;

int main() {
    vector<int> v{1,1,2,3,5,8};
    list<int> l;
    unique_copy(v.cbegin(), v.cend(), inserter(l, l.begin()));
    for (const auto &i : l) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

### Q28

1. front_inserter：反序
2. inserter：正序
3. back_inserter：正序

```c++
#include <iostream>
#include <algorithm>
#include <vector>
#include <list>

using namespace std;

int main() {
    vector<int> v{0,1,2,3,4,5,6,7,8,9};
    list<int> l1, l2, l3;
    copy(v.cbegin(), v.cend(), front_inserter(l1));
    copy(v.cbegin(), v.cend(), inserter(l2, l2.begin()));
    copy(v.cbegin(), v.cend(), back_inserter(l3));
    for (const auto &i : l1) {
        cout << i << " ";
    }
    cout << endl;
    for (const auto &i : l2) {
        cout << i << " ";
    }
    cout << endl;
    for (const auto &i : l3) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

### Q29

```c++
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>

using namespace std;

int main() {
    ifstream ifs("./test.txt");
    istream_iterator<string> in_iter(ifs), eof;
    vector<string> v(in_iter, eof);
    ostream_iterator<string> out_iter(cout, " ");
    copy(v.begin(), v.end(), out_iter);
    cout << endl;
    ifs.close();
    return 0;
}
```

### Q30

```c++
#include <iostream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <string>

using namespace std;

int main() {
    istream_iterator<int> in_iter(cin), eof;
    vector<int> v(in_iter, eof);
    sort(v.begin(), v.end(), [](int x, int y) { return x < y; });
    ostream_iterator<int> out_iter(cout, " ");
    copy(v.begin(), v.end(), out_iter);
    cout << endl;
    return 0;
}
```

### Q31

```c++
#include <iostream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <string>

using namespace std;

int main() {
    istream_iterator<int> in_iter(cin), eof;
    vector<int> v(in_iter, eof);
    sort(v.begin(), v.end(), [](int x, int y) { return x < y; });
    ostream_iterator<int> out_iter(cout, " ");
    unique_copy(v.begin(), v.end(), out_iter);
    cout << endl;
    return 0;
}
```

### Q32

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
/*
 * This file contains code from "C++ Primer, Fifth Edition", by Stanley B.
 * Lippman, Josee Lajoie, and Barbara E. Moo, and is covered under the
 * copyright and warranty notices given in that book:
 * 
 * "Copyright (c) 2013 by Objectwrite, Inc., Josee Lajoie, and Barbara E. Moo."
 * 
 * 
 * "The authors and publisher have taken care in the preparation of this book,
 * but make no expressed or implied warranty of any kind and assume no
 * responsibility for errors or omissions. No liability is assumed for
 * incidental or consequential damages in connection with or arising out of the
 * use of the information or programs contained herein."
 * 
 * Permission is granted for this code to be used for educational purposes in
 * association with the book, given proper citation if and when posted or
 * reproduced.Any commercial use of this code requires the explicit written
 * permission of the publisher, Addison-Wesley Professional, a division of
 * Pearson Education, Inc. Send your request for permission, stating clearly
 * what code you would like to use, and in what specific way, to the following
 * address: 
 * 
 *     Pearson Education, Inc.
 *     Rights and Permissions Department
 *     One Lake Street
 *     Upper Saddle River, NJ  07458
 *     Fax: (201) 236-3290
*/ 

/* This file defines the Sales_item class used in chapter 1.
 * The code used in this file will be explained in
 * Chapter 7 (Classes) and Chapter 14 (Overloaded Operators)
 * Readers shouldn't try to understand the code in this file
 * until they have read those chapters.
*/

#ifndef SALESITEM_H
// we're here only if SALESITEM_H has not yet been defined 
#define SALESITEM_H

// Definition of Sales_item class and related functions goes here
#include <iostream>
#include <string>

class Sales_item {
// these declarations are explained section 7.2.1, p. 270 
// and in chapter 14, pages 557, 558, 561
friend std::istream& operator>>(std::istream&, Sales_item&);
friend std::ostream& operator<<(std::ostream&, const Sales_item&);
friend bool operator<(const Sales_item&, const Sales_item&);
friend bool 
operator==(const Sales_item&, const Sales_item&);
public:
    // constructors are explained in section 7.1.4, pages 262 - 265
    // default constructor needed to initialize members of built-in type
    Sales_item(): units_sold(0), revenue(0.0) { }
    Sales_item(const std::string &book): 
                  bookNo(book), units_sold(0), revenue(0.0) { }
    Sales_item(std::istream &is) { is >> *this; }
public:
    // operations on Sales_item objects
    // member binary operator: left-hand operand bound to implicit this pointer
    Sales_item& operator+=(const Sales_item&);
    
    // operations on Sales_item objects
    std::string isbn() const { return bookNo; }
    double avg_price() const;
// private members as before
private:
    std::string bookNo;      // implicitly initialized to the empty string
    unsigned units_sold;
    double revenue;
};

// used in chapter 10
inline
bool compareIsbn(const Sales_item &lhs, const Sales_item &rhs) 
{ return lhs.isbn() == rhs.isbn(); }

// nonmember binary operator: must declare a parameter for each operand
Sales_item operator+(const Sales_item&, const Sales_item&);

inline bool 
operator==(const Sales_item &lhs, const Sales_item &rhs)
{
    // must be made a friend of Sales_item
    return lhs.units_sold == rhs.units_sold &&
           lhs.revenue == rhs.revenue &&
           lhs.isbn() == rhs.isbn();
}

inline bool 
operator!=(const Sales_item &lhs, const Sales_item &rhs)
{
    return !(lhs == rhs); // != defined in terms of operator==
}

// assumes that both objects refer to the same ISBN
Sales_item& Sales_item::operator+=(const Sales_item& rhs) 
{
    units_sold += rhs.units_sold; 
    revenue += rhs.revenue; 
    return *this;
}

// assumes that both objects refer to the same ISBN
Sales_item 
operator+(const Sales_item& lhs, const Sales_item& rhs) 
{
    Sales_item ret(lhs);  // copy (|lhs|) into a local object that we'll return
    ret += rhs;           // add in the contents of (|rhs|) 
    return ret;           // return (|ret|) by value
}

std::istream& 
operator>>(std::istream& in, Sales_item& s)
{
    double price;
    in >> s.bookNo >> s.units_sold >> price;
    // check that the inputs succeeded
    if (in)
        s.revenue = s.units_sold * price;
    else 
        s = Sales_item();  // input failed: reset object to default state
    return in;
}

std::ostream& 
operator<<(std::ostream& out, const Sales_item& s)
{
    out << s.isbn() << " " << s.units_sold << " "
        << s.revenue << " " << s.avg_price();
    return out;
}

double Sales_item::avg_price() const
{
    if (units_sold) 
        return revenue/units_sold; 
    else 
        return 0;
}
#endif
```

```c++
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <vector>
#include "Sales_item.h"

using namespace std;

int main() {
    istream_iterator<Sales_item> item_iter(cin), eof;
    vector<Sales_item> v(item_iter, eof);
    sort(v.begin(), v.end(), compareIsbn);
    ostream_iterator<Sales_item> out_iter(cout, "\n");
    copy(v.begin(), v.end(), out_iter);
    cout << "^^^^^^^^^" << endl;
    auto v1 = v.cbegin(), v2 = v1, v3 = v1;
    auto val = v1->isbn();
    auto v4 = v.cend();
    while (true) {
        v2 = v1;
        v3 = v1;
        val = v1->isbn();
        while ((v2 = find_if(v1, v4, [v3](const Sales_item &sale) { return sale.isbn() == v3->isbn(); })) != v4) {
            v1 = v2 + 1;
        };
        cout << accumulate(v3, v1, Sales_item(v3->isbn())) << endl;
        if (v1 == v4) {
            break;
        }
    }
    return 0;
}
```

### Q33

```c++
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>

using namespace std;

int main(int argc, char * argv[]) {
    ifstream in(argv[1]);
    ofstream out1(argv[2]), out2(argv[3]);
    istream_iterator<int> in_iter(in), eof;
    vector<int> v(in_iter, eof);
    ostream_iterator<int> out_iter1(out1, " ");
    ostream_iterator<int> out_iter2(out2, " ");
    for (const auto &i : v) {
        (i % 2) ? *out_iter1++ = i : *out_iter2++ = i;
    }
    in.close();
    out1.close();
    out2.close();
    return 0;
}
```

### Q34

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> v{2,3,5,8,9,7,4,5};
    for (auto r_iter = v.crbegin(); r_iter != v.crend(); ++r_iter) {
        cout << *r_iter << " ";
    }
    cout << endl;
    return 0;
}
```

### Q35

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> v{2,3,5,8,9,7,4,5};
    for (auto r_iter = v.cend() - 1; r_iter != v.cbegin() - 1; --r_iter) {
        cout << *r_iter << " ";
    }
    cout << endl;
    return 0;
}
```

### Q36

```c++
#include <iostream>
#include <algorithm>
#include <list>

using namespace std;

int main() {
    list<int> l{2,0,3,5,8,9,7,4,0,5};
    auto rval = find(l.crbegin(), l.crend(), 0);
    cout << distance(rval, l.crend()) << endl;
    return 0;
}
```

### Q37

```c++
#include <iostream>
#include <algorithm>
#include <vector>
#include <list>

using namespace std;

int main() {
    vector<int> v{2,0,3,5,8,9,7,4,0,5};
    list<int> l(v.crbegin()+2, v.crbegin()+7);
    for (const auto &i : l) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

### Q38

1. 输入迭代器：==、!=、++、*（解引用）、->；
2. 输出迭代器：++、*（解引用）；
3. 前项迭代器：==、!=、++、*（解引用）、->；
4. 双向迭代器：==、!=、++、--、*（解引用）、->；
5. 随机访问迭代器：==、!=、++、--、*（解引用）、->、<、<=、>、>=、+、+=、-、-=、-（迭代器相减）、iter[n]。

### Q39

1. 双向迭代器
2. 随机访问迭代器

### Q40

1. copy：输入迭代器、输入迭代器、输出迭代器
2. reverse：双向迭代器
3. unique：前向迭代器

### Q41

1. 在beg迭代器和end迭代器之间，如果值为old_val，则替换为new_val
2. 在beg迭代器和end迭代器之间，如果值old_val满足谓词条件，则替换为new_val
3. 在beg迭代器和end迭代器之间，如果值为old_val，则替换为new_val，将结果拷贝到dest
4. 在beg迭代器和end迭代器之间，如果值old_val满足谓词条件，则替换为new_val，将结果拷贝到dest

### Q42

```c++
#include <iostream>
#include <algorithm>
#include <list>
#include <string>

using namespace std;

void elimDups(list<string>&);

int main() {
    list<string> words;
    string s;
    while (cin >> s) {
        words.push_back(s);
    }
    elimDups(words);
    for (const auto &s : words) {
        cout << s << " ";
    }
    cout << endl;
    
    return 0;
}

void elimDups(list<string> &words) {
    words.sort();
    words.unique();
}
```