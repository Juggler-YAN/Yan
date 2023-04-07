### Q1

编译：g++ test.cpp -o test
运行：./test

```c++
int main() {
    return 0;
}
```

### Q2

编译：g++ test.cpp -o test
运行：./test

```c++
int main() {
    return -1;
}
```

### Q3

```c++
#include <iostream>

using namespace std;

int main() {
    cout << "hello,world" << endl;
    return 0;
}
```

### Q4

```c++
#include <iostream>

using namespace std;

int main() {
    cout << "Enter two nums:" << endl;
    int v1 = 0, v2 = 0;
    cin >> v1 >> v2;
    cout << "The product of " << v1 << " and " << v2 <<
    " is " << v1*v2 << endl;
    return 0;
}
```

### Q5

```c++
#include <iostream>

using namespace std;

int main() {
    cout << "Enter two nums:" << endl;
    int v1 = 0, v2 = 0;
    cin >> v1 >> v2;
    cout << "The product of ";
    cout << v1;
    cout << " and ";
    cout << v2;
    cout << " is ";
    cout << v1*v2;
    cout << endl;
    return 0;
}
```

### Q6

不合法，C++中分号代表语句的结束，应该去掉分号。

```c++
#include <iostream>

using namespace std;

int main() {
    cout << "Enter two nums:" << endl;
    int v1 = 0, v2 = 0;
    cin >> v1 >> v2;
    cout << "The product of " << v1
        << " and " << v2
        << " is " << v1+v2 << endl;
    return 0;
}
```

### Q7

```c++
/*/**/*/
```

### Q8

1，2，4是合法的

```c++
#include <iostream>

using namespace std;

int main() {
    cout << "/*";
    cout << "*/";
    // cout << /* "*/" */;
    cout << /* "*/" */";
    cout << /* "*/" /* "/*" */;
    return 0;
}
```

### Q9

```c++
#include <iostream>

using namespace std;

int main() {
    int sum = 0, val = 50;
    while (val <= 100) {
        sum += val;
        val++;
    }
    cout << "Sum of 50 to 100 inclusive is " << sum << endl;
    return 0;
}
```

### Q10

```c++
#include <iostream>

using namespace std;

int main() {
    int val = 10;
    while (val >= 0) {
        cout << val << " ";
        val--;
    }
    cout << endl;
    return 0;
}
```

### Q11

```c++
#include <iostream>

using namespace std;

int main() {
    int v1 = 0, v2 = 0;
    cout << "Enter two nums(The former is smaller than the latter):" << endl;
    cin >> v1 >> v2;
    while (v1 <= v2) {
        cout << v1 << " ";
        v1++;
    }
    cout << endl;
    return 0;
}
```

### Q12

求-100到100中所有整数之和，终值为0

```c++
#include <iostream>

using namespace std;

int main() {
    int sum = 0;
    for (int i = -100; i <= 100; i++)
        sum += i;
    cout << sum << endl;
    return 0;
}
```

### Q13

```c++
#include <iostream>

using namespace std;

int main() {
    int sum = 0;
    for (int val = 50; val <= 100; val++) {
        sum += val;
    }
    cout << "Sum of 50 to 100 inclusive is " << sum << endl;
    return 0;
}
```

```c++
#include <iostream>

using namespace std;

int main() {
    for (int val = 10; val >= 0; val--) {
        cout << val << " ";
    }
    cout << endl;
    return 0;
}
```

```c++
#include <iostream>

using namespace std;

int main() {
    int v1 = 0, v2 = 0;
    cout << "Enter two nums(The former is smaller than the latter):" << endl;
    cin >> v1 >> v2;
    for (; v1 <= v2; v1++) {
        cout << v1 << " ";
    }
    cout << endl;
    return 0;
}
```

### Q14

已知迭代次数用for循环，否则用while循环

### Q15

略

### Q16

```c++
#include <iostream>

using namespace std;

int main() {
    int sum = 0, val = 0;
    while (cin >> val) {
        sum += val;
    }
    cout << "Sum is: " << sum << endl;
    return 0;
}
```

### Q17

该值的出现次数等于输入的值数；每个值只出现一次。

### Q18

```c++
#include <iostream>

using namespace std;

int main() {
    int currval = 0, val = 0;
    if (cin >> currval) {
        int cnt = 1;
        while (cin >> val) {
            if (val == currval) {
                cnt++;
            }
            else {
                cout << currval << " occurs " << cnt <<
                " times " << endl;
                currval = val;
                cnt = 1;
            }
        }
        cout << currval << " occurs " << cnt <<
        " times " << endl;
    }
    return 0;
}
```

### Q19

```c++
#include <iostream>

using namespace std;

int main() {
    int v1 = 0, v2 = 0;
    cout << "Enter two numbers:" << endl;
    cin >> v1 >> v2;
    if (v1 > v2) {
        int temp;
        temp = v1;
        v1 = v2;
        v2 = temp;
    }
    while (v1 <= v2) {
        cout << v1 << " ";
        ++v1;
    }
    cout << endl;
    return 0;
}
```

### Q20

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
// Sales_item.h
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
#include "Sales_item.h"

using namespace std;

int main() {
    Sales_item book;
    while (cin >> book) {
        cout << book << endl;
    }
    return 0;
}
```

### Q21

```
0-201-78345-X 3 20.00
0-201-78345-X 2 25.00
```

```c++
// Sales_item.h
// 见Q20
```

```c++
#include <iostream>
#include "Sales_item.h"

using namespace std;

int main() {
    Sales_item item1, item2;
    cin >> item1 >> item2;
    cout << item1 + item2 << endl;
    return 0;
}
```

### Q22

```
0-201-78345-X 3 20.00
0-201-78345-X 3 20.00
0-201-78345-X 2 25.00
0-201-78345-X 2 25.00
```

```c++
// Sales_item.h
// 见Q20
```

```c++
#include <iostream>
#include "Sales_item.h"

using namespace std;

int main() {
    Sales_item sum, item;
    if (cin >> sum) {
        while (cin >> item) {
            sum += item;
        }
    }
    cout << sum << endl;
    return 0;
}
```

### Q23

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
// Sales_item.h
// 见Q20
```

```c++
#include <iostream>
#include "Sales_item.h"

using namespace std;

int main() {
    Sales_item sum, item;
    if (cin >> sum) {
        while (cin >> item) {
            if (sum.isbn() == item.isbn()) {
                sum += item;
            }
            else {
                cout << sum << endl;
                sum = item;
            }
        }
        cout << sum << endl;
    }
    return 0;
}
```

### Q24

见Q23

### Q25

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
// Sales_item.h
// 见Q20
```

```c++
#include <iostream>
#include "Sales_item.h"

int main() 
{
    Sales_item total; // variable to hold data for the next transaction

    // read the first transaction and ensure that there are data to process
    if (std::cin >> total) {
		Sales_item trans; // variable to hold the running sum
        // read and process the remaining transactions
        while (std::cin >> trans) {
			// if we're still processing the same book
            if (total.isbn() == trans.isbn()) 
                total += trans; // update the running total 
            else {              
		        // print results for the previous book 
                std::cout << total << std::endl;  
                total = trans;  // total now refers to the next book
            }
		}
        std::cout << total << std::endl; // print the last transaction
    } else {
        // no input! warn the user
        std::cerr << "No data?!" << std::endl;
        return -1;  // indicate failure
    }

    return 0;
}
```