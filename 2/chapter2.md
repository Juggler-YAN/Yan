# Chapter 2

### Q1

1. 占用空间和表示范围不同
2. 有符号首位为符号位，所以两者表示范围不同
3. 具体实现和精度不同

### Q2

double。利率、本金和付款需要精确到小数点后两位，所以选用浮点数表示，用浮点数首选double类型。

### Q3

```c++
#include <iostream>

using namespace std;

int main() {
    unsigned u = 10, u2 = 42;
    cout << u2 - u << endl;
    cout << u - u2 << endl;
    int i = 10, i2 = 42;
    cout << i2 - i << endl;
    cout << i - i2 << endl;
    cout << i - u << endl;
    cout << u - i << endl;
    return  0;
}
```

### Q4

见Q3

### Q5

1. 字符型字面值，宽字符型字面值，字符串字面值，宽字符串字面值
2. 整形字面值，无符号整形字面值，长整形字面值，无符号长整形字面值，八进制整形字面值，十六进制整形字面值
3. 浮点型字面值，单精度浮点型字面值，扩展精度浮点型字面值
4. 整形字面值，无符号整形字面值，浮点型字面值，浮点型字面值

### Q6

有,第一组为十进制，第二组为八进制，另外"09"是错误的，八进制中不存在"9"

```c++
#include <iostream>

using namespace std;

int main() {
    int month = 9, day = 7;
    cout << month << " " << day << endl;
    month = 09, day = 07;
    cout << month << " " << day << endl;
    return 0;
}
```

### Q7

1. 字符串字面值
2. 扩展精度浮点型字面值
3. 错误
4. 扩展精度浮点型字面值


```c++
#include <iostream>

using namespace std;

int main() {
    cout << "Who goes with F\145rgus?\012" << endl;
    cout << 3.14e1L << endl;
    // cout << 1024f << endl;
    cout << 3.14L << endl;
    return 0;
}
```

### Q8

```c++
#include <iostream>

using namespace std;

int main() {
    cout << "2M" << endl;
    return 0;
}
```

```c++
#include <iostream>

using namespace std;

int main() {
    cout << "2\tM" << endl;
    return 0;
}
```

### Q9

1. 非法，>>运算符后不能定义变量
2. 非法，列表初始化中初始值存在丢失信息的风险会报错
3. 非法，wage未定义
4. 合法

```c++
#include <iostream>

using namespace std;

int main() {

    // cin >> int input_value;
    int input_value;
    cin >> input_value;
    // int i = { 3.14 };
    int i = 3.14;
    // double salary = wage = 9999.99;
    double salary, wage;
    salary = wage = 9999.99;
    // int i = 3.14;
    return 0;
}
```

### Q10

1. 空串
2. 0
3. 未定义
4. 空串

```c++
#include <iostream>
#include <string>

using namespace std;

string global_str;
int global_int;
int main() {
    int local_int;
    string local_str;
    cout << global_str << endl;
    cout << global_int << endl;
    cout << local_str << endl;
    cout << local_int << endl;
    return 0;
}
```

### Q11

1. 定义
2. 定义/声明
3. 声明

### Q12

1. (a) 非法，变量名不能为关键字
2. (c) 非法，变量名不能包含中划线
3. (d) 非法，变量名不能以数字开头

### Q13

100

```c++
#include <iostream>

using namespace std;

int i = 42;
int main() {
    int i = 100;
    int j = i;
    cout  << j << endl;
}
```

### Q14

合法，100 45

```c++
#include <iostream>

using namespace std;

int main() {
    int i = 100, sum = 0;
    for (int i = 0; i != 10; ++i)
        sum += i;
    cout << i << " " << sum << endl;
    return 0;
}
```

### Q15

1. (b) 不合法，引用的初始值必须是一个对象
2. (d) 不合法，引用必须初始化

```c++
#include <iostream>

using namespace std;

int main() {
    int ival = 1.01;
    int &rval1 = 1.01;
    int &rval2 = ival;
    int &rval3;
    return 0;
}
```

### Q16

全部合法
1. (a) 赋3.14159给r2绑定的对象d
2. (b) 将r1绑定的对象i的值赋给r2绑定的对象d
3. (c) 将r2绑定的对象d的值赋给i
4. (d) 将d的值赋给r1绑定的对象i

```c++
#include <iostream>

using namespace std;

int main() {
    int i = 0, &r1 = i;
    double d = 0, &r2 = d;
    r2 = 3.14159;
    cout << i << " " << d << endl;
    r2 = r1;
    cout << i << " " << d << endl;
    i = r2;
    cout << i << " " << d << endl;
    r1 = d;
    cout << i << " " << d << endl;
    return 0;
}
```

### Q17

10 10

```c++
#include <iostream>

using namespace std;

int main() {
    int i, &ri = i;
    i = 5; ri = 10;
    cout << i << " " << ri << endl;
}
```

### Q18

```c++
#include <iostream>

using namespace std;

int main() {
    int a = 1, b = 2;
    int * pa = &a;
    int * pb = &b;

    pa = pb;
    *pb = 3;

    cout << a << " " << b << endl;
    cout << *pa << " " << *pb << endl;

    return 0;
}
```

### Q19

指针是一个变量，而引用是原变量的一个别名。

### Q20

求解42*42

```c++
#include <iostream>

using namespace std;

int main() {
    int i = 42;
    int *p1 = &i; 
    *p1 = *p1 * *p1;
    cout << *p1 << endl;
    return 0;
}
```

### Q21

1. (a) 非法，左边为double *，右边为int *
2. (b) 非法，左边为int *，右边为int

```c
#include <iostream>

using namespace std;

int main() {
    int i = 0;
    double* dp = &i;
    int* ip = i;
    int* p = &i;
    return 0;
}
```

### Q22

1. 指针p不为NULL时，执行代码
2. 指针p指向对象的值不为0时，执行代码

### Q23

不能，不能判断指针是否有效

### Q24

void指针可以指向任何类型

### Q25

1. int类型的指针，int类型，int类型的引用；
2. int类型，int类型的指针；
3. int类型的指针，int类型。

### Q26

1. (a) const对象必须初始化
2. (d) const对象值一旦确定，不能更改

```c++
#include <iostream>

using namespace std;

int main() {
    const int buf;
    int cnt = 0;
    const int sz = cnt;
    ++cnt; ++sz;
    return 0;
}
```

### Q27

1. (a) int &r = 0不合法，非常量引用的初始值必须为左值
2. (b) 不合法，i2可能为const int
3. (f) 不合法，引用初始化后就无法再与其他对象绑定，所以没有这一用法

```c++
#include <iostream>

using namespace std;

int main() {
    // int i = -1, &r = 0;
    // int i2 = 0;
    // const int i2 = 0;
    // int *const p2 = &i2;
    // const int i = -1, &r = 0;
    // int i2 = 0;
    // const int i2 = 0;
    // const int *const p3 = &i2;
    // int i2 = 0;
    // const int i2 = 0;
    // const int *p1 = &i2;
    // const int &const r2;
    // int i = 0;
    const int i = 0;
    const int i2 = i, &r = i;
    return 0;
}
```

### Q28

1. (a) 不合法，常量指针未初始化
2. (b) 不合法，常量指针未初始化
3. (c) 不合法，常量未初始化
4. (d) 不合法，指向常量的常量指针未初始化

```c++
#include <iostream>

using namespace std;

int main() {
    int i, *const cp;
    int *p1, *const p2;
    const int ic, &r = ic;
    const int *const p3;
    const int *p;
    return 0;
}
```

### Q29

1. (b) 不合法，const int* -> int*
2. (c) 不合法，const int* -> int*
3. (d) 不合法，常量指针不能再次赋值
4. (e) 不合法，常量指针不能再次赋值
5. (f) 不合法，常量不能再次赋值

```c++
#include <iostream>

using namespace std;

int main() {
    int i, *const cp;
    int *p1, *const p2;
    const int ic, &r = ic;
    const int *const p3;
    const int *p;
    i = ic;
    p1 = p3;
    p1 = &ic;
    p3 = &ic;
    p2 = p1;
    ic = *p3;
    return 0;
}
```

### Q30

1. v2：顶层
2. p2：底层
3. p3：顶层/底层
4. p4：底层

### Q31

1. (a) 合法，v2为顶层const
2. (b) 不合法，const int* -> int*，且p2为底层const
3. (c) 合法，虽然p2为底层const，但是int * -> const int*
4. (d) 不合法，const int* -> int*，且p3为底层const
5. (e) 合法，p2,p3均为底层const

```c++
#include <iostream>

using namespace std;

int main() {
    // int i = 0;
    const int i = 0;
    const int v2 = 0;
    int v1 = v2;
    int *p1 = &v1, &r1 = v1;
    const int *p2 = &v2, *const p3 = &i, &r2 = v2;
    r1 = v2;
    p1 = v2; p2 = p1;
    p1 = p3; p2 = p3;
    return 0;
}
```

### Q32

不合法，应该改为int null = 0, *p = &null;

### Q33

1. (d) 非法，int -> int *
2. (e) 非法，int -> const int *
3. (f) 非法

### Q34

```c++
#include <iostream>

using namespace std;

int main() {
    int i = 0, &r = i;
    auto a = r;
    const int ci = i, &cr = ci;
    auto b = ci;
    auto c = cr;
    auto d = &i;
    auto e = &ci;
    auto &g = ci;
    a = 42;
    b = 42;
    c = 42;
    d = 42;
    e = 42;
    g = 42;
    return 0;
}
```

### Q35

1. int
2. const int &
3. const int *
4. const int
5. const int &

```c++
#include <iostream>

using namespace std;

int main() {
    // const int i = 42;
    // auto j = i; const auto &k = i; auto *p = &i;
    // const auto j2 = i, &k2 = i;
    const int i = 42;
    int j = i; const int &k = i; const int *p = &i;
    const int j2 = i, &k2 = i;
    return 0;
}
```

### Q36

1. a,int,4
2. b,int,4
3. c,int,4
4. d,int&,4

```c++
#include <iostream>

using namespace std;

int main() {
    int a = 3, b = 4;
    decltype(a) c = a;
    decltype((b)) d = a;
    ++c;
    ++d;
    cout << a << " " << b << " " << c << " "
         << d << endl;
    return 0;
}
```

### Q37

1. a,int,3
2. b,int,4
3. c,int,3
4. d,int&,3

```c++
#include <iostream>

using namespace std;

int main() {
    int a = 3, b = 4;
    decltype(a) c = a;
    decltype(a = b) d = a;
    cout << a << " " << b << " " << c << " "
         << d << endl;
    return 0;
}
```

### Q38

```c++
#include <iostream>

using namespace std;

int main() {
    int a = 0;
    auto b = a;
    decltype(a) c = a;
    cout << a << b << c << endl;
    ++a;
    cout << a << b << c << endl;
    int &d = a;
    auto e = d;
    decltype(d) f = d;
    cout << d << e << f << endl;
    ++d;
    cout << d << e << f << endl;
    return 0;
}
```

### Q39

```c++
struct Foo { /* empty  */ } // Note: no semicolon
int main()
{
    return 0;
}
```

### Q40

```c++
struct Sales_data {
	std::string bookNo;
	unsigned units_sold;
	double revenue;
};
```

### Q41

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

using namespace std;

struct Sales_data {
    std::string bookNo;
    unsigned units_sold;
    double revenue;
};

int main() {
    Sales_data book;
    while (cin >> book.bookNo >> book.units_sold >> book.revenue) {
        cout << book.bookNo << " " << book.units_sold << " " 
             << book.revenue << endl;
    }
    return 0;
}
```

```
0-201-78345-X 3 20.00
0-201-78345-X 2 25.00
```

```c++
#include <iostream>
#include <string>

using namespace std;

struct Sales_data {
    std::string bookNo;
    unsigned units_sold;
    double revenue;
};

int main() {
    double price;
    Sales_data item1, item2;
    cin >> item1.bookNo >> item1.units_sold >> item1.revenue;
    cin >> item2.bookNo >> item2.units_sold >> item2.revenue;
    price = (item1.units_sold * item1.revenue + item2.units_sold * item2.revenue)/
            (item1.units_sold + item2.units_sold);
    cout << item1.bookNo << " " << item1.units_sold + item2.units_sold
         << " " << item1.units_sold * item1.revenue + item2.units_sold * item2.revenue
        << " " << price << endl;
    return 0;
}
```

```
0-201-78345-X 3 20.00
0-201-78345-X 3 20.00
0-201-78345-X 2 25.00
0-201-78345-X 2 25.00
```

```c++
#include <iostream>
#include <string>

using namespace std;

struct Sales_data {
    std::string bookNo;
    unsigned units_sold;
    double revenue;
};

int main() {
    double allprice, num;
    Sales_data sum, item;
    if (cin >> sum.bookNo >> sum.units_sold >> sum.revenue) {
        allprice = sum.units_sold * sum.revenue;
        num = sum.units_sold;
        while (cin >> item.bookNo >> item.units_sold >> item.revenue) {
            allprice += item.units_sold * item.revenue;
            num += item.units_sold;
        }
        cout << sum.bookNo << " " << num << " " << allprice << " " << allprice/num << " " << endl;
    }
    return 0;
}
```

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

using namespace std;

struct Sales_data {
    std::string bookNo;
    unsigned units_sold;
    double revenue;
};

int main() {
    double allprice, num;
    Sales_data sum, item;
    if (cin >> sum.bookNo >> sum.units_sold >> sum.revenue) {
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
    return 0;
}
```

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

using namespace std;

struct Sales_data {
    std::string bookNo;
    unsigned units_sold;
    double revenue;
};

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

### Q42

```c++
// Sales_data.h
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <string>

struct Sales_data {
    std::string bookNo;
    unsigned units_sold;
    double revenue;
};
#endif
```

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
    Sales_data book;
    while (cin >> book.bookNo >> book.units_sold >> book.revenue) {
        cout << book.bookNo << " " << book.units_sold << " " 
             << book.revenue << endl;
    }
    return 0;
}
```

```
0-201-78345-X 3 20.00
0-201-78345-X 2 25.00
```

```c++
#include <iostream>
#include <string>
#include "Sales_data.h"

using namespace std;

int main() {
    double price;
    Sales_data item1, item2;
    cin >> item1.bookNo >> item1.units_sold >> item1.revenue;
    cin >> item2.bookNo >> item2.units_sold >> item2.revenue;
    price = (item1.units_sold * item1.revenue + item2.units_sold * item2.revenue)/
            (item1.units_sold + item2.units_sold);
    cout << item1.bookNo << " " << item1.units_sold + item2.units_sold
         << " " << item1.units_sold * item1.revenue + item2.units_sold * item2.revenue
        << " " << price << endl;
    return 0;
}
```

```
0-201-78345-X 3 20.00
0-201-78345-X 3 20.00
0-201-78345-X 2 25.00
0-201-78345-X 2 25.00
```

```c++
#include <iostream>
#include <string>
#include "Sales_data.h"

using namespace std;

int main() {
    double allprice, num;
    Sales_data sum, item;
    if (cin >> sum.bookNo >> sum.units_sold >> sum.revenue) {
        allprice = sum.units_sold * sum.revenue;
        num = sum.units_sold;
        while (cin >> item.bookNo >> item.units_sold >> item.revenue) {
            allprice += item.units_sold * item.revenue;
            num += item.units_sold;
        }
        cout << sum.bookNo << " " << num << " " << allprice << " " << allprice/num << " " << endl;
    }
    return 0;
}
```

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
    Sales_data sum, item;
    if (cin >> sum.bookNo >> sum.units_sold >> sum.revenue) {
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
    return 0;
}
```

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