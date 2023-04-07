# Chapter 6

### Q1

形参在函数的定义中声明；实参是形参的初始值;实参的类型必须与对应的形参类型匹配。

### Q2

```c++
string f() {
    string s;
    // ...
    return s;
}
```

```c++
void f2(int i) { /* ... */ }
```

```c++
int calc(int v1, int v2) { /* ... */ }
```

```c++
double square (double x) {return x * x;}
```

### Q3

```c++
#include <iostream>

using namespace std;

int fact(int);

int main() {
    cout << fact(1) << endl;
    cout << fact(5) << endl;
    return 0;
}

int fact(int val) {
    if (val == 1) {
        return 1;
    }
    else {
        return val*fact(val-1);
    }
}
```

### Q4

```c++
#include <iostream>

using namespace std;

int fact(int);

int main() {
    int num;
    cin >> num;
    cout << fact(num) << endl;
    return 0;
}

int fact(int val) {
    if (val == 1) {
        return 1;
    }
    else {
        return val*fact(val-1);
    }
}
```

### Q5

```c++
#include <iostream>

using namespace std;

int myabs(int);

int main() {
    int num;
    cin >> num;
    cout << myabs(num) << endl;
    return 0;
}

int myabs(int val) {
    if (val < 0) {
        return -val;
    }
    return val;
}
```

### Q6

形参是局部变量的一种。形参和函数体内部定义的变量统称为局部变量。局部静态变量的生命周期贯穿函数调用及之后的时间。

```c++
#include <iostream>

using namespace std;

int myabs(int);

int main() {
    int num;
    cin >> num;
    cout << myabs(num) << endl;
    return 0;
}

int myabs(int val) {
    static int count = 0;
    int num = ++count;
    cout << num << endl;
    if (val < 0) {
        return -val;
    }
    return val;
}
```

### Q7

```c++
#include <iostream>

using namespace std;

int count();

int main() {
    cout << count() << endl;
    cout << count() << endl;
    cout << count() << endl;
    return 0;
}

int count() {
    static int count = 0;
    return count++;
}
```

### Q8

```c++
#ifndef CHAPTER6_H
#define CHAPTER6_H

int fact(int);
int myabs(int);

#endif
```

### Q9

```c++
// Chapter6.h
#ifndef CHAPTER6_H
#define CHAPTER6_H

int fact(int);
int myabs(int);

#endif
```

```c++
// fact.cc
#include "Chapter6.h"

int fact(int val) {
    if (val == 1) {
        return 1;
    }
    else {
        return val*fact(val-1);
    }
}
```

```c++
// factMain.cc
#include <iostream>
#include "Chapter6.h"

int fact(int);

int main() {
    int num;
    std::cin >> num;
    std::cout << fact(num) << std::endl;
    return 0;
}
```

### Q10

```c++
#include <iostream>

void swap(int *, int *);

int main() {
    int a = 1, b = 2;
    std::cout << a << " " << b << std::endl;
    swap(&a, &b);
    std::cout << a << " " << b << std::endl;
    return 0;
}

void swap(int * a, int * b) {
    int temp;
    temp = *a;
    *a = *b;
    *b = temp;
}
```

### Q11

```c++
#include <iostream>

void reset(int &);

int main() {
	int i = 42;
	std::cout << i << std::endl;
	reset(i);
	std::cout << i << std::endl;
	return 0;
}

void reset(int &i) {
	i = 0;
}
```

### Q12

引用更易于使用，不用考虑传递的是指针，避免语法错误

```c++
#include <iostream>

void swap(int &, int &);

int main() {
    int a = 1, b = 2;
    std::cout << a << " " << b << std::endl;
    swap(a, b);
    std::cout << a << " " << b << std::endl;
    return 0;
}

void swap(int & a, int & b) {
    int temp;
    temp = a;
    a = b;
    b = temp;
}
```

### Q13

将实参的值拷贝后赋给形参，不能通过改变形参的值来改变实参；使用引用将形参绑定到实参上，可以通过改变形参来改变实参。

### Q14

```c++
// 引用
void reset(int &i) {
	i = 0;
}
```

```c++
// 非引用
void print(std::vector<int>::iterator begin, std::vector<int>::iterator end) {
        for (std::vector<int>::iterator iter = begin; iter != end; ++iter)
                std::cout << *iter << std::endl;
}
```

### Q15

1. s不需要改变，occurs需要改变
2. s可能很大，减小开销，occurs需要改变，c没有上述两个需求
3. s可能会改变
4. occurs不能改变，会报错

### Q16

该函数无需改变实参，所以建议设置成const，这样也能传入const string对象或字符串字面值

```c++
bool is_empty(const string& s) { return s.empty(); }
```

### Q17

不同，前者不需要改变实参，后者需要改变实参

```c++
#include <iostream>
#include <string>

using namespace std;

bool hasUpper(const string &);
void toLower(string &);

int main() {
    string s1;
    cin >> s1;
    string s2 = "Hello";
    cout << hasUpper(s1) << endl;
    cout << hasUpper(s2) << endl;
    toLower(s1);
    toLower(s2);
    cout << s1 << endl;
    cout << s2 << endl;
	return 0;
}

bool hasUpper(const string &s) {
    for (auto i : s) {
        if (isupper(i)) {
            return true;
        }
    }
    return false;
}

void toLower(string &s) {
    for (auto &i : s) {
        i = tolower(i);
    }
}
```

### Q18

1. （a）bool compare(matrix &,matrix &);
2. （b）vector<int>::iterator change_val(int,vector<int>::iterator);

### Q19

1. （a）不合法，只能一个形参；
2. （b）合法；
3. （c）合法；
4. （d）合法。

### Q20

不需要改变实参的时；传入常量实参时会报错。

### Q21

```c++
#include <iostream>

using namespace std;

int compare(int, int *);

int main()
{
	int i = 0, j = 1;
	cout << compare(i, &j) << endl;
	return 0;
}

int compare(int i, int *j) {
	return i > *j ? i : *j; 
}
```

### Q22

```c++
#include <iostream>

void swap(int *&, int *&);

int main() {
	int i = 0, j = 1;
	int *pi = &i, *pj = &j;
	std::cout << *pi << " " << *pj << std::endl;
	swap(pi, pj);
	std::cout << *pi << " " << *pj << std::endl;
	return 0;
}

void swap(int *&i, int *&j) {
	int *tmp;
	tmp = i;
	i = j;
	j = tmp;
}
```

### Q23

```c++
#include <iostream>

using namespace std;

void print(const int *);
void print(const int *, const int *);
void print(const int [], size_t);
void print(const int (&)[2]);

int main() {
	int i = 0, j[2] = {0, 1};

	print(&i);
	print(begin(j), end(j));
	print(j, end(j) - begin(j));
	print(j);

	return 0;
}

void print(const int * pi) {
	cout << *pi << endl;
}

void print(const int * beg, const int * end) {
	while (beg != end) {
		cout << *beg++ << " ";
    }
	cout << endl;
}

void print(const int ia[], size_t size) {
	for (size_t i = 0; i != size; ++i) {
		cout << ia[i] << " ";
    }
	cout << endl;
}

void print(const int (&arr)[2]) {
	for (auto e : arr) {
		cout << e << " ";
    }
	cout << endl;
}
```

### Q24

```c++
void print(const int (&ia)[10]) {
    for (size_t i = 0; i != 10; ++i)
		cout << ia[i] << endl;
}
```

### Q25

```c++
#include <iostream>
#include <string>

using namespace std;

int main(int argc, char *argv[]) {
	string s;
    s += argv[1];
    s += argv[2];
	cout << s << endl;
	return 0;
}
```

### Q26

```c++
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
    for (int i = 0; i != argc; ++i) {
        cout << argv[i] << " ";
    }
    cout << endl;
	return 0;
}
```

### Q27

```c++
#include <iostream>

using namespace std;

int sum(initializer_list<int>);

int main() {
    cout << sum({1,2,3}) << endl;
    cout << sum({1,2,3,4}) << endl;
    return 0;
}

int sum(initializer_list<int> list) {
    int val = 0;
    for (auto beg = list.begin(); beg != list.end(); ++beg) {
        val += *beg;
    }
    return val;
}
```

### Q28

const std::string&

### Q29

视情况而定，如果拷贝代价小，可以不设成引用

### Q30

```c++
bool str_subrange(const string &str1, const string &str2)
{
    if (str1.size() == str2.size())
        return str1 == str2;
    auto size = (str1.size() < str2.size())
                ? str1.size() : str2.size();
    for (decltype(size) i = 0; i != size; ++i) {
        if (str1[i] != str2[i])
            return;
    }
}
```

### Q31

返回的引用是局部对象的引用，返回的常量引用是局部常量对象的引用时。

### Q32

返回数组指定下标成员的引用

```c++
int &get(int *array, int index) { return array[index]; }
int main()
{
    int ia[10];
    for (int i = 0; i != 10; ++i)
        get(ia, i) = i;
}
```

### Q33

```c++
#include <iostream>
#include <vector>

using namespace std;

void print(vector<int>::const_iterator, vector<int>::const_iterator);

int main() {
    vector<int> v = {1,2,3,4};
    print(v.cbegin(), v.cend());
    return 0;
}

void print(vector<int>::const_iterator iterator_begin, vector<int>::const_iterator iterator_end) {
	if (iterator_begin != iterator_end) {
		cout << *iterator_begin << " ";
		return print(++iterator_begin, iterator_end);
	}
    else {
		cout << endl;
		return;
	}
}
```

### Q34

如果val>=0，函数将会多乘以一个1；如果val<0，函数将会不断地调用它自身直到空间耗尽为止。

### Q35

val--会返回未修改的val内容，使程序陷入无限循环；val--会修改val的内容，使程序运行结果不符合预期。

### Q36

```c++
string (&fun(string (&)[10]))[10];
```

### Q37

```c++
using string_10 = string[10];
typedef string string_10[10];
string_10 &fun(string_10 &);
```

```c++
auto fun(string (&)[10]) -> string (&)[10];
```

```c++
string arr[10];
decltype(arr) &fun(decltype(arr) &arrs);
```

### Q38

```c++
decltype(odd) &arrPtr(int i) {
    return (i % 2) ? odd : even;
}
```

### Q39

1. （a）合法；
2. （b）非法，仅返回值不同；
3. （c）合法。

### Q40

1. （a）正确；
2. （b）错误。一旦函数中某个形参被赋予了默认值，它后面所有形参都必须有默认值。

### Q41

1. （a）非法，第一个形参没有默认实参，必须给出实参；
2. （b）合法；
3. （c）合法，但与初衷不符，'*'转换成int。

### Q42

```c++
#include <iostream>
#include <string>

using namespace std;

string make_plural(size_t ctr, const string &word, const string &ending = "s") {
	return (ctr > 1) ? word + ending : word;
}

int main() {
	cout << make_plural(1, "success", "es") << endl;
	cout << make_plural(2, "success", "es") << endl;
	cout << make_plural(1, "failure") << endl;
	cout << make_plural(2, "failure") << endl;
    return 0;
}
```

### Q43

1. （a）头文件，尽管内联函数在程序中可以多次定义，但它的多个定义必须完全一致，所以放在头文件中比较好；
2. （b）头文件，声明放在头文件中。

### Q44

```c++
#include <iostream>
#include <string>

using namespace std;

inline bool isShorter(const string &, const string &);

int main() {
	string s1("A"), s2("B");
	cout << isShorter(s1, s2) << endl;
	return 0;
}

inline bool isShorter(const string &s1, const string &s2) {
	return s1.size() < s2.size();
}
```

### Q45

有些适合，有些不适合，判断标准如下。
1. （a）函数代码量多，功能复杂，体积庞大。
2. （b）递归函数

### Q46

不能，因为返回值不是一个常量表达式

### Q47

打开：g++ test.cpp -o test
关闭：g++ test.cpp -D NDEBUG -o test

```c++
#include <iostream>
#include <vector>
#include <cassert>

using namespace std;

void print(vector<int>::const_iterator, vector<int>::const_iterator);

int main() {
    vector<int> v = {1,2,3,4};
    print(v.cbegin(), v.cend());
    return 0;
}

void print(vector<int>::const_iterator iterator_begin, vector<int>::const_iterator iterator_end) {
#ifdef NDEBUG
    cerr << iterator_end-iterator_begin << endl;
    cerr << __func__ << endl;
    cerr << __FILE__ << endl;
    cerr << __LINE__ << endl;
    cerr << __TIME__ << endl;
    cerr << __DATE__ << endl;
#endif
    assert(0);

	if (iterator_begin != iterator_end) {
		cout << *iterator_begin << " ";
		return print(++iterator_begin, iterator_end);
	}
    else {
		cout << endl;
		return;
	}
}
```

### Q48

不合理。assert宏通常用于检查不能发生的条件，改为assert(!cin || s == sought)。

### Q49

1. 候选函数具备两个特征：一是与被调用的函数同名，二是其声明在调用点可见。
2. 可行函数是从候选函数中选出的，有两个特征：一是其形参数量与本次调用提供的实参数量相等，二是每个实参的类型与对应的形参类型相同，或者能转换成形参的类型。

### Q50

1. （a）不合法，二义性；
2. （b）合法，最佳匹配void f(int)；
3. （c）合法，最佳匹配void f(int, int)；
4. （d）合法，最佳匹配void f(double, double = 3.14)。

### Q51

```c++
#include <iostream>

using namespace std;

void f();
void f(int);
void f(int, int);
void f(double, double);

int main() {
    // f(2.56, 42);
    f(42);
    f(42, 0);
    f(2.56, 3.14);
    return 0;
}

void f() {
    cout << "f()" << endl;
}

void f(int a) {
    cout << "f(int)" << endl;
}

void f(int a, int b) {
    cout << "f(int, int)" << endl;
}

void f(double a, double b = 3.14) {
    cout << "f(double, double)" << endl;
}
```

### Q52

1. （a）3等级，通过类型提升实现的匹配；
2. （b）4等级，通过算数类型转换实现的匹配。

### Q53

1. （a）合法，函数重载；
2. （b）合法，函数重载；
3. （c）合法，顶层const，只能重复声明不能重复定义。

### Q54

见Q56

### Q55

见Q56

### Q56

```c++
#include <iostream>
#include <vector>

using namespace std;

int add(int, int);
int subtract(int, int);
int multiply(int, int);
int divide(int, int);

int main() {
    int a = 1, b = 2;
    vector<int (*)(int, int)> vf{add, subtract, multiply, divide};
    for (auto i : vf) {
        cout << (*i)(a,b) << endl;
    }
    return 0;
}

int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}

int multiply(int a, int b) {
    return a * b;
}

int divide(int a, int b) {
    return b != 0 ? a / b : 0;
}
```