# Chapter 3

### Q1

```c++
#include <iostream>

using std::cout;
using std::endl;

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

```c++
#include <iostream>

using std::cout;
using std::endl;

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

```c++
#include <iostream>

using std::cin;
using std::cout;
using std::endl;

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

```c++
#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::endl;

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

```c++
#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::endl;

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

```c++
#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::endl;

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

```c++
#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::endl;

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

```c++
#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::cerr;
using std::endl;

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

### Q2

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
    string line;
    while (getline(cin, line)) {
        cout << line << endl;
    }
    return 0;
}
```

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
    string word;
    while (cin >> word) {
        cout << word << endl;
    }
    return 0;
}
```

### Q3

1. 输入运算符：忽略开头的空白，从第一个字符开始直到遇见下一处空白停止。
2. getline：直到遇见换行符停止。

### Q4

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
    string s1, s2;
    cin >> s1 >> s2;
    if (s1 == s2) {
        cout << "Two strings are equal." << endl;
    }
    else {
        if (s1 > s2) {
            cout << s1 << endl;
        }
        else {
            cout << s2 << endl;
        }
    }
    return 0;
}
```

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
    string s1, s2;
    cin >> s1 >> s2;
    if (s1.length() == s2.length()) {
        cout << "Two strings are equal in length." << endl;
    }
    else {
        if (s1.length() > s2.length()) {
            cout << s1 << endl;
        }
        else {
            cout << s2 << endl;
        }
    }
    return 0;
}
```

### Q5

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
    string s = "", temp;
    while (cin >> temp) {
        s += temp;
    }
    cout << s << endl;
    return 0;
}
```

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
    string s = "", temp;
    while (cin >> temp) {
        s += temp + " ";
    }
    cout << s << endl;
    return 0;
}
```

### Q6

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
    string s;
    cin >> s;
    for (auto &c : s) {
        c = 'X';
    }
    cout << s << endl;
    return 0;
}
```

### Q7

字符串s无变化

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
    string s;
    cin >> s;
    for (auto c : s) {
        c = 'X';
    }
    cout << s << endl;
    return 0;
}
```

### Q8

for循环。已知迭代次数用for循环，否则用while循环。

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
    string s;
    cin >> s;
    decltype(s.size()) i = 0;
    while (i < s.size()) {
        s[i] = 'X';
        i++;
    }
    cout << s << endl;
    return 0;
}
```

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
    string s;
    cin >> s;
    for (decltype(s.size()) i = 0; i < s.size(); i++) {
        s[i] = 'X';
    }
    cout << s << endl;
    return 0;
}
```

### Q9

输出string对象s中第一个字符
合法，string对象定义后默认初始化为'\0'

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
    string s;
    cout << s[0] << endl;
    return 0;
}
```

### Q10

```c++
#include <iostream>
#include <string>
#include <cctype>

using namespace std;

int main() {
    string s1, s2;
    cin >> s1;
    for (auto c : s1) {
        if (!ispunct(c)) {
            s2 += c;
        }
    }
    cout << s2 << endl;
    return 0;
}
```

### Q11

合法，const char &，但是不能改变c

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
    const string s = "Keep out!";
    for(auto &c : s){ /* ... */ }
    return 0;
}

```

### Q12

1. (a) 正确。创建一个元素为vector<int>的vector对象.
2. (b) 错误。vector对象类型不一致。
3. (c) 正确。创建一个有10个“null”string对象的vector对象。

```c++
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main() {
    vector<vector<int>> ivec;
    vector<string> svec = ivec;
    vector<string> svec(10, "null");
    return 0;
}
```

### Q13

1. 0
2. 10 0
3. 10 42
4. 1 10
5. 2 10,42
6. 10 ""
7. 10 "hi"

### Q14

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> v;
    int temp;
    while (cin >> temp) {
        v.push_back(temp);
    }
    for (auto i : v) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

### Q15

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<string> v;
    string temp;
    while (cin >> temp) {
        v.push_back(temp);
    }
    for (auto i : v) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

### Q16

```c++
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main() {
    vector<int> v1;
    cout << v1.size() << endl;
    for (auto i : v1) {
        cout << i << " ";
    }
    cout << endl;
    vector<int> v2(10);
    cout << v2.size() << endl;
    for (auto i : v2) {
        cout << i << " ";
    }
    cout << endl;
    vector<int> v3(10, 42);
    cout << v3.size() << endl;
    for (auto i : v3) {
        cout << i << " ";
    }
    cout << endl;
    vector<int> v4{ 10 };
    cout << v4.size() << endl;
    for (auto i : v4) {
        cout << i << " ";
    }
    cout << endl;
    vector<int> v5{ 10, 42 };
    cout << v5.size() << endl;
    for (auto i : v5) {
        cout << i << " ";
    }
    cout << endl;
    vector<string> v6{ 10 };
    cout << v6.size() << endl;
    for (auto i : v6) {
        cout << i << " ";
    }
    cout << endl;
    vector<string> v7{ 10, "hi" };
    cout << v7.size() << endl;
    for (auto i : v7) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

### Q17

```c++
#include <iostream>
#include <vector>
#include <cctype>

using namespace std;

int main() {
    vector<string> v;
    string temp;
    while (cin >> temp) {
        v.push_back(temp);
    }
    for (auto &i : v) {
        for (auto &c : i) {
            c = toupper(c);
        }
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

### Q18

不合法

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> ivec;
    // ivec[0] = 42;
    ivec.push_back(42);
    return 0;
}
```

### Q19

第一种最好，简洁

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> v1(10,42);
    vector<int> v2{42, 42, 42, 42, 42, 42, 42, 42, 42, 42};
    vector<int> v3 = {42, 42, 42, 42, 42, 42, 42, 42, 42, 42};
    return 0;
}
```

### Q20

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> v;
    int temp;
    while (cin >> temp) {
        v.push_back(temp);
    }
    for (decltype(v.size()) i = 0; i < v.size() - 1; i++) {
        cout << v[i] + v[i+1] << " ";
    }
    cout << endl;
    for (decltype(v.size()) i = 0; i * 2 < v.size(); i++) {
        if (i * 2 == v.size() - 1) {
            cout << v[i] << " ";
        }
        else {
            cout << v[i] + v[v.size() - i - 1] << " ";
        }
    }
    cout << endl;
    return 0;
}
```

### Q21

```c++
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main() {
    vector<int> v1;
    cout << v1.size() << endl;
    for (auto i = v1.cbegin(); i != v1.cend(); i++) {
        cout << *i << " ";
    }
    cout << endl;
    vector<int> v2(10);
    cout << v2.size() << endl;
    for (auto i = v2.cbegin(); i != v2.cend(); i++) {
        cout << *i << " ";
    }
    cout << endl;
    vector<int> v3(10, 42);
    cout << v3.size() << endl;
    for (auto i = v3.cbegin(); i != v3.cend(); i++) {
        cout << *i << " ";
    }
    cout << endl;
    vector<int> v4{ 10 };
    cout << v4.size() << endl;
    for (auto i = v4.cbegin(); i != v4.cend(); i++) {
        cout << *i << " ";
    }
    cout << endl;
    vector<int> v5{ 10, 42 };
    cout << v5.size() << endl;
    for (auto i = v5.cbegin(); i != v5.cend(); i++) {
        cout << *i << " ";
    }
    cout << endl;
    vector<string> v6{ 10 };
    cout << v6.size() << endl;
    for (auto i = v6.cbegin(); i != v6.cend(); i++) {
        cout << *i << " ";
    }
    cout << endl;
    vector<string> v7{ 10, "hi" };
    cout << v7.size() << endl;
    for (auto i = v7.cbegin(); i != v7.cend(); i++) {
        cout << *i << " ";
    }
    cout << endl;
    return 0;
}
```

### Q22

```c++
#include <iostream>
#include <vector>
#include <string>
#include <cctype>

using namespace std;

int main() {
    vector<string> text{"hello,world","","hh"};
    for (auto it = text.begin(); it != text.end() && !it->empty(); ++it) {
        for (auto &i : *it) {
            i = toupper(i);
        }
    }
    for (auto it = text.cbegin(); it != text.cend() && !it->empty(); ++it) {
        cout << *it << endl;
    }
    return 0;
}
```

### Q23

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> v{0,1,2,3,4,5,6,7,8,9};
    for (auto i = v.begin(); i != v.end(); i++) {
        *i = *i * 2;
    }
    for (auto i = v.begin(); i != v.end(); i++) {
        cout << *i << endl;
    }
    return 0;
}
```

### Q24

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> v;
    int temp;
    while (cin >> temp) {
        v.push_back(temp);
    }
    for (auto i = v.cbegin(); i < v.cend() - 1; i++) {
        cout << *i + *(i+1) << " ";
    }
    cout << endl;
    for (auto i = v.cbegin(); i < v.cbegin()+(v.cend()-v.cbegin()+1)/2; i++) {
        if (i == v.cbegin()+(v.cend()-v.cbegin())/2) {
            cout << *i << " ";
        }
        else {
            cout << *i + *(v.cbegin()+(v.cend()-i-1)) << " ";
        }
    }
    cout << endl;
    return 0;
}
```

### Q25

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<unsigned> scores(11, 0);
    unsigned grade;
    while (cin >> grade) {
        if (grade <= 100) {
            (*(scores.begin()+grade/10))++;
        }
    }
    for (auto i : scores) {
        cout << i << endl;
    }
    return 0;
}
```

### Q26

迭代器相加无意义

### Q27

1. 非法，数组维度必须是常量表达式
2. 合法
3. 非法，当txt_size()是constexpr时正确。
4. 非法，没有空间可存放结尾的空字符

### Q28

1. ""
2. 0
3. ""
4. 未定义

### Q29

数组大小是固定的，相比于vector缺少灵活性

### Q30

```c++
#include <iostream>

int main() {
    constexpr size_t array_size = 10;
    int ia[array_size];
    // for (size_t ix = 1; ix <= array_size; ++ix)
    for (size_t ix = 0; ix < array_size; ++ix)
        ia[ix] = ix;
}
```

### Q31

```c++
#include <iostream>

using namespace std;

int main() {
    int a[10];
    for (size_t i = 0; i < 10; i++) {
        a[i] = i;
    }
    for (size_t i = 0; i < 10; i++) {
        cout << a[i] << " ";
    }
    cout << endl;
    return 0;
}
```

### Q32

```c++
#include <iostream>

using namespace std;

int main() {
    int a[10], b[10];
    for (size_t i = 0; i < 10; i++) {
        a[i] = i;
    }
    for (size_t i = 0; i < 10; i++) {
        b[i] = a[i];
    }
    for (auto i : a) {
        cout << i << " ";
    }
    cout << endl;
    for (auto i : b) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> a, b;
    for (size_t i = 0; i < 10; i++) {
        a.push_back(i);
    }
    for (size_t i = 0; i < 10; i++) {
        b.push_back(a[i]);
    }
    for (auto i : a) {
        cout << i << " ";
    }
    cout << endl;
    for (auto i : b) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

### Q33

数组值未定义

```c++
#include <iostream>

using namespace std;

int main() {
    unsigned scores[11];
    unsigned grade;
    while (cin >> grade) {
        if (grade <= 100) {
            ++scores[grade/10];
        }
    }
    for (auto i : scores) {
        cout << i << endl;
    }
    return 0;
}
```

### Q34

将p1移动(p2-p1)个位置；p1或p2是非法的，该程序就是非法的。

### Q35

```c++
#include <iostream>

using namespace std;

int main() {
    int a[5] = {1,2,3,4,5};
    int * pbeg = begin(a);
    int * pend = end(a);
    while (pbeg != pend) {
        *pbeg = 0;
        ++pbeg;
    }
    for (auto i : a) {
        cout << i << endl;
    }
    return 0;
}
```

### Q36

```c++
#include <iostream>
#include <vector>

using namespace std;

bool compare_arr (int *, int *, int *, int *);
bool compare_vector (vector<int>, vector<int>);

int main() {
    int a[] = {1,2,3,4,5};
    int b[] = {1,2,3,4,5};
    cout << compare_arr(begin(a),end(a),begin(b),end(b)) << endl;
    vector<int> c = {1,2,3,4,5};
    vector<int> d = {1,2,3,4,5};
    cout << compare_vector(c, d) << endl;
    return 0;
}

bool compare_arr (int * a1beg, int * a1end, int * a2beg, int * a2end) {
    if (a1end - a1beg == a2end- a2beg) {
        for (auto i = a1beg, j = a2beg; i != a1end && j != a2end; i++,j++) {
            if (*i != *j) {
                return false;
            }
        }
        return true;
    }
    return false;
}

bool compare_vector (vector<int> a, vector<int> b) {
    return a == b;
}
```

### Q37

打印字符数组内容，但是没有'\0'，循环可能不会在字符数组结尾停止

```c++
#include <iostream>

using namespace std;

int main() {
    const char ca[] = { 'h', 'e', 'l', 'l', 'o' };
    const char *cp = ca;
    while (*cp) {
        cout << *cp << endl;
        ++cp;
    }
    return 0;
}
```

### Q38

两个指针相加实际为地址相加，没什么意义

### Q39

```c++
#include <iostream>
#include <string>
#include <cstring>

using namespace std;

int main() {
    char a[] = "hello";
    char b[] = "hello";
    cout << (strcmp(a,b)==0) << endl;
    string c = "hello";
    string d = "hello";
    cout << (c==d) << endl;
    return 0;
}
```

### Q40

```c++
#include <iostream>
#include <cstring>

using namespace std;

int main() {
    char a[] = "hello";
    char b[] = "hello";
    char c[20];
    strcpy(c,a);
    cout << c << endl;
    strcat(c,b);
    cout << c << endl;
    return 0;
}
```

### Q41

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {
    int arr[] = {1,2,3,4,5};
    vector<int> v(begin(arr), end(arr));
    for (auto i : v) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

### Q42

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> v(5,1);
    int arr[5];
    for (decltype(v.size()) i = 0; i < v.size(); i++) {
        arr[i] = v[i];
    }
    for (auto i : v) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

### Q43

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {

    int ia[3][4] = {
        {0,1,2,3},
        {4,5,6,7},
        {8,9,10,11}
    };

    for (const int (&row)[4] : ia) {
        for (int i: row) {
            cout << i << " ";
        }
        cout << endl;
    }
    
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 4; j++) {
            cout << ia[i][j] << " ";
        }
        cout << endl;
    }

    for (int (*p)[4] = begin(ia); p != end(ia); p++) {
        for (int * q = begin(*p); q != end(*p); q++) {
            cout << *q << " ";
        }
        cout << endl;
    }
    return 0;
}
```

### Q44

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {

    // using int_array = int[4];
    typedef int int_array[4];

    int ia[3][4] = {
        {0,1,2,3},
        {4,5,6,7},
        {8,9,10,11}
    };

    for (const int_array &row : ia) {
        for (int i: row) {
            cout << i << " ";
        }
        cout << endl;
    }
    
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 4; j++) {
            cout << ia[i][j] << " ";
        }
        cout << endl;
    }

    for (int_array *p = begin(ia); p != end(ia); p++) {
        for (int * q = begin(*p); q != end(*p); q++) {
            cout << *q << " ";
        }
        cout << endl;
    }
    return 0;
}
```

### Q45

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {

    int ia[3][4] = {
        {0,1,2,3},
        {4,5,6,7},
        {8,9,10,11}
    };

    for (const auto &row : ia) {
        for (int i: row) {
            cout << i << " ";
        }
        cout << endl;
    }
    
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 4; j++) {
            cout << ia[i][j] << " ";
        }
        cout << endl;
    }

    for (auto p = begin(ia); p != end(ia); p++) {
        for (auto q = begin(*p); q != end(*p); q++) {
            cout << *q << " ";
        }
        cout << endl;
    }
    return 0;
}
```