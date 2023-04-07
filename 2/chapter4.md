# Chapter 4

### Q1

105

### Q2

1. *(vec.begin())
2. (*(vec.begin()))+1

### Q3

可以接受，效率是C++最大的优势

### Q4

91

```c++
#include <iostream>

using namespace std;

int main() {
    cout << 12 / 3 * 4 + 5 * 15 + 24 % 4 / 2 << endl;
    cout << ((12 / 3) * 4) + (5 * 15) + ((24 % 4) / 2) << endl;
    return 0;
}
```

### Q5

-86
-18
0
-2

### Q6

```c++
#include <iostream>

using namespace std;

int main() {
    int i;
    cin >> i;
    cout << ((i%2)?"Odd":"Even") << endl;
    return 0;
}
```

### Q7

计算结果超出该类型所能表示的范围就会产生溢出。

```c++
#include <iostream>

using namespace std;

int main() {
    short svalue = 32767;
    svalue++;
    cout << svalue << endl;
    unsigned uivalue = 0;
    uivalue--;
    cout << uivalue << endl;
    unsigned short usvalue = 65535;
    usvalue++;
    cout << usvalue << endl;
    return 0;
}
```

### Q8

逻辑与、逻辑或运算符都是先求左侧运算对象的值再求右侧运算对象的值，当且仅当左侧运算对象无法确定表达式的结果时才会计算右侧运算对象的值。
逻辑与：当且仅当左侧运算对象为真时才对右侧运算对象求值。
逻辑或：当且仅当左侧运算对象为假时才对右侧运算对象求值。
相等性运算符：求值顺序不明确。

### Q9

当指针cp不为空时，才会判断指针cp指向的值是否为空

### Q10

```c++
#include <iostream>

using namespace std;

int main() {
    int a;
    while (cin >> a && a != 42) {
        cout << a << endl;
    }
    return 0;
}
```

### Q11

a > b && b > c && c > d

### Q12

i!=(j<k)

### Q13

1. 3 3
2. 3.5 3

```c++
#include <iostream>

using namespace std;

int main() {
    int i;
    double d;
    d = i = 3.5;
    cout << d << " " << i << endl;
    i = d = 3.5;
    cout << d << " " << i << endl;
    return 0;
}
```

### Q14

1. 编译错误：赋值运算符左边要求是可修改的左值
2. 恒为true

### Q15

非法，int*不能转换为int

```c++
#include <iostream>

using namespace std;

int main() {
    double dval; int ival; int *pi;
    // dval = ival = pi = 0;
    dval = ival = *pi = 0;
    return 0;
}
```

### Q16

1. if ((p = getPtr()) != 0)
2. if (i == 1024)

### Q17

1. 前置递增运算符：将运算对象加1，然后将改变后的对象作为求值结果；
2. 后置递增运算符：将运算对象加1，但是求值结果是运算对象改变之前那个值的副本。

### Q18

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> v = {1,2,-1,3};
    auto pbeg = v.begin();
    while (pbeg != v.end() && *pbeg >= 0)
        cout << *pbeg++ << endl;
    cout << "****************" << endl;
    pbeg = v.begin();
    while (pbeg != v.end() && *pbeg >= 0)
        cout << *++pbeg << endl;
    return 0;
}
```

### Q19

1. （a）指针ptr不为空时，判断指针所指的值是不是不为0，最后递增指针ptr；
2. （b）判断ival和ival+1是不是不为0；
3. （c）不正确，vec[ival] <= vec[ival+1]。

### Q20

1. （a）合法，解引用iter，然后再递增iter；
2. （b）不合法，解引用iter后不支持递增运算；
3. （c）不合法，iter没有empty()成员函数；
4. （d）合法，指针iter指向的string对象是否为空；
5. （e）不合法，解引用iter后不支持递增运算；
6. （f）合法，先判断指针iter是否为空，然后再递增iter。

### Q21

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> v = {1,2,-1,3};
    for (auto &i : v) {
        i = (i % 2) ? (2 * i) : i;
    }
    for (auto &i : v) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

### Q22

第二个版本。条件运算符随嵌套层数增加可读性会很差。

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
    unsigned int grade;
    string finalgrade;
    cin >> grade;
    finalgrade = (grade > 90) ? "high pass"
                              : (grade > 75) ? "low pass"
                                             : (grade < 60) ? "fail" : "pass"; 
    cout << finalgrade << endl;
    return 0;
}
```

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
    unsigned int grade;
    string finalgrade;
    cin >> grade;
    if (grade > 90) {
        finalgrade = "high pass";
    }
    else if (grade > 75) {
        finalgrade = "low pass";
    }
    else if (grade < 60) {
        finalgrade = "fail";
    }
    else {
        finalgrade = "pass";
    }
    cout << finalgrade << endl;
    return 0;
}
```

### Q23

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
    string s = "word";
    // string pl = s + s[s.size() - 1] == 's' ? "" : "s" ;
    string pl = s + (s[s.size() - 1] == 's' ? "" : "s") ;
    cout << pl << endl;
    return 0;
}
```

### Q24

结合律从左向右，左结合律从右向左

### Q25

01000000,@

### Q26

unsigned int最小为16位，可能出现位数不够的情况

### Q27

1. 3
2. 7
3. 1
4. 1

```c++
#include <iostream>

using namespace std;

int main() {
    unsigned long ul1 = 3, ul2 = 7;
    cout << (ul1 & ul2) << endl;
    cout << (ul1 | ul2) << endl;
    cout << (ul1 && ul2) << endl;
    cout << (ul1 || ul2) << endl;
    return 0;
}
```

### Q28

```c++
#include <iostream>

using namespace std;

int main() {
    
    // void type
    cout << "void: nullptr_t\t" << sizeof(nullptr_t) << " bytes" << endl << endl;
    
    // boolean type
    cout << "bool:\t\t" << sizeof(bool) << " bytes" << endl << endl;
    
    // charactor type
    cout << "char:\t\t" << sizeof(char) << " bytes" << endl;
    cout << "wchar_t:\t" << sizeof(wchar_t) << " bytes" << endl;
    cout << "char16_t:\t" << sizeof(char16_t) << " bytes" << endl;
    cout << "char32_t:\t" << sizeof(char32_t) << " bytes" << endl << endl;
    
    // integers type
    cout << "short:\t\t" << sizeof(short) << " bytes" << endl;
    cout << "int:\t\t" << sizeof(int) << " bytes" << endl;
    cout << "long:\t\t" << sizeof(long) << " bytes" << endl;
    cout << "long long:\t" << sizeof(long long) << " bytes" << endl << endl;
    
    // floating point type
    cout << "float:\t\t" << sizeof(float) << " bytes" << endl;
    cout << "double:\t\t" << sizeof(double) << " bytes" << endl;
    cout << "long double:\t" << sizeof(long double) << " bytes" << endl << endl;
	
    // Fixed width integers
    cout << "int8_t:\t\t" << sizeof(int8_t) << " bytes" << endl;
    cout << "uint8_t:\t" << sizeof(uint8_t) << " bytes" << endl;
    cout << "int16_t:\t" << sizeof(int16_t) << " bytes" << endl;
    cout << "uint16_t:\t" << sizeof(uint16_t) << " bytes" << endl;
    cout << "int32_t:\t" << sizeof(int32_t) << " bytes" << endl;
    cout << "uint32_t:\t" << sizeof(uint32_t) << " bytes" << endl;
    cout << "int64_t:\t" << sizeof(int64_t) << " bytes" << endl;
    cout << "uint64_t:\t" << sizeof(uint64_t) << " bytes" << endl;
    	
    return 0;

}
```

### Q29

40/4=10，int数组所占的字节数/int所占的字节数=数组个数；
8/4=2，指针所占的字节数/int所占的字节数。

```c++
#include <iostream>

using namespace std;

int main() {
    int x[10];
    int *p = x;
    cout << sizeof(x)/sizeof(*x) << endl;
    cout << sizeof(p)/sizeof(*p) << endl;
    return 0;
}
```

### Q30

1. （a）(sizeof x) + y；
2. （b）sizeof(p->mem[i])；
3. （c）(sizeof a) < b；
4. （d）sizeof(f())。

### Q31

后置版本需要保存未修改的值，如果不需要未修改的值就使用前置版本。使用后置版本无需改动。

### Q32

遍历数组，ix是以索引方式变现，ptr是以指针方式变现

### Q33

(someValue ? ++x, ++y : --x), --y

### Q34

1. （a）fval:float->bool；
2. （b）ival:int->float，结果:float->double；
3. （c）cval:char->int，结果:int->double。

### Q35

1. （a）'a':char->int,结果:int->char;
2. （b）ival:int->double,ui:unsigned int->double,结果:double->float;
3. （c）ui:unsigned int->float,结果:float->double；
4. （d）ival:int->float,ival+fval:float->double,结果:double->char。

### Q36

i *= static_cast<int>(d)

### Q37

1. （a）pv = static_cast<void*>(const_cast<string*>(ps));
2. （b）i = static_cast<int>(*pc);
3. （c）pv = static_cast<void*>(&d);
4. （d）pc = reinterpret_cast<char*>(pv);

### Q38

将(j/i)强制转换为double类型，并赋给slope。