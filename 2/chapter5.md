# Chapter 5

### Q1

只有一个分号的语句。在程序的某个地方，语法上需要一条语句但是逻辑上不需要，此时应该使用空语句

### Q2

复合语句是指用花括号括起来的语句和声明的序列，复合语句也被称作块。如果在程序的某个地方，语法上需要一条语句，但是逻辑上需要多条语句，则应使用复合语句。

### Q3

可读性降低

```c++
#include <iostream>

int main() {
    int sum = 0, val = 1;
    while (val <= 10)
        sum += val, ++val;
    std::cout << "Sum of 1 to 10 inclusive is " << sum << std::endl;
 
    return 0;
}
```

### Q4

```c++
while (iter++ != s.end()) { /* . . . */ }
```

```c++
bool status;
while (status = find(word)) { /* . . . */ }
if (!status) { /* . . . */ }
```

### Q5

```c++
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main()
{
    vector<string> scores = {"F", "D", "C", "B", "A", "S"};
    int grade;
    while (cin >> grade) {
        string lettergrade;
        if (grade < 60)
            lettergrade = scores[0];
        else {
            lettergrade = scores[(grade - 50) / 10];
        }
        cout << lettergrade << endl;
    }
    return 0;
}
```

### Q6

```c++
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main()
{
    vector<string> scores = {"F", "D", "C", "B", "A", "S"};
    int grade;
    while (cin >> grade) {
        cout << ((grade < 60)?scores[0]:scores[(grade-50)/10]) << endl;
    }
    return 0;
}
```

### Q7

```c++
if (ival1 != ival2) 
    ival1 = ival2;
else 
    ival1 = ival2 = 0;
```

```c++
if (ival < minval) {
    minval = ival;
    occurs = 1;
}
```

```c++
int ival;
if (ival = get_value())
    cout << "ival = " << ival << endl;
if (!ival)
    cout << "ival = 0\n";
```

```c++
if (ival == 0)
    ival = get_value();
```

### Q8

当一个if语句嵌套在另一个if语句内部时，很可能if分支会多于else分支。这时候我们怎么知道某个给定的else是和哪个if匹配呢。这个问题通常称作悬垂else。就C++而言，它规定else与离它最近的尚未匹配的if匹配的if匹配，从而消除了程序的二义性。

### Q9

```c++
#include <iostream>

using namespace std;

int main()
{
	char c;
	unsigned aCnt = 0, eCnt = 0, iCnt = 0, oCnt = 0, uCnt = 0;
	while (cin >> c) {
		if (c == 'a')
            ++aCnt;
        else if (c == 'e')
            ++eCnt;
        else if (c == 'i')
            ++iCnt;
        else if (c == 'o')
            ++oCnt;
        else if (c == 'u')
            ++uCnt;
	}
	cout << "a:" << aCnt << endl;
	cout << "e:" << eCnt << endl;
	cout << "i:" << iCnt << endl;
	cout << "o:" << oCnt << endl;
	cout << "u:" << uCnt << endl;
	return 0;
}
```

### Q10

```c++
#include <iostream>

using namespace std;

int main() {
    unsigned aCnt = 0, eCnt = 0, iCnt = 0, oCnt = 0, uCnt = 0;
    char ch;
    while (cin >> ch) {
        switch (ch) {
            case 'a':
            case 'A':
                ++aCnt;
                break;
            case 'e':
            case 'E':
                ++eCnt;
                break;
            case 'i':
            case 'I':
                ++iCnt;
                break;
            case 'o':
            case 'O':
                ++oCnt;
                break;
            case 'u':
            case 'U':
                ++uCnt;
                break;
        }
    }
	cout << "a:" << aCnt << endl;
	cout << "e:" << eCnt << endl;
	cout << "i:" << iCnt << endl;
	cout << "o:" << oCnt << endl;
	cout << "u:" << uCnt << endl;
    return 0;
}
```

### Q11

```c++
#include <iostream>

using namespace std;

int main() {
    unsigned aCnt = 0, eCnt = 0, iCnt = 0, oCnt = 0, uCnt = 0, 
             spaceCnt = 0, newlineCnt = 0, tabCnt = 0;
    char ch, prech;
    while (cin >> std::noskipws >> ch) {
        switch (ch) {
            case 'a':
            case 'A':
                ++aCnt;
                break;
            case 'e':
            case 'E':
                ++eCnt;
                break;
            case 'i':
            case 'I':
                ++iCnt;
                break;
            case 'o':
            case 'O':
                ++oCnt;
                break;
            case 'u':
            case 'U':
                ++uCnt;
                break;
            case ' ':
                ++spaceCnt;
                break;
            case '\v':
            case '\t':
                ++newlineCnt;
                break;
            case '\n':
                ++tabCnt;
                break;
        }
        prech = ch;
    }
	cout << "a:" << aCnt << endl;
	cout << "e:" << eCnt << endl;
	cout << "i:" << iCnt << endl;
	cout << "o:" << oCnt << endl;
	cout << "u:" << uCnt << endl;
	cout << " :" << spaceCnt << endl;
	cout << "\\n:" << newlineCnt << endl;
	cout << "\\v or \\t:" << tabCnt << endl;
    return 0;
}
```

### Q12

```c++
#include <iostream>

using namespace std;

int main() {
    unsigned aCnt = 0, eCnt = 0, iCnt = 0, oCnt = 0, uCnt = 0, 
             spaceCnt = 0, newlineCnt = 0, tabCnt = 0,
             fiCnt = 0, flCnt = 0, ffCnt = 0;
    char ch, prech;
    while (cin >> std::noskipws >> ch) {
        switch (ch) {
            case 'a':
            case 'A':
                ++aCnt;
                break;
            case 'e':
            case 'E':
                ++eCnt;
                break;
            case 'i':
                if (prech == 'f') ++fiCnt;
            case 'I':
                ++iCnt;
                break;
            case 'o':
            case 'O':
                ++oCnt;
                break;
            case 'u':
            case 'U':
                ++uCnt;
                break;
            case ' ':
                ++spaceCnt;
                break;
            case '\v':
            case '\t':
                ++newlineCnt;
                break;
            case '\n':
                ++tabCnt;
                break;
            case 'f':
                if (prech == 'f') ++ffCnt;
                break;
            case 'l':
                if (prech == 'f') ++flCnt;
                break;
        }
        prech = ch;
    }
	cout << "a:" << aCnt << endl;
	cout << "e:" << eCnt << endl;
	cout << "i:" << iCnt << endl;
	cout << "o:" << oCnt << endl;
	cout << "u:" << uCnt << endl;
	cout << " :" << spaceCnt << endl;
	cout << "\\n:" << newlineCnt << endl;
	cout << "\\v or \\t:" << tabCnt << endl;
	cout << "ff:" << ffCnt << endl;
	cout << "fl:" << flCnt << endl;
	cout << "fi:" << fiCnt << endl;
    return 0;
}
```

### Q13

```c++
unsigned aCnt = 0, eCnt = 0, iouCnt = 0;
char ch = next_text();
switch (ch) {
    case 'a': aCnt++; break;
    case 'e': eCnt++; break;
    default: iouCnt++; break;
}
```

```c++
unsigned index = some_value();
int ix;
switch (index) {
    case 1:
        ix = get_value();
        ivec[ ix ] = index;
        break;
    default:
        ix = ivec.size()-1;
        ivec[ ix ] = index;
}
```

```c++
unsigned evenCnt = 0, oddCnt = 0;
int digit = get_num() % 10;
switch (digit) {
    case 1: case 3: case 5: case 7: case 9:
        oddcnt++;
        break;
    case 2: case 4: case 6: case 8: case 0:
        evencnt++;
        break;
}
```

```c++
const unsigned ival=512, jval=1024, kval=4096;
unsigned bufsize;
unsigned swt = get_bufCnt();
switch(swt) {
    case ival:
        bufsize = ival * sizeof(int);
        break;
    case jval:
        bufsize = jval * sizeof(int);
        break;
    case kval:
        bufsize = kval * sizeof(int);
        break;
}
```

### Q14

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
    string s;
    if (cin >> s) {
        string item, maxitem;
        unsigned int cnt = 1,
                     maxcnt = cnt;
        while (cin >> item) {
            if (item == s) {
                ++cnt;
            }
            else {
                if (cnt > maxcnt) {
                    maxcnt = cnt;
                    maxitem = s;
                }
                s = item;
                cnt = 1;
            }
        }
        if (cnt > maxcnt) {
            maxcnt = cnt;
            maxitem = s;
        }
        if (maxcnt > 1) {
            cout << maxitem << ":" << maxcnt << endl;
        }
        else {
            cout << "no repeat!!!" << endl;
        }
    }
    return 0;
}
```

### Q15

ix只能在循环体内部使用

```c++
for (int ix = 0; ix != sz; ++ix) {}
```

循环缺少初始化语句

```c++
int ix;
for (int ix = 0; ix != sz; ++ix) {}
```

sz为0不会进入循环，sz不为0循环内部必须有结束循环的语句，否则会一直执行下去

```c++
for (int ix = 0; ix != sz; ++ix) {}
```

### Q16

已知迭代次数用for循环，否则用while循环

```c++
#include <iostream>

using namespace std;

int main() {
    int i = 0;
    while (i != 3) {
        cout << i << endl;
        i++;
    }
    return 0;
}
```

```c++
#include <iostream>

using namespace std;

int main() {
    for (int i = 0; i != 3; i++) {
        cout << i << endl;
    }
    return 0;
}
```

```c++
#include <iostream>

using namespace std;

int main() {
    int i;
    while (cin >> i) {
        cout << i << endl;
    }
    return 0;
}
```

```c++
#include <iostream>

using namespace std;

int main() {
    for (int i; cin >> i; ) {
        cout << i << endl;
    }
    return 0;
}
```

### Q17

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {
    bool res = true;
    vector<int> v1 = {0,1,1,2};
    vector<int> v2 = {0,1,1,2,3,5,8};
    for (decltype(v1.size()) i = 0; i != v1.size() && i != v2.size(); i++) {
        if (v1[i] != v2[i]) {
            res = false;
        }
    }
    cout << res << endl;
    return 0;
}
```

### Q18

```c++
do {
    int v1, v2;
    cout << "Please enter two numbers to sum:" ;
    if (cin >> v1 >> v2)
        cout << "Sum is: " << v1 + v2 << endl;
} while (cin);
```

```c++
int ival;
do {
    // . . .
} while (ival = get_response());
```

```c++
int ival;
do {
    ival = get_response();
} while (ival);
```

### Q19

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
    string a, b;
    if (cin >> a >> b) {
        do {
            if (a.size() == b.size()) {
                cout << a << " " << b << endl;
            }
            else if (a.size() < b.size()) {
                cout << a << endl;
            }
            else {
                cout << b << endl;
            }
        } while(cin >> a >> b);
    }
    return 0;
}
```

### Q20

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
    bool flag = true;
    string s, pres;
    if (cin >> s) {
        pres = s;
        while (cin >> s) {
            if (s == pres) {
                flag = false;
                cout << s << endl;
                break;
            }
            pres = s;
        }
        if (flag) {
            cout << "no repeat" << endl;
        }
    }
    return 0;
}
```

### Q21

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
    bool flag = true;
    string s, pres;
    if (cin >> s) {
        pres = s;
        while (cin >> s) {
            if (!isupper(s[0])) {
                pres = s;
                continue;
            }
            if (s == pres) {
                flag = false;
                cout << s << endl;
                break;
            }
            pres = s;
        }
        if (flag) {
            cout << "no repeat" << endl;
        }
    }
    return 0;
}
```

### Q22

```c++
int sz;
do {
    sz = get_size();
} while(sz <= 0);
```

### Q23

```c++
#include <iostream>

using namespace std;

int main() {
    int i1, i2;
    cin >> i1 >> i2;
    cout << i1/i2 << endl;
    return 0;
}
```

### Q24

Floating point exception

### Q25

```c++
#include <iostream>
#include <stdexcept>

using namespace std;

int main() {
    int i1, i2;
    while (cin >> i1 >> i2) {
        try {
            if (i2 == 0) {
                throw runtime_error("Divisor can't be 0!!!");
            }
            cout << i1/i2 << endl;
        }
        catch (runtime_error err) {
            cout << err.what() << endl;
            cout << "Try again? Enter y or n" << endl;
            char c;
            cin >> c;
            if (!cin || c == 'n')
                break;
        }
    }
    return 0;
}
```