# Chapter 9

### Q1

1. list，中间插入元素
2. deque，尾位置插入，头位置删除
3. vector，没有特别需求

### Q2

```c++
list<deque<int>> l;
```

### Q3

两个迭代器begin和end满足如下条件：它们指向同一个容器中的元素，或者是容器中的最后一个元素之后的位置，且我们可以通过反复递增begin来到达end。换句话说，end不在begin之前。

### Q4

```c++
#include <iostream>
#include <vector>

using namespace std;

bool find(vector<int>::const_iterator, vector<int>::const_iterator, int);

int main() {
    vector<int> v{1,2,3,4,5};
    int n;
    cin >> n;
    cout << find(v.cbegin(), v.cend(), n) << endl;
    return 0;
}

bool find(vector<int>::const_iterator beg, vector<int>::const_iterator end, int n) {
    while (beg != end) {
        if (*beg == n) {
            return true;
        }
        ++beg;
    }
    return false;
}
```

### Q5

```c++
#include <iostream>
#include <vector>

using namespace std;

vector<int>::const_iterator find(vector<int>::const_iterator, vector<int>::const_iterator, int);

int main() {
    vector<int> v{1,2,3,4,5};
    int n;
    cin >> n;
    vector<int>::const_iterator res = find(v.cbegin(), v.cend(), n);
    if (res != v.cend()) {
        cout << *res << endl;
    }
    else {
        cout << "No data!!!" << endl;
    }
    return 0;
}

vector<int>::const_iterator find(vector<int>::const_iterator beg, vector<int>::const_iterator end, int n) {
    while (beg != end) {
        if (*beg == n) {
            return beg;
        }
        ++beg;
    }
    return beg;
}
```

### Q6

```c++
list<int> lst1;
list<int>::iterator iter1 = lst1.begin(),
                    iter2 = lst1.end();
while (iter1 != iter2) /* ... */
```

### Q7

```c++
vector<int>::size_type
```

### Q8

```c++
list<string>::const_iterator
list<string>::iterator
```

### Q9

begin返回容器的iterator类型，cbegin返回容器的const_iterator类型。

### Q10

```c++
vector<int>::iterator，vector<int>::const_iterator
vector<int>::const_iterator，vector<int>::const_iterator
```

### Q11

```c++
vector<int> v1;
vector<int> v2(v1);
vector<int> v3 = v1;
vector<int> v4{1,2,3};
vector<int> v5 = {1,2,3};
vector<int> v6(v1.begin(), v1.end());
vector<int> v7(3);
vector<int> v8(3,1);
```

### Q12

为了创建一个容器为另一个容器的拷贝，两个容器的类型及其元素类型必须匹配；不过，当传递迭代器参数来拷贝一个范围时，就不要求容器类型是相同的了。而且，新容器和原容器中的元素类型也可以不同，只要能将要拷贝的元素转换为要初始化的容器的元素类型即可。

### Q13

```c++
#include <iostream>
#include <list>
#include <vector>

using namespace std;

int main() {
    list<int> l1{1,2,3,4,5};
    vector<double> v2(l1.begin(), l1.end());
    for (const auto &i : v2) {
        cout << i << " ";
    }
    cout << endl;

    vector<int> v3{1,2,3,4,5};
    vector<double> v4(v3.begin(), v3.end());
    for (const auto &i : v4) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

### Q14

```c++
#include <iostream>
#include <list>
#include <vector>
#include <string>

using namespace std;

int main() {
    list<char const *> l1{"Hello",",","world"};
    vector<string> v2;
    v2.assign(l1.begin(), l1.end());
    for (const auto &i : v2) {
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
    vector<int> v1{1,2,3};
    vector<int> v2{1,2,3};
    cout << boolalpha << (v1 == v2) << endl;
    return 0;
}
```

### Q16

```c++
#include <iostream>
#include <vector>
#include <list>

using namespace std;

int main() {
    vector<int> v1{1,2,3};
    list<int> l2{1,2,3};
    cout << boolalpha << (v1 == vector<int>(l2.begin(),l2.end())) << endl;
    return 0;
}
```

### Q17

c1和c2必须是相同类型的容器，且必须保存相同类型的元素，最后元素类型要支持该运算符。

### Q18

```c++
#include <iostream>
#include <string>
#include <deque>

using namespace std;

int main() {
    string s;
    deque<string> de;
    while (cin >> s) {
        de.push_back(s);
    }
    for (auto i = de.cbegin(); i != de.cend(); ++i) {
        cout << *i << " ";
    }
    cout << endl;
    return 0;
}
```

### Q19

```c++
#include <iostream>
#include <string>
#include <list>

using namespace std;

int main() {
    string s;
    list<string> l;
    while (cin >> s) {
        l.push_back(s);
    }
    for (auto i = l.cbegin(); i != l.cend(); ++i) {
        cout << *i << " ";
    }
    cout << endl;
    return 0;
}
```

### Q20

```c++
#include <iostream>
#include <list>
#include <deque>

using namespace std;

int main() {
    list<int> l{1,2,3,4,5};
    deque<int> de1, de2;
    for (const auto &i : l) {
        if (i % 2) {
            de1.push_back(i);
        }
        else {
            de2.push_back(i);
        }
    }
    for (const auto &i : de1) {
        cout << i << " ";
    }
    cout << endl;
    for (const auto &i : de2) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

### Q21

循环与list的工作原理是一样的

### Q22

迭代器失效

```c++
#include <iostream>
#include <vector>

using namespace std;

void fun(vector<int>& iv, int some_val);

int main() {
    vector<int> v{1,3,1,1,3,3,3,1};
    fun(v, 1);
    for (auto i : v) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}

void fun(vector<int>& iv, int some_val) {
    auto iter = iv.begin();
    auto mid = [&]{ return iv.begin() + iv.size() / 2; };   // 闭包，保存了已经无效的迭代器
    while (iter != mid()){
        if (*iter == some_val) {
            iter = iv.insert(iter, 2 * some_val);
            ++iter;
        }
        ++iter;
    }
}
```

### Q23

均为同一个元素

### Q24

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> v;
    cout << v[0] << endl;
    cout << v.at(0) << endl;
    cout << v.front() << endl;
    cout << *(v.begin()) << endl;
    return 0;
} 
```

### Q25

1. 如果elem1与elem2相等，则一个元素都不会删除；
2. 如果elem2是尾后迭代器，则会从elem1迭代器对应的元素删除到最后一个元素；
3. 如果elem1与elem2都是尾后迭代器，则一个元素都不会删除。

### Q26

```c++
#include <iostream>
#include <vector>
#include <list>

using namespace std;

int main() {
    int ia[] = {0,1,1,2,3,5,8,13,21,34,55,89};
    vector<int> v(begin(ia), end(ia));
    list<int> l(begin(ia), end(ia));
    for (auto i = v.begin(); i != v.end(); ++i) {
        if (!(*i % 2)) {
            i = v.erase(i);
            --i;
        }
    }
    for (auto i = l.begin(); i != l.end(); ++i) {
        if (*i % 2) {
            i = l.erase(i);
            --i;
        }
    }
    for (auto i : v) {
        cout << i << " ";
    }
    cout << endl;
    for (auto i : l) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

### Q27

```c++
#include <iostream>
#include <forward_list>

using namespace std;

int main() {
    forward_list<int> fl{1,2,3,4,5};
    auto prev = fl.before_begin();
    auto curr = fl.begin();
    while (curr != fl.end()) {
        if (*curr % 2) {
            curr = fl.erase_after(prev);
        }
        else {
            prev = curr;
            ++curr;
        }
    }
    for (auto i : fl) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

### Q28

```c++
#include <iostream>
#include <forward_list>
#include <string>

using namespace std;

void fun(forward_list<string>&, string, string);

int main() {
    forward_list<string> fl{"Hello",",","world!","Hello"};
    fun(fl,"Hello","my");
    for (auto i : fl) {
        cout << i << " ";
    }
    cout << endl;
    fun(fl,"Hello1","my");
    for (auto i : fl) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}

void fun(forward_list<string>& fl, string s1, string s2) {
    auto prev = fl.before_begin();
    auto curr = fl.begin();
    bool flag = true;
    while (curr != fl.end()) {
        if (*curr == s1) {
            curr = fl.insert_after(curr, s2);
            flag = false;
        }
        prev = curr;
        ++curr;
    }
    if (flag) {
        fl.insert_after(prev, s2);
    }
}
```

### Q29

1. 会添加75个新元素，并对新元素进行初始化；
2. 后面90个元素会被丢弃。

### Q30

如果元素类型的类类型，则元素类型必须提供一个默认构造函数。

### Q31

list迭代器不支持加法操作

```c++
#include <iostream>
#include <list>

using namespace std;

int main() {
	list<int> l1 = {0,1,2,3,4,5,6,7,8,9};
	auto iter = l1.begin();
	while (iter != l1.end()) {
		if(*iter % 2) {
			iter = l1.insert(iter, *iter);
			++iter;
			++iter;
		}
        else {
			iter = l1.erase(iter);
		}
	}

	for(const auto i : l1) {
		cout << i << " ";
    }
	cout << endl;

	return 0;
}
```

forward_list迭代器有专属的insert()和erase()

```c++
#include <iostream>
#include <forward_list>

using namespace std;

int main() {
	forward_list<int> l1 = {0,1,2,3,4,5,6,7,8,9};
    auto prev = l1.before_begin();
	auto iter = l1.begin();
	while (iter != l1.end()) {
		if(*iter % 2) {
			iter = l1.insert_after(iter, *iter);
            prev = iter;
			++iter;
		}
        else {
			iter = l1.erase_after(prev);
		}
	}

	for(const auto i : l1) {
		cout << i << " ";
    }
	cout << endl;

	return 0;
}
```

### Q32

不合法，insert中的参数运行顺序是未定义的。

### Q33

如果存储空间被重新分配，则指向容器的迭代器全部失效；如果未重新分配，插入位置之后的迭代器将会失效。

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {
	vector<int> v = {0,1,2,3,4,5,6,7,8,9};
	auto iter = v.begin();
	while(iter != v.end()) {
		++iter;
		// iter = v1.insert(iter, 42);
		v.insert(iter, 42);
		++iter;
	}

	for(const auto i : v) {
		cout << i << " ";
    }
	cout << endl;
	return 0;
}
```

### Q34

会无限循环，当碰到第一个奇数时，iter从insert()中得到插入在该奇数前元素的迭代器，++iter后，迭代器再次指向该奇数，程序陷入无限循环。

```c++
#include <iostream>
#include <list>

using namespace std;

int main() {
	list<int> l1 = {0,1,2,3,4,5,6,7,8,9};
	auto iter = l1.begin();
	while (iter != l1.end()) {
		if(*iter % 2) {
			iter = l1.insert(iter, *iter);
		}
		++iter;
	}

	for(const auto i : l1) {
		cout << i << " ";
    }
	cout << endl;

	return 0;
}
```

### Q35

容器的size是指它已经保存的元素的数目；而capacity则是在不分配新的内存空间的前提下最多可以保存多少元素。

### Q36

不可能。

### Q37

list存储空间是不连续的；array是固定size的。

### Q38

```c++
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> v;
    int num;
    while (cin >> num) {
        v.push_back(num);
        cout << v.size() << " " << v.capacity() << endl;
    }
    return 0;
}
```

### Q39

为svec分配至少能容纳1024个string的空间，将输入添加到svec中，将svec的size增加当前size的一半。

### Q40

1. 256，size为384，capacity为1024；
2. 512，size为768，capacity为1024；
3. 1000，size为1500，capacity至少为可以容纳当前size；
4. 1048，size为1572，capacity至少为可以容纳当前size。

### Q41

```c++
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main() {
    vector<char> v{'H','e','l','l','o'};
    string s(v.begin(), v.end());
    cout << s << endl;
    return 0;
}
```

### Q42

```c++
string s;
s.reserve(100);
```

### Q43

```c++
#include <iostream>
#include <string>

using namespace std;

void myreplace(string&, const string&, const string&);

int main() {
    string s("tho thru");
	myreplace(s, "tho", "though");
	cout << s << endl;
	myreplace(s, "thru", "through");
	cout << s << endl;
    return 0;
}

void myreplace(string& s, const string& oldVal, const string& newVal) {
    auto iter = s.begin();
    while (iter <= s.end() - oldVal.size()) {
        if (string(iter, iter + oldVal.size()) == oldVal) {
            s.erase(iter, iter + oldVal.size());
            s.insert(iter, newVal.begin(), newVal.end());
            iter += newVal.size();
        }
        else {
            ++iter;
        }
    }
}
```

### Q44

```c++
#include <iostream>
#include <string>

using namespace std;

void myreplace(string&, const string&, const string&);

int main() {
    string s("tho thru");
	myreplace(s, "tho", "though");
	cout << s << endl;
	myreplace(s, "thru", "through");
	cout << s << endl;
    return 0;
}

void myreplace(string& s, const string& oldVal, const string& newVal) {
    for (decltype(s.size()) i = 0; i != s.size(); ++i) {
        if (string(s, i, oldVal.size()) == oldVal) {
            s.replace(i, oldVal.size(), newVal);
            i += newVal.size() - oldVal.size();
        }
    }
}
```

### Q45

```c++
#include <iostream>
#include <string>

using namespace std;

void fun(string&, const string&, const string&);

int main() {
    string s{"Zhang"};
    fun(s, "Mr. ", " Jr.");
    cout << s << endl;
    return 0;
}

void fun(string& s, const string& pre, const string& post) {
    s.insert(s.begin(), pre.begin(), pre.end());
    s.append(post);
}
```

### Q46

```c++
#include <iostream>
#include <string>

using namespace std;

void fun(string&, const string&, const string&);

int main() {
    string s{"Zhang"};
    fun(s, "Mr. ", " Jr.");
    cout << s << endl;
    return 0;
}

void fun(string& s, const string& pre, const string& post) {
    s.insert(0, pre);
    s.insert(s.size(), post);
}
```

### Q47

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
    string s{"ab2c3d7R4E6"};
    string number{"0123456789"};
	string alphabet{"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"};
    string::size_type pos = 0;
    while ((pos = s.find_first_of(number, pos)) != string::npos) {
        cout << pos << ": " << s[pos] << endl;
        ++pos;
    }
    cout << "***********" << endl;
    pos = 0;
    while ((pos = s.find_first_of(alphabet, pos)) != string::npos) {
        cout << pos << ": " << s[pos] << endl;
        ++pos;
    }
    cout << "***********" << endl;
    pos = 0;
    while ((pos = s.find_first_not_of(alphabet, pos)) != string::npos) {
        cout << pos << ": " << s[pos] << endl;
        ++pos;
    }
    cout << "***********" << endl;
    pos = 0;
    while ((pos = s.find_first_not_of(number, pos)) != string::npos) {
        cout << pos << ": " << s[pos] << endl;
        ++pos;
    }
    return 0;
}
```

### Q48

string::npos

### Q49

```
i believe there is a person who brings sunshine into your life  that person may have enough to spread around  but if you really have to wait for someone to bring you the sun and give you a good feeling  then you may have to wait a long time
```

```c++
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

int main() {
    string s{"acemnorsuvwxz"};
    ifstream in("test.txt");
    if (in) {
        string word;
        string maxword;
        while (in >> word) {
            if (word.find_first_not_of(s) == string::npos) {
                if (word.size() > maxword.size()) {
                    maxword = word;
                }
            }
        }
        cout << maxword <<endl;
    }
    in.close();
    return 0;
}
```

### Q50

```c++
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main() {
    string res;

    vector<string> v1{"1","2","3"};
    int sum1 = 0.0;
    for (const auto &i : v1) {
        sum1 += stod(i);
    }
    res = to_string(sum1);
    cout << res << endl;

    vector<string> v2{"1.0","2.0","3.0"};
    float sum2 = 0.0;
    for (const auto &i : v2) {
        sum2 +=  stof(i);
    }
    res = to_string(sum2);
    cout << res << endl;

    return 0;
}
```

### Q51

```c++
// Date.h
#ifndef DATE_H
#define DATE_H

#include <string>

using namespace std;

class Date {
public:
    Date(const string&);
private:
    unsigned int year;
    unsigned int month;
    unsigned int day;
};

Date::Date(const string& s) {
    string date = s;
    string::size_type i1 = 0, i2 = 0;
    if (s.find(',') != string::npos) {
		i1 = s.find(' ');
		i2 = s.find(',', i1+1);
		if(s.find("January") != string::npos) month = 1;
		if(s.find("February") != string::npos) month = 2;
		if(s.find("March") != string::npos) month = 3;
		if(s.find("April") != string::npos) month = 4;
		if(s.find("May") != string::npos) month = 5;
		if(s.find("June") != string::npos) month = 6;
		if(s.find("July") != string::npos) month = 7;
		if(s.find("August") != string::npos) month = 8;
		if(s.find("September") != string::npos) month = 9;
		if(s.find("October") != string::npos) month = 10;
		if(s.find("November") != string::npos) month = 11;
		if(s.find("December") != string::npos) month = 12;
		day = stoi(s.substr(i1+1, i2-i1-1));
		year = stoi(s.substr(i2+1, s.size()));
    }
    else if (s.find('/') != string::npos) {
		i1 = s.find('/');
		i2 = s.find('/', i1+1);
		month = stoi(s.substr(0, i1));
		day = stoi(s.substr(i1+1, i2-i1-1));
		year = stoi(s.substr(i2+1, s.size()));
    }
    else {
		i1 = s.find(' ');
		i2 = s.find(' ', i1+1);
		if(s.find("Jan") != string::npos)  month = 1;
		if(s.find("Feb") != string::npos)  month = 2;
		if(s.find("Mar") != string::npos)  month = 3;
		if(s.find("Apr") != string::npos)  month = 4;
		if(s.find("May") != string::npos)  month = 5;
		if(s.find("Jun") != string::npos)  month = 6;
		if(s.find("Jul") != string::npos)  month = 7;
		if(s.find("Aug") != string::npos)  month = 8;
		if(s.find("Sep") != string::npos)  month = 9;
		if(s.find("Oct") != string::npos)  month = 10;
		if(s.find("Nov") != string::npos)  month = 11;
		if(s.find("Dec") != string::npos)  month = 12;
		day = stoi(s.substr(i1+1, i2-i1-1));
		year = stoi(s.substr(i2+1, s.size()));
    }
}

#endif
```

```c++
#include <iostream>
#include "Date.h"

int main() {
	Date date1("January 1, 1900");
	Date date2("1/1/1900");
	Date date3("Jan 1 1900");
    return 0;
}
```

### Q52

```c++
#include <iostream>
#include <string>
#include <stack>

using namespace std;

int main() {
    string s{"Hello,(world)!"};
    stack<char> stk;
    bool flag = false;
    for (const auto &i : s) {
        if (i == '(') {
            flag = true;
            continue;
        }
        else if (i == ')') {
            flag = false;
        }
        if (flag) stk.push(i);
    }
    string rep;
    while (!stk.empty()) {
        rep.push_back(stk.top());
        stk.pop();
    }
    for (decltype(rep.size()) i = 0; i != rep.size() / 2; ++i) {
        char temp;
        temp = rep[i];
        rep[i] = rep[rep.size()-i-1];
        rep[rep.size()-i-1] = temp;
    }
    s.replace(s.find("("), rep.size()+2, rep);
    cout << s << endl;
    return 0;
}
```