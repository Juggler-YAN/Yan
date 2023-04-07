# Chapter 11

### Q1

map是关联容器，其元素是按关键字来保存和访问的；vector是顺序容器，其元素是按它们在容器中的位置来顺序保存和访问的。

### Q2

1. list：插入元素较多时使用；
2. vector：动态数组；
3. deque：需要头尾添加删除元素；
4. map：字典；
5. set：集合。

### Q3

```c++
#include <iostream>
#include <string>
#include <map>

using namespace std;

int main() {
    map<string, size_t> word_count;
    string word;
    while(cin >> word) {
        ++word_count[word];
    }
    for (const auto &w : word_count) {
        cout << w.first << " occurs " << w.second
                  << ((w.second > 1) ? " times" : " time") << endl;
    }
    return 0;
}
```

### Q4

```
I believe there is a person who brings sunshine into your life. That person may have enough to spread around. But if you really have to wait for someone to bring you the sun and give you a good feeling, then you may have to wait a long time.
```

```c++
#include <iostream>
#include <algorithm>
#include <string>
#include <map>
#include <cctype>

using namespace std;

int main() {
    map<string, size_t> word_count;
    string word;
    while(cin >> word) {
        word.erase(find_if(word.begin(), word.end(), ::ispunct), word.end());
        for_each(word.begin(), word.end(), [](char &c) {return c=tolower(c);});
        ++word_count[word];
    }
    for (const auto &w : word_count) {
        cout << w.first << " occurs " << w.second
                  << ((w.second > 1) ? " times" : " time") << endl;
    }
    return 0;
}
```

### Q5

map关键字-值对；set仅有关键字。

### Q6

set：关联容器，保存关键字；list：顺序容器，链表

### Q7

```c++
#include <iostream>
#include <map>
#include <vector>

using namespace std;

int main() {
    map<string, vector<string>> family;
    string lname, fname;
    while (cin >> lname >> fname) {
        family[lname].push_back(fname);
    }
    for (const auto &f : family) {
        cout << f.first << endl;
        for (const auto &s : f.second) {
            cout << s << " ";
        }
        cout << endl;
    }
    return 0;
}
```

### Q8

```c++
#include <iostream>
#include <set>
#include <vector>
#include <string>

using namespace std;

int main() {
    set<string> st;
    string s;
    while (cin >> s) {
        st.insert(s);
    }
    vector<string> v(st.begin(), st.end());
    for (const auto &i : v) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

### Q9

```c++
map<string, list<unsigned int>> wordline
```

### Q10

1. vector::iterator可以，因为定义了<；
2. list::iterator不可以，未定义<。

### Q11

```c++
multiset<Sales_data, bool (*)(const Sales_data &, const Sales_data &)> bookstore(compareIsbn);
```

### Q12

```c++
#include <iostream>
#include <utility>
#include <vector>
#include <string>

using namespace std;

int main() {
    string s;
    int n;
    vector<pair<string, int>> v;
    while (cin >> s >> n) {
        v.push_back(make_pair(s, n));
    }
    for (const auto &i : v) {
        cout << i.first << " " << i.second << endl;
    }
    return 0;
}
```

### Q13


```c++
#include <iostream>
#include <utility>
#include <vector>
#include <string>

using namespace std;

int main() {
    string s;
    int n;
    vector<pair<string, int>> v;
    while (cin >> s >> n) {
        pair<string, int> p(s, n);
        v.push_back(p);
    }
    for (const auto &i : v) {
        cout << i.first << " " << i.second << endl;
    }
    return 0;
}
```

```c++
#include <iostream>
#include <utility>
#include <vector>
#include <string>

using namespace std;

int main() {
    string s;
    int n;
    vector<pair<string, int>> v;
    while (cin >> s >> n) {
        pair<string, int> p = {s, n};
        v.push_back(p);
    }
    for (const auto &i : v) {
        cout << i.first << " " << i.second << endl;
    }
    return 0;
}
```

```c++
#include <iostream>
#include <utility>
#include <vector>
#include <string>

using namespace std;

int main() {
    string s;
    int n;
    vector<pair<string, int>> v;
    while (cin >> s >> n) {
        v.push_back(make_pair(s, n));
    }
    for (const auto &i : v) {
        cout << i.first << " " << i.second << endl;
    }
    return 0;
}
```

### Q14

```c++
#include <iostream>
#include <utility>
#include <map>
#include <vector>

using namespace std;

int main() {
    map<string, vector<pair<string, string>>> fmaily;
    string lname, fname, birth;
    while (cin >> lname >> fname >> birth) {
        fmaily[lname].push_back(make_pair(fname, birth));
    }
    for (const auto &f : fmaily) {
        cout << f.first << endl;
        for (const auto &s : f.second) {
            cout << s.first << " " << s.second << endl;
        }
    }
    return 0;
}
```

### Q15

```c++
mapped_type：vector<int>
key_type：int
value_type：pair<int, vector<int>>
```

### Q16

```c++
#include <iostream>
#include <map>
#include <string>

using namespace std;

int main() {
    map<string, string::size_type> m = {{"hello", 0}};
    map<string, string::size_type>::iterator iter = m.begin();
    iter->second = 5;
    for (const auto &i : m) {
        cout << i.first << " " << i.second << endl;
    }
    return 0;
}
```

### Q17

1. 合法
2. 非法，muliset中没有push_back操作
3. 合法
4. 合法

### Q18

```c++
map<string, size_t>::const_iterator
```

### Q19

```c++
multiset<Sales_data, bool (*)(const Sales_data &, const Sales_data &)> bookstore(compareIsbn)> bookstore(compareIsbn);
multiset<Sales_data, bool (*)(const Sales_data &, const Sales_data &)> bookstore(compareIsbn)>::iterator bookstore_iter = bookstore.begin();
```

### Q20

```c++
#include <iostream>
#include <string>
#include <map>

using namespace std;

int main() {
    map<string, size_t> word_count;
    string word;
    while(cin >> word) {
        auto ret = word_count.insert({word, 1});
        if (!ret.second) {
            ++ret.first->second;
        }
    }
    for (const auto &w : word_count) {
        cout << w.first << " occurs " << w.second
                  << ((w.second > 1) ? " times" : " time") << endl;
    }
    return 0;
}
```

### Q21

```c++
得到insert的返回值，是一个pair
word_count.insert({word, 0})
是pair的第一个成员，是一个map迭代器，指向具有给定关键字的元素
word_count.insert({word, 0}).first
map中元素的值部分
word_count.insert({word, 0}).first->second
递增此值
++word_count.insert({word, 0}).first->second
```

### Q22

```c++
参数类型：pair<string, vector<int>>
返回类型：pair<map<string, vector<int>>::iterator,bool>
```

### Q23

```c++
#include <iostream>
#include <map>
#include <vector>

using namespace std;

int main() {
    multimap<string, string> family;
    string lname, fname;
    while (cin >> lname >> fname) {
        family.insert({lname, fname});
    }
    for (auto f = family.begin(); f != family.end(); f = family.upper_bound(f->first)) {
        cout << f->first << endl;
        auto lu = family.equal_range(f->first);
        for (auto i = lu.first; i != lu.second; ++i){
            cout << i->second << " ";
        }
        cout << endl;
    }
    return 0;
}
```

### Q24

在m中添加一个关键字为0值为1的元素

### Q25

v是一个空容器，v[0]超出范围

### Q26

可以用key_type类型来对一个map进行下标操作；下标运算符返回的类型是mapped_type。举个例子，map<int, string>进行下标操作的类型为int，下标运算将要返回的类型string

### Q27

find判断是否存在特定元素的关键字；count统计特定元素的关键字数量

### Q28

```c++
#include <iostream>
#include <algorithm>
#include <map>

using namespace std;

int main() {
    map<string, int> m{{"hello",5},{"world",5}};
    cout << (m.find("hello")!=m.end()) << endl;
    cout << (m.find("hello1")!=m.end()) << endl;
	return 0;
}
```

### Q29

如果没有元素与给定关键字匹配，则lower_bound和upper_bound会返回相等的迭代器——指向一个不影响排序的关键字插入位置。如果equal_range未匹配到元素，则两个迭代器都指向关键字可以插入的位置。

### Q30

第一个与search_item匹配元素的书名

### Q31

```c++
#include <iostream>
#include <map>
#include <string>

using namespace std;

int main() {
    pair<string, string> item("a", "aa");
    multimap<string, string> m{{"a","a"},{"a","aa"},{"a","aaa"},{"b","cc"}};
    for (auto beg = m.lower_bound(item.first), end = m.upper_bound(item.first); beg != end; ++beg) {
        if (beg->second == item.second) {
            m.erase(beg);
            break;
        }
    }
    for (const auto &i : m) {
        cout << i.first << " " << i.second << endl;
    }
    return 0;
}
```

### Q32

```c++
#include <iostream>
#include <map>
#include <set>
#include <string>

using namespace std;

int main() {
    multimap<string, string> m{{"a","a"},{"b","cc"},{"a","aa"},{"a","aaa"}};
    map<string, set<string>> orderm;
    for (const auto &i : m) {
        orderm[i.first].insert(i.second);
    }
    for (const auto &i : orderm) {
        cout << i.first << endl;
        for (const auto &j : i.second) {
            cout << j << " ";
        }
        cout << endl;
    }
    return 0;
}
```

### Q33

```
// map.txt
brb be right back
k okay?
y why
r are
u you
pic picture
thk thanks!
18r later
```

```
// file.txt
where r u
y dont u send me a pic
k thk 18r
```

```c++
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <string>

using namespace std;

map<string, string> buildMap(ifstream &);
const string & transform(const string &, const map<string, string> &);
void word_transform(ifstream &, ifstream &);

int main() {
    ifstream map_file("./map.txt"), input("./file.txt");
    word_transform(map_file, input);
    map_file.close();
    input.close();
    return 0;
}

map<string, string> buildMap(ifstream &map_file) {
    map<string, string> trans_map;
    string key, value;
    while (map_file >> key && getline(map_file, value)) {
        if (value.size() > 1) {
            trans_map[key] = value.substr(1);
        }
        else {
            throw runtime_error("no rule for" + key);
        }
    }
    return trans_map;
}

const string & transform(const string &s, const map<string, string> &m) {
    auto map_it = m.find(s);
    if (map_it != m.cend()) {
        return map_it->second;
    }
    else {
        return s;
    }
}

void word_transform(ifstream &map_file, ifstream &input) {
    auto trans_map = buildMap(map_file);
    string text;
    while (getline(input, text)) {
        istringstream stream(text);
        string word;
        bool firstword = true;
        while (stream >> word) {
            if (firstword) {
                firstword = false;
            }
            else {
                cout <<  " ";
            }
            cout << transform(word, trans_map);
        }
        cout << endl;
    }
}
```

### Q34

当map中没有指定关键字元素时会插入该指定关键词元素，与预期不符。另外，返回含关键字元素的值而不是指向指定关键词的迭代器

### Q35

无影响，但如果关键词出现多次，下标操作会保存最后一次出现的元素，而插入保存第一次

### Q36

value.size()<1，将会抛出runtime_error “no rule for ” + key。

### Q37

无序容器可以获得更好的平均性能；有序容器可以自定义排序。

### Q38

```c++
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;

int main() {
    unordered_map<string, size_t> word_count;
    string word;
    while(cin >> word) {
        ++word_count[word];
    }
    for (const auto &w : word_count) {
        cout << w.first << " occurs " << w.second
                  << ((w.second > 1) ? " times" : " time") << endl;
    }
    return 0;
}
```

```
// map.txt
brb be right back
k okay?
y why
r are
u you
pic picture
thk thanks!
18r later
```

```
// file.txt
where r u
y dont u send me a pic
k thk 18r
```

```c++
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>

using namespace std;

unordered_map<string, string> buildMap(ifstream &);
const string & transform(const string &, const unordered_map<string, string> &);
void word_transform(ifstream &, ifstream &);

int main() {
    ifstream map_file("./map.txt"), input("./file.txt");
    word_transform(map_file, input);
    map_file.close();
    input.close();
    return 0;
}

unordered_map<string, string> buildMap(ifstream &map_file) {
    unordered_map<string, string> trans_map;
    string key, value;
    while (map_file >> key && getline(map_file, value)) {
        if (value.size() > 1) {
            trans_map[key] = value.substr(1);
        }
        else {
            throw runtime_error("no rule for" + key);
        }
    }
    return trans_map;
}

const string & transform(const string &s, const unordered_map<string, string> &m) {
    auto map_it = m.find(s);
    if (map_it != m.cend()) {
        return map_it->second;
    }
    else {
        return s;
    }
}

void word_transform(ifstream &map_file, ifstream &input) {
    auto trans_map = buildMap(map_file);
    string text;
    while (getline(input, text)) {
        istringstream stream(text);
        string word;
        bool firstword = true;
        while (stream >> word) {
            if (firstword) {
                firstword = false;
            }
            else {
                cout <<  " ";
            }
            cout << transform(word, trans_map);
        }
        cout << endl;
    }
}
```