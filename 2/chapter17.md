### Q1

```c++
#include <iostream>
#include <tuple>

using namespace std;

int main() {
    tuple<int, int, int> threeD{10, 20, 30};
    cout << get<0>(threeD) << endl;
    cout << get<1>(threeD) << endl;
    cout << get<2>(threeD) << endl;
    return 0;
}
```

### Q2

```c++
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <tuple>

using namespace std;

int main() {
    tuple<string, vector<string>, pair<string, int>> three{"a", {"b1", "b2", "b3"}, {"c1", 3}};
    cout << get<0>(three) << endl;
    for (const auto &i : get<1>(three)) {
        cout << i << endl;
    }
    cout << get<2>(three).first << " " << get<2>(three).second << endl;
    return 0;
}
```

### Q3

```c++
// TextQuery.h
#ifndef TEXTQUERY_H
#define TEXTQUERY_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <tuple>

class QueryResult;
class TextQuery {
public:
    using line_no = vector<string>::size_type;
    TextQuery(ifstream&);
    tuple<string, shared_ptr<set<TextQuery::line_no>>, shared_ptr<vector<string>>> query(const string&) const;
private:
    shared_ptr<vector<string>> file;
    map<string, shared_ptr<set<line_no>>> wm;
};

class QueryResult {
friend ostream& print(ostream&, const QueryResult&);
public:
    QueryResult(string s,
                shared_ptr<set<TextQuery::line_no>> p,
                shared_ptr<vector<string>> f) :
        sought(s), lines(p), file(f) {}
    auto begin() const { return lines->cbegin(); }
    auto end() const { return lines->cend(); }
    auto get_file() const { return file; }
private:
    string sought;
    shared_ptr<set<TextQuery::line_no>> lines;
    shared_ptr<vector<string>> file;
};

TextQuery::TextQuery(ifstream &is) : file(new vector<string>) {
    string text;
    while (getline(is, text)) {
        file->push_back(text);
        int n = file->size() - 1;
        istringstream line(text);
        string word;
        while (line >> word) {
            auto &lines = wm[word];
            if (!lines)
                lines.reset(new set<line_no>);
            lines->insert(n);
        }
    }
}

tuple<string, shared_ptr<set<TextQuery::line_no>>, shared_ptr<vector<string>>> TextQuery::query(const string &sought) const {
    static shared_ptr<set<line_no>> nodata(new set<line_no>);
    auto loc = wm.find(sought);
    if (loc == wm.end())
        return tuple<string, shared_ptr<set<TextQuery::line_no>>, shared_ptr<vector<string>>>(sought, nodata, file);
    else
        return tuple<string, shared_ptr<set<TextQuery::line_no>>, shared_ptr<vector<string>>>(sought, loc->second, file);
}

string make_plural(size_t ctr, const string &word, const string &ending) {
    return (ctr > 1) ? word + ending : word;
}

ostream &print(ostream & os, tuple<string, shared_ptr<set<TextQuery::line_no>>, shared_ptr<vector<string>>> qr) {
    os << get<0>(qr) << " occurs " << get<1>(qr)->size() << " "
        << make_plural(get<1>(qr)->size(), "times", "s") << endl;
    for (auto num : *get<1>(qr))
        os << "\t(line " << num+1 << ") " << *(get<2>(qr)->begin()+num) << endl;
    return os;
}

#endif
```

```c++
#include <iostream>
#include <fstream>
#include "TextQuery.h"

void runQueries(ifstream &);

int main() {
    ifstream in("./data/17-3");
    runQueries(in);
    return 0;
}

void runQueries(ifstream &infile) {
    TextQuery tq(infile);
    while (true) {
        cout << "enter word to look for, or q to quit: ";
        string s;
        if (!(cin >> s) || s == "q") break;
        print(cout, tq.query(s)) << endl;
    }
}
```

### Q4

```c++
// Sales_data.h
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <tuple>

struct Sales_data {

    friend istream& operator>>(istream&, Sales_data&);
    friend ostream& operator<<(ostream&, const Sales_data&);
    friend Sales_data operator+(const Sales_data&, const Sales_data&);
    friend bool operator==(const Sales_data&, const Sales_data&);
	friend class hash<Sales_data>;

public:
    Sales_data(string s, unsigned n, double p) :
                bookNo(s), units_sold(n), revenue(p*n) {};
    Sales_data() : Sales_data("", 0, 0) {}
    Sales_data(string s) : Sales_data(s, 0, 0) {}
    Sales_data(istream &is) : Sales_data() { is >> *this; }
    string isbn() const { return bookNo; }
    Sales_data& operator+=(const Sales_data&);

private:
    inline double avg_price() const;
    string bookNo;
    unsigned units_sold = 0;
    double revenue = 0.0;

};

inline double Sales_data::avg_price() const {
	if (units_sold) {
		return revenue / units_sold;
    }
	else {
		return 0;
    }
}

Sales_data& Sales_data::operator+=(const Sales_data &rhs) {
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}

istream& operator>>(istream &is, Sales_data &item) {
	double price = 0;
	is >> item.bookNo >> item.units_sold >> price;
    if (is) {
        item.revenue = price * item.units_sold;
    }
	else {
        item = Sales_data();
    }
	return is;
}

ostream& operator<<(ostream &os, const Sales_data &item) {
	os << item.isbn() << " " << item.units_sold << " " << item.revenue << " " << item.avg_price();
	return os;
}

Sales_data operator+(const Sales_data &lhs, const Sales_data &rhs) {
	Sales_data sum = lhs;
	sum += rhs;
	return sum;
}

bool operator==(const Sales_data &lhs, const Sales_data &rhs) {
	 return lhs.isbn() == rhs.isbn() && 
        lhs.units_sold == rhs.units_sold && 
        lhs.revenue == rhs.revenue;
}

bool compareIsbn(const Sales_data &sale1, const Sales_data &sale2) {
    return sale1.isbn() < sale2.isbn();
}

typedef tuple<vector<Sales_data>::size_type,
                   vector<Sales_data>::const_iterator,
                   vector<Sales_data>::const_iterator> matches;

vector<matches> findBook(const vector<vector<Sales_data>> &files,
                              const string &book) {
    vector<matches> ret;
    for (auto it = files.cbegin(); it != files.cend(); ++it) {
        auto found = equal_range(it->cbegin(), it->cend(), book, compareIsbn);
        if (found.first != found.second) {
            ret.push_back(make_tuple(it-files.cbegin(), found.first, found.second));
        }
    }
    return ret;
}

void reportResults(istream &in, ostream &os, const vector<vector<Sales_data>> &files) {
    string s;
    while (in >> s) {
        auto trans = findBook(files, s);
        if (trans.empty()) {
            cout << s << " not found in any stores" << endl;
            continue;
        }
        for (const auto &store : trans)
            os << "store " << get<0>(store) << " sales " << accumulate(get<1>(store), 
                get<2>(store), Sales_data(s)) << endl;
    }
}

#endif
```

```c++
#include <iostream>
#include <vector>
#include "Sales_data.h"

int main() {
    Sales_data s1("a", 1, 100);
    Sales_data s2("a", 1, 200);
    Sales_data s3("b", 2, 100);
    vector<Sales_data> v1 = {s1, s3};
    vector<Sales_data> v2 = {s2};
    vector<vector<Sales_data>> v = {v1, v2};
    reportResults(cin, cout, v);
    return 0;
}
```

### Q5

```c++
// Sales_data.h
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <algorithm>
#include <utility>
#include <numeric>
#include <vector>
#include <string>

struct Sales_data {

    friend istream& operator>>(istream&, Sales_data&);
    friend ostream& operator<<(ostream&, const Sales_data&);
    friend Sales_data operator+(const Sales_data&, const Sales_data&);
    friend bool operator==(const Sales_data&, const Sales_data&);
	friend class hash<Sales_data>;

public:
    Sales_data(string s, unsigned n, double p) :
                bookNo(s), units_sold(n), revenue(p*n) {};
    Sales_data() : Sales_data("", 0, 0) {}
    Sales_data(string s) : Sales_data(s, 0, 0) {}
    Sales_data(istream &is) : Sales_data() { is >> *this; }
    string isbn() const { return bookNo; }
    Sales_data& operator+=(const Sales_data&);

private:
    inline double avg_price() const;
    string bookNo;
    unsigned units_sold = 0;
    double revenue = 0.0;

};

inline double Sales_data::avg_price() const {
	if (units_sold) {
		return revenue / units_sold;
    }
	else {
		return 0;
    }
}

Sales_data& Sales_data::operator+=(const Sales_data &rhs) {
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}

istream& operator>>(istream &is, Sales_data &item) {
	double price = 0;
	is >> item.bookNo >> item.units_sold >> price;
    if (is) {
        item.revenue = price * item.units_sold;
    }
	else {
        item = Sales_data();
    }
	return is;
}

ostream& operator<<(ostream &os, const Sales_data &item) {
	os << item.isbn() << " " << item.units_sold << " " << item.revenue << " " << item.avg_price();
	return os;
}

Sales_data operator+(const Sales_data &lhs, const Sales_data &rhs) {
	Sales_data sum = lhs;
	sum += rhs;
	return sum;
}

bool operator==(const Sales_data &lhs, const Sales_data &rhs) {
	 return lhs.isbn() == rhs.isbn() && 
        lhs.units_sold == rhs.units_sold && 
        lhs.revenue == rhs.revenue;
}

bool compareIsbn(const Sales_data &sale1, const Sales_data &sale2) {
    return sale1.isbn() < sale2.isbn();
}

typedef pair<vector<Sales_data>::size_type,
    pair<vector<Sales_data>::const_iterator, vector<Sales_data>::const_iterator>> matches;

vector<matches> findBook(const vector<vector<Sales_data>> &files,
                              const string &book) {
    vector<matches> ret;
    for (auto it = files.cbegin(); it != files.cend(); ++it) {
        auto found = equal_range(it->cbegin(), it->cend(), book, compareIsbn);
        if (found.first != found.second) {
            ret.push_back(make_pair(it-files.cbegin(), make_pair(found.first, found.second)));
        }
    }
    return ret;
}

void reportResults(istream &in, ostream &os, const vector<vector<Sales_data>> &files) {
    string s;
    while (in >> s) {
        auto trans = findBook(files, s);
        if (trans.empty()) {
            cout << s << " not found in any stores" << endl;
            continue;
        }
        for (const auto &store : trans)
            os << "store " << store.first << " sales " << accumulate(store.second.first, 
                store.second.second, Sales_data(s)) << endl;
    }
}

#endif
```

```c++
#include <iostream>
#include <vector>
#include "Sales_data.h"

int main() {
    Sales_data s1("a", 1, 100);
    Sales_data s2("a", 1, 200);
    Sales_data s3("b", 2, 100);
    vector<Sales_data> v1 = {s1, s3};
    vector<Sales_data> v2 = {s2};
    vector<vector<Sales_data>> v = {v1, v2};
    reportResults(cin, cout, v);
    return 0;
}
```

### Q6

```c++
// Sales_data.h
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <algorithm>
#include <numeric>
#include <vector>
#include <string>

struct Sales_data {

    friend istream& operator>>(istream&, Sales_data&);
    friend ostream& operator<<(ostream&, const Sales_data&);
    friend Sales_data operator+(const Sales_data&, const Sales_data&);
    friend bool operator==(const Sales_data&, const Sales_data&);
	friend class hash<Sales_data>;

public:
    Sales_data(string s, unsigned n, double p) :
                bookNo(s), units_sold(n), revenue(p*n) {};
    Sales_data() : Sales_data("", 0, 0) {}
    Sales_data(string s) : Sales_data(s, 0, 0) {}
    Sales_data(istream &is) : Sales_data() { is >> *this; }
    string isbn() const { return bookNo; }
    Sales_data& operator+=(const Sales_data&);

private:
    inline double avg_price() const;
    string bookNo;
    unsigned units_sold = 0;
    double revenue = 0.0;

};

inline double Sales_data::avg_price() const {
	if (units_sold) {
		return revenue / units_sold;
    }
	else {
		return 0;
    }
}

Sales_data& Sales_data::operator+=(const Sales_data &rhs) {
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}

istream& operator>>(istream &is, Sales_data &item) {
	double price = 0;
	is >> item.bookNo >> item.units_sold >> price;
    if (is) {
        item.revenue = price * item.units_sold;
    }
	else {
        item = Sales_data();
    }
	return is;
}

ostream& operator<<(ostream &os, const Sales_data &item) {
	os << item.isbn() << " " << item.units_sold << " " << item.revenue << " " << item.avg_price();
	return os;
}

Sales_data operator+(const Sales_data &lhs, const Sales_data &rhs) {
	Sales_data sum = lhs;
	sum += rhs;
	return sum;
}

bool operator==(const Sales_data &lhs, const Sales_data &rhs) {
	 return lhs.isbn() == rhs.isbn() && 
        lhs.units_sold == rhs.units_sold && 
        lhs.revenue == rhs.revenue;
}

bool compareIsbn(const Sales_data &sale1, const Sales_data &sale2) {
    return sale1.isbn() < sale2.isbn();
}

struct matches {
    vector<Sales_data>::size_type index;
    vector<Sales_data>::const_iterator first;
    vector<Sales_data>::const_iterator last;
    matches(vector<Sales_data>::size_type i, vector<Sales_data>::const_iterator f, vector<Sales_data>::const_iterator l) : 
        index(i), first(f), last(l) {}
};

vector<matches> findBook(const vector<vector<Sales_data>> &files,
                              const string &book) {
    vector<matches> ret;
    for (auto it = files.cbegin(); it != files.cend(); ++it) {
        auto found = equal_range(it->cbegin(), it->cend(), book, compareIsbn);
        if (found.first != found.second) {
            ret.push_back(matches(it-files.cbegin(), found.first, found.second));
        }
    }
    return ret;
}

void reportResults(istream &in, ostream &os, const vector<vector<Sales_data>> &files) {
    string s;
    while (in >> s) {
        auto trans = findBook(files, s);
        if (trans.empty()) {
            cout << s << " not found in any stores" << endl;
            continue;
        }
        for (const auto &store : trans)
            os << "store " << store.index << " sales " << accumulate(store.first, 
                store.last, Sales_data(s)) << endl;
    }
}

#endif
```

```c++
#include <iostream>
#include <vector>
#include "Sales_data.h"

int main() {
    Sales_data s1("a", 1, 100);
    Sales_data s2("a", 1, 200);
    Sales_data s3("b", 2, 100);
    vector<Sales_data> v1 = {s1, s3};
    vector<Sales_data> v2 = {s2};
    vector<vector<Sales_data>> v = {v1, v2};
    reportResults(cin, cout, v);
    return 0;
}
```

### Q7

第一种，实现起来简单

### Q8

输出的Sales_data中bookNo为空

### Q9

用unsigned long long初始化：0000000000000000000000000000000000000000000000000000000000100000
用unsigned long long初始化：00000000000011110110100110110101
用string初始化：与cin有关

### Q10

```c++
#include <iostream>
#include <vector>
#include <bitset>

int main() {
    vector<int> v{1,2,3,5,8,13,21};
    bitset<32> b1;
    for (auto i : v) {
        b1.set(i);
    }
    cout << b1 << endl;
    bitset<32> b2(2105646ULL);
    cout << b2 << endl;
    return 0;
}
```

### Q11

```c++
#include <iostream>
#include <string>
#include <bitset>

template <unsigned> class quiz;
template <unsigned N>
    ostream& operator<<(ostream&, const quiz<N>&);

template <unsigned N>
class quiz {
    friend ostream& operator<<<N>(ostream&, const quiz<N>&);
public:
    quiz(const string &s) : b(s) {}
    bitset<N>& get_bitset() { return b; }
private:
    bitset<N> b;
};

template <unsigned N>
ostream& operator<<(ostream& os, const quiz<N> &q) {
    os << q.b;
    return os;
}

int main() {
    quiz<10> q1(string("01010101010101"));
    quiz<100> q2(string("01010101010101"));
    cout << q1 << endl;
    cout << q2 << endl;
    return 0;
}
```

### Q12

```c++
#include <iostream>
#include <string>
#include <bitset>

template <unsigned> class quiz;
template <unsigned N>
    ostream& operator<<(ostream&, const quiz<N>&);

template <unsigned N>
class quiz {
    friend ostream& operator<<<N>(ostream&, const quiz<N>&);
public:
    quiz(const string &s) : b(s) {}
    void update(size_t n, bool res) {
        if (n < N) {
            b[n] = res;
        }
    }
private:
    bitset<N> b;
};

template <unsigned N>
ostream& operator<<(ostream& os, const quiz<N> &q) {
    os << q.b;
    return os;
}

int main() {
    quiz<10> q(string("01010101010101"));
    cout << q << endl;
    q.update(1,true);
    cout << q << endl;
    return 0;
}
```

### Q13

```c++
#include <iostream>
#include <string>
#include <bitset>

template <unsigned> class quiz;
template <unsigned N>
    ostream& operator<<(ostream&, const quiz<N>&);
template <unsigned N>
    size_t grade(const quiz<N>&, const quiz<N>&);

template <unsigned N>
class quiz {
    friend ostream& operator<<<N>(ostream&, const quiz<N>&);
    friend size_t grade<N>(const quiz<N>&, const quiz<N>&);
public:
    quiz(const string &s) : b(s) {}
    void update(size_t n, bool res) {
        if (n < N) {
            b[n] = res;
        }
    }
private:
    bitset<N> b;
};

template <unsigned N>
ostream& operator<<(ostream& os, const quiz<N> &q) {
    os << q.b;
    return os;
}

template <unsigned N>
size_t grade(const quiz<N> &lhs, const quiz<N> &rhs) {
    return (lhs.b^rhs.b).flip().count();    
}

int main() {
    quiz<10> q1(string("0101010101"));
    quiz<10> q2(string("1101011100"));
    cout << grade(q1,q2) << endl;
    return 0;
}
```

### Q14

```c++
#include <iostream>
#include <regex>

int main() {
    try {
        regex r("[[:alnum:]+\\.(cpp|cxx|cc)$", regex::icase);
    }
    catch (regex_error e) {
        cout << e.what() << "\ncode: " << e.code() << endl;
    }
    try {
        regex r("[[:alnum:]]+\\.(cpp|cxx|cc$", regex::icase);
    }
    catch (regex_error e) {
        cout << e.what() << "\ncode: " << e.code() << endl;
    }
    return 0;
}
```

### Q15

```c++
#include <iostream>
#include <string>
#include <regex>

int main() {
    string pattern("[^c]ei");
    pattern = "[[:alpha:]]*" + pattern + "[[:alpha:]]*";
    regex r(pattern);
    smatch results;
    string s;
    while (cin >> s) {
        if (regex_search(s, results, r)) {
            cout << s << " : correct." << endl;
            cout << results.str() << endl;
        }
        else {
            cout << s << " : error." << endl;
        }
    }
    return 0;
}
```

### Q16

匹配得到的结果不同，匹配成功后结果只有3个字符

```c++
#include <iostream>
#include <string>
#include <regex>

int main() {
    string pattern("[^c]ei");
    regex r(pattern);
    smatch results;
    string s;
    while (cin >> s) {
        if (regex_search(s, results, r)) {
            cout << s << " : correct." << endl;
            cout << results.str() << endl;
        }
        else {
            cout << s << " : error." << endl;
        }
    }
    return 0;
}
```

### Q17

```c++
#include <iostream>
#include <string>
#include <regex>

int main() {
    string pattern("[^c]ei");
    pattern = "[[:alpha:]]*" + pattern + "[[:alpha:]]*";
    regex r(pattern, regex::icase);
	string file("freind receipt theif receive");
    for (sregex_iterator it(file.begin(), file.end(), r), end_it; it != end_it; ++it)
        cout << it->str() << endl;
    return 0;
}
```

### Q18

```c++
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <regex>

int main() {
    string pattern("[^c]ei");
    pattern = "[[:alpha:]]*" + pattern + "[[:alpha:]]*";
    regex r(pattern, regex::icase);
	string file("albeit neighbor freind receipt theif receive");
    vector<string> v{"albeit", "neighbor"};
    for (sregex_iterator it(file.begin(), file.end(), r), end_it; it != end_it; ++it) {
        if (find(v.begin(), v.end(),it->str()) != v.end()) {
            continue;
        }
        cout << it->str() << endl;
    }
    return 0;
}
```

### Q19

没有匹配返回空字符串，也是可以比较的

### Q20

```c++
#include <iostream>
#include <string>
#include <regex>

bool valid(const smatch&);

int main() {
    string phone = "(\\()?(\\d{3})(\\))?([-. ])?(\\d{3})([-. ])?(\\d{4})";
    regex r(phone);
    smatch m;
    string s;
    while (getline(cin, s)) {
        for (sregex_iterator it(s.begin(), s.end(), r), end_it; it != end_it; ++it) {
            if (valid(*it)) {
                cout << "valid: " << it->str() << endl;
            }
            else {
                cout << "not valid: " << it->str() << endl;
            }
        }
    }
    return 0;
}

bool valid(const smatch &m) {
    if (m[1].matched) {
        return m[3].matched && (m[4].matched == 0 || m[4].str() == " ");
    }
    else {
        return !m[3].matched && m[4].str() == m[6].str();
    }
}
```

### Q21

```c++
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <regex>

bool valid(const smatch&);

struct PersonInfo {
    string name;
    vector<string> phones;
};

int main() {
	string line, word;
	vector<PersonInfo> people;
	istringstream record;
    ifstream in("./data/17-21");

	while(getline(in, line)) {
		record.str(line);
		PersonInfo info;
		record >> info.name;
		while(record >> word)
			info.phones.push_back(word);
		record.clear();
		people.push_back(info);
	}

    
    string phone = "(\\()?(\\d{3})(\\))?([-. ])?(\\d{3})([-. ])?(\\d{4})";
    regex r(phone);
    smatch m;

	for(const auto &person : people) {
		cout << person.name << "  ";
		for(const auto &ph : person.phones) {
            for (sregex_iterator it(ph.begin(), ph.end(), r), end_it; it != end_it; ++it) {
                if (valid(*it)) {
                    cout << it->str() << " ";
                }
            }
		}
		cout << endl;
	}

	in.close();

	return 0;
}

bool valid(const smatch &m) {
    if (m[1].matched) {
        return m[3].matched && (m[4].matched == 0 || m[4].str() == " ");
    }
    else {
        return !m[3].matched && m[4].str() == m[6].str();
    }
}
```

### Q22

```c++
#include <iostream>
#include <string>
#include <regex>

bool valid(const smatch&);

int main() {
    string phone = "(\\()?(\\d{3})(\\))?([-. ])?([ ]*)?(\\d{3})([-. ])?([ ]*)?(\\d{4})";
    regex r(phone);
    smatch m;
    string s;
    while (getline(cin, s)) {
        for (sregex_iterator it(s.begin(), s.end(), r), end_it; it != end_it; ++it) {
            if (valid(*it)) {
                cout << "valid: " << it->str() << endl;
            }
            else {
                cout << "not valid: " << it->str() << endl;
            }
        }
    }
    return 0;
}

bool valid(const smatch &m) {
    if (m[1].matched) {
        return m[3].matched && (m[4].matched == 0 || m[4].str() == " ");
    }
    else {
        return !m[3].matched && m[4].str() == m[7].str();
    }
}
```

### Q23

```c++
#include <iostream>
#include <string>
#include <regex>

bool valid(const smatch&);

int main() {
    string mail = "(\\d{5})(-)?(\\d{4})?";
    regex r(mail);
    smatch m;
    string s;
    while (getline(cin, s)) {
        for (sregex_iterator it(s.begin(), s.end(), r), end_it; it != end_it; ++it) {
            if (valid(*it)) {
                cout << "valid: " << it->str() << endl;
            }
            else {
                cout << "not valid: " << it->str() << endl;
            }
        }
    }
    return 0;
}

bool valid(const smatch &m) {
    if (!m[3].matched) {
        return !m[2].matched;
    }
    return true;
}
```

### Q24

```c++
#include <iostream>
#include <string>
#include <regex>

int main() {
    string phone = "(\\()?(\\d{3})(\\))?([-. ])?(\\d{3})([-. ])?(\\d{4})";
    regex r(phone);
    smatch m;
    string s;
    string fmt = "$2.$5.$7";
    while (getline(cin, s)) {
        cout << regex_replace(s, r, fmt) << endl;
    }
    return 0;
}
```

### Q25

```c++
#include <iostream>
#include <fstream>
#include <string>
#include <regex>

int main() {
    string phone = "(\\()?(\\d{3})(\\))?([-. ])?(\\d{3})([-. ])?(\\d{4})";
    regex r(phone);
    string fmt("$2.$5.$7");
    ifstream in("./data/17-25");
    string line;
	while(getline(in, line)) {
    	smatch m;
		regex_search(line, m, r);
		if (!m.empty()) {
			cout << m.prefix().str() << m.format(fmt) << endl;
		}
	}
	in.close();
	return 0;
}
```

### Q26

```c++
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <regex>

bool valid(const smatch&);

int main() {
    string phone = "(\\()?(\\d{3})(\\))?([-. ])?(\\d{3})([-. ])?(\\d{4})";
    regex r(phone);
    smatch m;
    ifstream in("./data/17-26");
    string line;
	while(getline(in, line)) {
        vector<string> v;
        for (sregex_iterator it(line.begin(), line.end(), r), end_it; it != end_it; ++it) {
            if (valid(*it)) {
                v.push_back(it->str());
            }
        }
        if (v.size() == 0) {
            continue;
        }
        else if (v.size() == 1) {
            cout << v[0] << endl;
        }
        else {
            for (decltype(v.size()) i = 1; i != v.size(); ++i) {
                cout << v[i] << " ";
            }
            cout << endl;
        }
	}
	in.close();
	return 0;
}

bool valid(const smatch &m) {
    if (m[1].matched) {
        return m[3].matched && (m[4].matched == 0 || m[4].str() == " ");
    }
    else {
        return !m[3].matched && m[4].str() == m[6].str();
    }
}
```

### Q27

```c++
#include <iostream>
#include <string>
#include <regex>

bool valid(const smatch&);

int main() {
    string mail = "(\\d{5})(-)?(\\d{4})?";
    regex r(mail);
	string fmt("$1-$3");
    smatch m;
    string s;
    while (getline(cin, s)) {
        for (sregex_iterator it(s.begin(), s.end(), r), end_it; it != end_it; ++it) {
            if (valid(*it)) {
                cout << (*it).format(fmt) << endl;
            }
        }
    }
    return 0;
}

bool valid(const smatch &m) {
    if (!m[3].matched) {
        return false;
    }
    return true;
}
```

### Q28

```c++
#include <iostream>
#include <random>

unsigned int myrand();

int main() {
    cout << myrand() << endl;
    cout << myrand() << endl;
    return 0;
}

unsigned int myrand() {
    static default_random_engine e;
    static uniform_int_distribution<unsigned int> u;
    return u(e);
}
```

### Q29

```c++
#include <iostream>
#include <random>

unsigned int myrand();
unsigned int myrand(unsigned int);

int main() {
    cout << myrand() << endl;
    cout << myrand() << endl;
    cout << myrand(2) << endl;
    cout << myrand(2) << endl;
    return 0;
}

unsigned int myrand() {
    static default_random_engine e;
    static uniform_int_distribution<unsigned int> u;
    return u(e);
}

unsigned int myrand(unsigned int i) {
    static default_random_engine e(i);
    static uniform_int_distribution<unsigned int> u;
    return u(e);
}
```

### Q30

```c++
#include <iostream>
#include <random>

unsigned int myrand();
unsigned int myrand(unsigned int);
unsigned int myrand(unsigned int, unsigned int, unsigned int);

int main() {
    cout << myrand() << endl;
    cout << myrand() << endl;
    cout << myrand(2) << endl;
    cout << myrand(2) << endl;
    cout << myrand(2,0,9) << endl;
    cout << myrand(2,0,9) << endl;
    return 0;
}

unsigned int myrand() {
    static default_random_engine e;
    static uniform_int_distribution<unsigned int> u;
    return u(e);
}

unsigned int myrand(unsigned int i) {
    static default_random_engine e(i);
    static uniform_int_distribution<unsigned int> u;
    return u(e);
}

unsigned int myrand(unsigned int i, unsigned int minval, unsigned int maxval) {
    static default_random_engine e(i);
    static uniform_int_distribution<unsigned int> u(minval, maxval);
    return u(e);
}
```

### Q31

每次生成的随机数都相同，first恒为某一bool值

### Q32

报错，未定义resp

### Q33

```c++
#include <map>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <ctime>
#include <vector>
#include <random>

using namespace std;

map<string, vector<string>> buildMap(ifstream &map_file)
{
	map<string, vector<string>> trans_map;
	string key;
	string value;
	while(map_file >> key && getline(map_file, value))
		if(value.size() > 1)
			trans_map[key].push_back(value.substr(1));
		else
			throw runtime_error("no rule for " + key);
	return trans_map;
}

const string &transform(const string &s, const map<string, vector<string>> &m)
{
	auto map_it = m.find(s);
	if(map_it != m.cend())
        if ((map_it->second).size() == 1) {
            return (map_it->second)[0];
        }
        else {
            static default_random_engine e(time(0));
            static uniform_int_distribution<unsigned> u(0,(map_it->second).size()-1);
		    return (map_it->second)[u(e)];
        }
	else
		return s;
}

void word_tranform(ifstream &map_file, ifstream &input)
{
	auto trans_map = buildMap(map_file);
	// for(const auto p : trans_map)
	// 	cout << p.first << "->" << p.second << endl;
	string text;
	while(getline(input, text))
	{
		istringstream stream(text);
		string word;
		bool firstword = true;
		while(stream >> word)
		{
			if(firstword)
				firstword = false;
			else
				cout << " ";
			cout << transform(word, trans_map);
		}
		cout << endl;
	}
}

int main()
{
	ifstream map_file("data/17-33-1"), input("data/17-33-2");
	word_tranform(map_file, input);

	return 0;
}
```

### Q34

```c++
#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    cout << true << " " << boolalpha << true << " " << noboolalpha << endl;
    cout << 20 << " " 
              << oct << 20 << " "
              << hex << 20 << " "
              << dec << 20 << " "
              << showbase
              << 20 << " "
              << oct << 20 << " "
              << uppercase
              << hex << 20 << " "
              << nouppercase
              << hex << 20 << " "
              << dec << 20 << " "
              << noshowbase
              << endl;
    cout << sqrt(2.0) << " "
              << cout.precision(12) << " " << sqrt(2.0) << " "
              << setprecision(3) << sqrt(2.0) 
              << setprecision(6)
              << endl;
    cout << 100 * sqrt(2.0) << " "
              << scientific << 100 * sqrt(2.0) << " "
              << fixed << 100 * sqrt(2.0) << " "
              << hexfloat << 100 * sqrt(2.0) << " " 
              << defaultfloat << 100 * sqrt(2.0) << " "
              << endl;
    cout << 10.0 << " "
              << showpoint << 10.0 << " "
              << noshowpoint << 10.0
              << endl;
    int i = -16;
    double d = 3.14159;
    cout << setw(12) << i << '\n' 
              << setw(12) << d << endl;
    cout << left << setw(12) << i << '\n' 
                           << setw(12) << d << right << endl;
    cout << internal << setw(12) << i << '\n' 
                               << setw(12) << d << endl;
    cout << setfill('#') << setw(12) << i << '\n' 
                                   << setw(12) << d << setfill(' ') << endl;
    char ch;
    cin >> noskipws;
    while (cin >> ch) {
        cout << ch;
    }
    cin >> skipws;
    return 0;
}
```

### Q35

```c++
#include <iostream>
#include <cmath>

int main() {
    cout << uppercase << hexfloat
              << sqrt(2)
              << nouppercase << defaultfloat
              << endl;
    return 0;
}
```

### Q36

```c++
#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    cout << left << setw(20) << 100 * sqrt(2.0) << '\n'
              << left << setw(20) << scientific << 100 * sqrt(2.0) << '\n'
              << left << setw(20) << fixed << 100 * sqrt(2.0) << '\n'
              << left << setw(20) << hexfloat << 100 * sqrt(2.0) << '\n'
              << left << setw(20) << defaultfloat << 100 * sqrt(2.0) << '\n'
              << endl;
    return 0;
}
```

### Q37

```c++
#include <iostream>
#include <fstream>
#include <string>

int main() {
    fstream is("./data/17-37");
    char temp[20];
    is.getline(temp, 4, '\n');
    cout << temp << endl;
    cout << is.gcount() << endl;
    is.getline(temp, 4, '\n');
    cout << temp << endl;
    cout << is.gcount() << endl;
	cout << boolalpha << (is.rdstate() == ios_base::failbit) << endl;
    is.getline(temp, 4, '\n');
    cout << temp << endl;
    cout << is.gcount() << endl;
    is.getline(temp, 4, '\n');
    cout << temp << endl;
    cout << is.gcount() << endl;
    return 0;
}
```

### Q38

```c++
见17-37.cpp
```

### Q39

```c++
#include <iostream>
#include <fstream>

int main() {
    fstream inOut("./data/17-39", fstream::ate | fstream::in | fstream::out);
    if (!inOut) {
        cerr << "Unable to open file!" << endl;
        return EXIT_FAILURE;
    }
    auto end_mark = inOut.tellg();
    inOut.seekg(0, fstream::beg);
    size_t cnt = 0;
    string line;
    while (inOut && inOut.tellg() != end_mark && getline(inOut, line)) {
        cnt += line.size() + 1;
        auto mark = inOut.tellg();
        inOut.seekp(0, fstream::end);
        inOut << cnt;
        if (mark != end_mark) inOut << " ";
        inOut.seekg(mark);
    }
    inOut.seekp(0, fstream::end);
    inOut << "\n";
    return 0;
}
```