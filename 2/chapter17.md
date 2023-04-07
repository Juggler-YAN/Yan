# Chapter 17

### Q1

```c++
#include <iostream>
#include <tuple>

int main() {
    std::tuple<int, int, int> threeD{10, 20, 30};
    std::cout << std::get<0>(threeD) << std::endl;
    std::cout << std::get<1>(threeD) << std::endl;
    std::cout << std::get<2>(threeD) << std::endl;
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

int main() {
    std::tuple<std::string, std::vector<std::string>, std::pair<std::string, int>> three{"a", {"b1", "b2", "b3"}, {"c1", 3}};
    std::cout << std::get<0>(three) << std::endl;
    for (const auto &i : std::get<1>(three)) {
        std::cout << i << std::endl;
    }
    std::cout << std::get<2>(three).first << " " << std::get<2>(three).second << std::endl;
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
    using line_no = std::vector<std::string>::size_type;
    TextQuery(std::ifstream&);
    std::tuple<std::string, std::shared_ptr<std::set<TextQuery::line_no>>, std::shared_ptr<std::vector<std::string>>> query(const std::string&) const;
private:
    std::shared_ptr<std::vector<std::string>> file;
    std::map<std::string, std::shared_ptr<std::set<line_no>>> wm;
};

class QueryResult {
friend std::ostream& print(std::ostream&, const QueryResult&);
public:
    QueryResult(std::string s,
                std::shared_ptr<std::set<TextQuery::line_no>> p,
                std::shared_ptr<std::vector<std::string>> f) :
        sought(s), lines(p), file(f) {}
    auto begin() const { return lines->cbegin(); }
    auto end() const { return lines->cend(); }
    auto get_file() const { return file; }
private:
    std::string sought;
    std::shared_ptr<std::set<TextQuery::line_no>> lines;
    std::shared_ptr<std::vector<std::string>> file;
};

TextQuery::TextQuery(std::ifstream &is) : file(new std::vector<std::string>) {
    std::string text;
    while (getline(is, text)) {
        file->push_back(text);
        int n = file->size() - 1;
        std::istringstream line(text);
        std::string word;
        while (line >> word) {
            auto &lines = wm[word];
            if (!lines)
                lines.reset(new std::set<line_no>);
            lines->insert(n);
        }
    }
}

std::tuple<std::string, std::shared_ptr<std::set<TextQuery::line_no>>, std::shared_ptr<std::vector<std::string>>> TextQuery::query(const std::string &sought) const {
    static std::shared_ptr<std::set<line_no>> nodata(new std::set<line_no>);
    auto loc = wm.find(sought);
    if (loc == wm.end())
        return std::tuple<std::string, std::shared_ptr<std::set<TextQuery::line_no>>, std::shared_ptr<std::vector<std::string>>>(sought, nodata, file);
    else
        return std::tuple<std::string, std::shared_ptr<std::set<TextQuery::line_no>>, std::shared_ptr<std::vector<std::string>>>(sought, loc->second, file);
}

std::string make_plural(size_t ctr, const std::string &word, const std::string &ending) {
    return (ctr > 1) ? word + ending : word;
}

std::ostream &print(std::ostream & os, std::tuple<std::string, std::shared_ptr<std::set<TextQuery::line_no>>, std::shared_ptr<std::vector<std::string>>> qr) {
    os << std::get<0>(qr) << " occurs " << std::get<1>(qr)->size() << " "
        << make_plural(std::get<1>(qr)->size(), "times", "s") << std::endl;
    for (auto num : *std::get<1>(qr))
        os << "\t(line " << num+1 << ") " << *(std::get<2>(qr)->begin()+num) << std::endl;
    return os;
}

#endif
```

```c++
#include <iostream>
#include <fstream>
#include "TextQuery.h"

void runQueries(std::ifstream &);

int main() {
    std::ifstream in("./data/17-3");
    runQueries(in);
    return 0;
}

void runQueries(std::ifstream &infile) {
    TextQuery tq(infile);
    while (true) {
        std::cout << "enter word to look for, or q to quit: ";
        std::string s;
        if (!(std::cin >> s) || s == "q") break;
        print(std::cout, tq.query(s)) << std::endl;
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

    friend std::istream& operator>>(std::istream&, Sales_data&);
    friend std::ostream& operator<<(std::ostream&, const Sales_data&);
    friend Sales_data operator+(const Sales_data&, const Sales_data&);
    friend bool operator==(const Sales_data&, const Sales_data&);
	friend class std::hash<Sales_data>;

public:
    Sales_data(std::string s, unsigned n, double p) :
                bookNo(s), units_sold(n), revenue(p*n) {};
    Sales_data() : Sales_data("", 0, 0) {}
    Sales_data(std::string s) : Sales_data(s, 0, 0) {}
    Sales_data(std::istream &is) : Sales_data() { is >> *this; }
    std::string isbn() const { return bookNo; }
    Sales_data& operator+=(const Sales_data&);

private:
    inline double avg_price() const;
    std::string bookNo;
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

std::istream& operator>>(std::istream &is, Sales_data &item) {
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

std::ostream& operator<<(std::ostream &os, const Sales_data &item) {
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

typedef std::tuple<std::vector<Sales_data>::size_type,
                   std::vector<Sales_data>::const_iterator,
                   std::vector<Sales_data>::const_iterator> matches;

std::vector<matches> findBook(const std::vector<std::vector<Sales_data>> &files,
                              const std::string &book) {
    std::vector<matches> ret;
    for (auto it = files.cbegin(); it != files.cend(); ++it) {
        auto found = equal_range(it->cbegin(), it->cend(), book, compareIsbn);
        if (found.first != found.second) {
            ret.push_back(std::make_tuple(it-files.cbegin(), found.first, found.second));
        }
    }
    return ret;
}

void reportResults(std::istream &in, std::ostream &os, const std::vector<std::vector<Sales_data>> &files) {
    std::string s;
    while (in >> s) {
        auto trans = findBook(files, s);
        if (trans.empty()) {
            std::cout << s << " not found in any stores" << std::endl;
            continue;
        }
        for (const auto &store : trans)
            os << "store " << std::get<0>(store) << " sales " << std::accumulate(std::get<1>(store), 
                std::get<2>(store), Sales_data(s)) << std::endl;
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
    std::vector<Sales_data> v1 = {s1, s3};
    std::vector<Sales_data> v2 = {s2};
    std::vector<std::vector<Sales_data>> v = {v1, v2};
    reportResults(std::cin, std::cout, v);
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

    friend std::istream& operator>>(std::istream&, Sales_data&);
    friend std::ostream& operator<<(std::ostream&, const Sales_data&);
    friend Sales_data operator+(const Sales_data&, const Sales_data&);
    friend bool operator==(const Sales_data&, const Sales_data&);
	friend class std::hash<Sales_data>;

public:
    Sales_data(std::string s, unsigned n, double p) :
                bookNo(s), units_sold(n), revenue(p*n) {};
    Sales_data() : Sales_data("", 0, 0) {}
    Sales_data(std::string s) : Sales_data(s, 0, 0) {}
    Sales_data(std::istream &is) : Sales_data() { is >> *this; }
    std::string isbn() const { return bookNo; }
    Sales_data& operator+=(const Sales_data&);

private:
    inline double avg_price() const;
    std::string bookNo;
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

std::istream& operator>>(std::istream &is, Sales_data &item) {
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

std::ostream& operator<<(std::ostream &os, const Sales_data &item) {
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

typedef std::pair<std::vector<Sales_data>::size_type,
    std::pair<std::vector<Sales_data>::const_iterator, std::vector<Sales_data>::const_iterator>> matches;

std::vector<matches> findBook(const std::vector<std::vector<Sales_data>> &files,
                              const std::string &book) {
    std::vector<matches> ret;
    for (auto it = files.cbegin(); it != files.cend(); ++it) {
        auto found = equal_range(it->cbegin(), it->cend(), book, compareIsbn);
        if (found.first != found.second) {
            ret.push_back(std::make_pair(it-files.cbegin(), make_pair(found.first, found.second)));
        }
    }
    return ret;
}

void reportResults(std::istream &in, std::ostream &os, const std::vector<std::vector<Sales_data>> &files) {
    std::string s;
    while (in >> s) {
        auto trans = findBook(files, s);
        if (trans.empty()) {
            std::cout << s << " not found in any stores" << std::endl;
            continue;
        }
        for (const auto &store : trans)
            os << "store " << store.first << " sales " << std::accumulate(store.second.first, 
                store.second.second, Sales_data(s)) << std::endl;
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
    std::vector<Sales_data> v1 = {s1, s3};
    std::vector<Sales_data> v2 = {s2};
    std::vector<std::vector<Sales_data>> v = {v1, v2};
    reportResults(std::cin, std::cout, v);
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

    friend std::istream& operator>>(std::istream&, Sales_data&);
    friend std::ostream& operator<<(std::ostream&, const Sales_data&);
    friend Sales_data operator+(const Sales_data&, const Sales_data&);
    friend bool operator==(const Sales_data&, const Sales_data&);
	friend class std::hash<Sales_data>;

public:
    Sales_data(std::string s, unsigned n, double p) :
                bookNo(s), units_sold(n), revenue(p*n) {};
    Sales_data() : Sales_data("", 0, 0) {}
    Sales_data(std::string s) : Sales_data(s, 0, 0) {}
    Sales_data(std::istream &is) : Sales_data() { is >> *this; }
    std::string isbn() const { return bookNo; }
    Sales_data& operator+=(const Sales_data&);

private:
    inline double avg_price() const;
    std::string bookNo;
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

std::istream& operator>>(std::istream &is, Sales_data &item) {
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

std::ostream& operator<<(std::ostream &os, const Sales_data &item) {
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
    std::vector<Sales_data>::size_type index;
    std::vector<Sales_data>::const_iterator first;
    std::vector<Sales_data>::const_iterator last;
    matches(std::vector<Sales_data>::size_type i, std::vector<Sales_data>::const_iterator f, std::vector<Sales_data>::const_iterator l) : 
        index(i), first(f), last(l) {}
};

std::vector<matches> findBook(const std::vector<std::vector<Sales_data>> &files,
                              const std::string &book) {
    std::vector<matches> ret;
    for (auto it = files.cbegin(); it != files.cend(); ++it) {
        auto found = equal_range(it->cbegin(), it->cend(), book, compareIsbn);
        if (found.first != found.second) {
            ret.push_back(matches(it-files.cbegin(), found.first, found.second));
        }
    }
    return ret;
}

void reportResults(std::istream &in, std::ostream &os, const std::vector<std::vector<Sales_data>> &files) {
    std::string s;
    while (in >> s) {
        auto trans = findBook(files, s);
        if (trans.empty()) {
            std::cout << s << " not found in any stores" << std::endl;
            continue;
        }
        for (const auto &store : trans)
            os << "store " << store.index << " sales " << std::accumulate(store.first, 
                store.last, Sales_data(s)) << std::endl;
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
    std::vector<Sales_data> v1 = {s1, s3};
    std::vector<Sales_data> v2 = {s2};
    std::vector<std::vector<Sales_data>> v = {v1, v2};
    reportResults(std::cin, std::cout, v);
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
    std::vector<int> v{1,2,3,5,8,13,21};
    std::bitset<32> b1;
    for (auto i : v) {
        b1.set(i);
    }
    std::cout << b1 << std::endl;
    std::bitset<32> b2(2105646ULL);
    std::cout << b2 << std::endl;
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
    std::ostream& operator<<(std::ostream&, const quiz<N>&);

template <unsigned N>
class quiz {
    friend std::ostream& operator<<<N>(std::ostream&, const quiz<N>&);
public:
    quiz(const std::string &s) : b(s) {}
    std::bitset<N>& get_bitset() { return b; }
private:
    std::bitset<N> b;
};

template <unsigned N>
std::ostream& operator<<(std::ostream& os, const quiz<N> &q) {
    os << q.b;
    return os;
}

int main() {
    quiz<10> q1(std::string("01010101010101"));
    quiz<100> q2(std::string("01010101010101"));
    std::cout << q1 << std::endl;
    std::cout << q2 << std::endl;
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
    std::ostream& operator<<(std::ostream&, const quiz<N>&);

template <unsigned N>
class quiz {
    friend std::ostream& operator<<<N>(std::ostream&, const quiz<N>&);
public:
    quiz(const std::string &s) : b(s) {}
    void update(std::size_t n, bool res) {
        if (n < N) {
            b[n] = res;
        }
    }
private:
    std::bitset<N> b;
};

template <unsigned N>
std::ostream& operator<<(std::ostream& os, const quiz<N> &q) {
    os << q.b;
    return os;
}

int main() {
    quiz<10> q(std::string("01010101010101"));
    std::cout << q << std::endl;
    q.update(1,true);
    std::cout << q << std::endl;
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
    std::ostream& operator<<(std::ostream&, const quiz<N>&);
template <unsigned N>
    std::size_t grade(const quiz<N>&, const quiz<N>&);

template <unsigned N>
class quiz {
    friend std::ostream& operator<<<N>(std::ostream&, const quiz<N>&);
    friend std::size_t grade<N>(const quiz<N>&, const quiz<N>&);
public:
    quiz(const std::string &s) : b(s) {}
    void update(std::size_t n, bool res) {
        if (n < N) {
            b[n] = res;
        }
    }
private:
    std::bitset<N> b;
};

template <unsigned N>
std::ostream& operator<<(std::ostream& os, const quiz<N> &q) {
    os << q.b;
    return os;
}

template <unsigned N>
std::size_t grade(const quiz<N> &lhs, const quiz<N> &rhs) {
    return (lhs.b^rhs.b).flip().count();    
}

int main() {
    quiz<10> q1(std::string("0101010101"));
    quiz<10> q2(std::string("1101011100"));
    std::cout << grade(q1,q2) << std::endl;
    return 0;
}
```

### Q14

```c++
#include <iostream>
#include <regex>

int main() {
    try {
        std::regex r("[[:alnum:]+\\.(cpp|cxx|cc)$", std::regex::icase);
    }
    catch (std::regex_error e) {
        std::cout << e.what() << "\ncode: " << e.code() << std::endl;
    }
    try {
        std::regex r("[[:alnum:]]+\\.(cpp|cxx|cc$", std::regex::icase);
    }
    catch (std::regex_error e) {
        std::cout << e.what() << "\ncode: " << e.code() << std::endl;
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
    std::string pattern("[^c]ei");
    pattern = "[[:alpha:]]*" + pattern + "[[:alpha:]]*";
    std::regex r(pattern);
    std::smatch results;
    std::string s;
    while (std::cin >> s) {
        if (std::regex_search(s, results, r)) {
            std::cout << s << " : correct." << std::endl;
            std::cout << results.str() << std::endl;
        }
        else {
            std::cout << s << " : error." << std::endl;
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
    std::string pattern("[^c]ei");
    std::regex r(pattern);
    std::smatch results;
    std::string s;
    while (std::cin >> s) {
        if (std::regex_search(s, results, r)) {
            std::cout << s << " : correct." << std::endl;
            std::cout << results.str() << std::endl;
        }
        else {
            std::cout << s << " : error." << std::endl;
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
    std::string pattern("[^c]ei");
    pattern = "[[:alpha:]]*" + pattern + "[[:alpha:]]*";
    std::regex r(pattern, std::regex::icase);
	std::string file("freind receipt theif receive");
    for (std::sregex_iterator it(file.begin(), file.end(), r), end_it; it != end_it; ++it)
        std::cout << it->str() << std::endl;
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
    std::string pattern("[^c]ei");
    pattern = "[[:alpha:]]*" + pattern + "[[:alpha:]]*";
    std::regex r(pattern, std::regex::icase);
	std::string file("albeit neighbor freind receipt theif receive");
    std::vector<std::string> v{"albeit", "neighbor"};
    for (std::sregex_iterator it(file.begin(), file.end(), r), end_it; it != end_it; ++it) {
        if (find(v.begin(), v.end(),it->str()) != v.end()) {
            continue;
        }
        std::cout << it->str() << std::endl;
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

bool valid(const std::smatch&);

int main() {
    std::string phone = "(\\()?(\\d{3})(\\))?([-. ])?(\\d{3})([-. ])?(\\d{4})";
    std::regex r(phone);
    std::smatch m;
    std::string s;
    while (getline(std::cin, s)) {
        for (std::sregex_iterator it(s.begin(), s.end(), r), end_it; it != end_it; ++it) {
            if (valid(*it)) {
                std::cout << "valid: " << it->str() << std::endl;
            }
            else {
                std::cout << "not valid: " << it->str() << std::endl;
            }
        }
    }
    return 0;
}

bool valid(const std::smatch &m) {
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

bool valid(const std::smatch&);

struct PersonInfo {
    std::string name;
    std::vector<std::string> phones;
};

int main() {
	std::string line, word;
	std::vector<PersonInfo> people;
	std::istringstream record;
    std::ifstream in("./data/17-21");

	while(getline(in, line)) {
		record.str(line);
		PersonInfo info;
		record >> info.name;
		while(record >> word)
			info.phones.push_back(word);
		record.clear();
		people.push_back(info);
	}

    
    std::string phone = "(\\()?(\\d{3})(\\))?([-. ])?(\\d{3})([-. ])?(\\d{4})";
    std::regex r(phone);
    std::smatch m;

	for(const auto &person : people) {
		std::cout << person.name << "  ";
		for(const auto &ph : person.phones) {
            for (std::sregex_iterator it(ph.begin(), ph.end(), r), end_it; it != end_it; ++it) {
                if (valid(*it)) {
                    std::cout << it->str() << " ";
                }
            }
		}
		std::cout << std::endl;
	}

	in.close();

	return 0;
}

bool valid(const std::smatch &m) {
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

bool valid(const std::smatch&);

int main() {
    std::string phone = "(\\()?(\\d{3})(\\))?([-. ])?([ ]*)?(\\d{3})([-. ])?([ ]*)?(\\d{4})";
    std::regex r(phone);
    std::smatch m;
    std::string s;
    while (getline(std::cin, s)) {
        for (std::sregex_iterator it(s.begin(), s.end(), r), end_it; it != end_it; ++it) {
            if (valid(*it)) {
                std::cout << "valid: " << it->str() << std::endl;
            }
            else {
                std::cout << "not valid: " << it->str() << std::endl;
            }
        }
    }
    return 0;
}

bool valid(const std::smatch &m) {
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

bool valid(const std::smatch&);

int main() {
    std::string mail = "(\\d{5})(-)?(\\d{4})?";
    std::regex r(mail);
    std::smatch m;
    std::string s;
    while (getline(std::cin, s)) {
        for (std::sregex_iterator it(s.begin(), s.end(), r), end_it; it != end_it; ++it) {
            if (valid(*it)) {
                std::cout << "valid: " << it->str() << std::endl;
            }
            else {
                std::cout << "not valid: " << it->str() << std::endl;
            }
        }
    }
    return 0;
}

bool valid(const std::smatch &m) {
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
    std::string phone = "(\\()?(\\d{3})(\\))?([-. ])?(\\d{3})([-. ])?(\\d{4})";
    std::regex r(phone);
    std::smatch m;
    std::string s;
    std::string fmt = "$2.$5.$7";
    while (getline(std::cin, s)) {
        std::cout << std::regex_replace(s, r, fmt) << std::endl;
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
    std::string phone = "(\\()?(\\d{3})(\\))?([-. ])?(\\d{3})([-. ])?(\\d{4})";
    std::regex r(phone);
    std::string fmt("$2.$5.$7");
    std::ifstream in("./data/17-25");
    std::string line;
	while(getline(in, line)) {
    	std::smatch m;
		std::regex_search(line, m, r);
		if (!m.empty()) {
			std::cout << m.prefix().str() << m.format(fmt) << std::endl;
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

bool valid(const std::smatch&);

int main() {
    std::string phone = "(\\()?(\\d{3})(\\))?([-. ])?(\\d{3})([-. ])?(\\d{4})";
    std::regex r(phone);
    std::smatch m;
    std::ifstream in("./data/17-26");
    std::string line;
	while(getline(in, line)) {
        std::vector<std::string> v;
        for (std::sregex_iterator it(line.begin(), line.end(), r), end_it; it != end_it; ++it) {
            if (valid(*it)) {
                v.push_back(it->str());
            }
        }
        if (v.size() == 0) {
            continue;
        }
        else if (v.size() == 1) {
            std::cout << v[0] << std::endl;
        }
        else {
            for (decltype(v.size()) i = 1; i != v.size(); ++i) {
                std::cout << v[i] << " ";
            }
            std::cout << std::endl;
        }
	}
	in.close();
	return 0;
}

bool valid(const std::smatch &m) {
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

bool valid(const std::smatch&);

int main() {
    std::string mail = "(\\d{5})(-)?(\\d{4})?";
    std::regex r(mail);
	std::string fmt("$1-$3");
    std::smatch m;
    std::string s;
    while (getline(std::cin, s)) {
        for (std::sregex_iterator it(s.begin(), s.end(), r), end_it; it != end_it; ++it) {
            if (valid(*it)) {
                std::cout << (*it).format(fmt) << std::endl;
            }
        }
    }
    return 0;
}

bool valid(const std::smatch &m) {
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
    std::cout << myrand() << std::endl;
    std::cout << myrand() << std::endl;
    return 0;
}

unsigned int myrand() {
    static std::default_random_engine e;
    static std::uniform_int_distribution<unsigned int> u;
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
    std::cout << myrand() << std::endl;
    std::cout << myrand() << std::endl;
    std::cout << myrand(2) << std::endl;
    std::cout << myrand(2) << std::endl;
    return 0;
}

unsigned int myrand() {
    static std::default_random_engine e;
    static std::uniform_int_distribution<unsigned int> u;
    return u(e);
}

unsigned int myrand(unsigned int i) {
    static std::default_random_engine e(i);
    static std::uniform_int_distribution<unsigned int> u;
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
    std::cout << myrand() << std::endl;
    std::cout << myrand() << std::endl;
    std::cout << myrand(2) << std::endl;
    std::cout << myrand(2) << std::endl;
    std::cout << myrand(2,0,9) << std::endl;
    std::cout << myrand(2,0,9) << std::endl;
    return 0;
}

unsigned int myrand() {
    static std::default_random_engine e;
    static std::uniform_int_distribution<unsigned int> u;
    return u(e);
}

unsigned int myrand(unsigned int i) {
    static std::default_random_engine e(i);
    static std::uniform_int_distribution<unsigned int> u;
    return u(e);
}

unsigned int myrand(unsigned int i, unsigned int minval, unsigned int maxval) {
    static std::default_random_engine e(i);
    static std::uniform_int_distribution<unsigned int> u(minval, maxval);
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
    std::cout << true << " " << std::boolalpha << true << " " << std::noboolalpha << std::endl;
    std::cout << 20 << " " 
              << std::oct << 20 << " "
              << std::hex << 20 << " "
              << std::dec << 20 << " "
              << std::showbase
              << 20 << " "
              << std::oct << 20 << " "
              << std::uppercase
              << std::hex << 20 << " "
              << std::nouppercase
              << std::hex << 20 << " "
              << std::dec << 20 << " "
              << std::noshowbase
              << std::endl;
    std::cout << sqrt(2.0) << " "
              << std::cout.precision(12) << " " << sqrt(2.0) << " "
              << std::setprecision(3) << sqrt(2.0) 
              << std::setprecision(6)
              << std::endl;
    std::cout << 100 * sqrt(2.0) << " "
              << std::scientific << 100 * sqrt(2.0) << " "
              << std::fixed << 100 * sqrt(2.0) << " "
              << std::hexfloat << 100 * sqrt(2.0) << " " 
              << std::defaultfloat << 100 * sqrt(2.0) << " "
              << std::endl;
    std::cout << 10.0 << " "
              << std::showpoint << 10.0 << " "
              << std::noshowpoint << 10.0
              << std::endl;
    int i = -16;
    double d = 3.14159;
    std::cout << std::setw(12) << i << '\n' 
              << std::setw(12) << d << std::endl;
    std::cout << std::left << std::setw(12) << i << '\n' 
                           << std::setw(12) << d << std::right << std::endl;
    std::cout << std::internal << std::setw(12) << i << '\n' 
                               << std::setw(12) << d << std::endl;
    std::cout << std::setfill('#') << std::setw(12) << i << '\n' 
                                   << std::setw(12) << d << std::setfill(' ') << std::endl;
    char ch;
    std::cin >> std::noskipws;
    while (std::cin >> ch) {
        std::cout << ch;
    }
    std::cin >> std::skipws;
    return 0;
}
```

### Q35

```c++
#include <iostream>
#include <cmath>

int main() {
    std::cout << std::uppercase << std::hexfloat
              << sqrt(2)
              << std::nouppercase << std::defaultfloat
              << std::endl;
    return 0;
}
```

### Q36

```c++
#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    std::cout << std::left << std::setw(20) << 100 * sqrt(2.0) << '\n'
              << std::left << std::setw(20) << std::scientific << 100 * sqrt(2.0) << '\n'
              << std::left << std::setw(20) << std::fixed << 100 * sqrt(2.0) << '\n'
              << std::left << std::setw(20) << std::hexfloat << 100 * sqrt(2.0) << '\n'
              << std::left << std::setw(20) << std::defaultfloat << 100 * sqrt(2.0) << '\n'
              << std::endl;
    return 0;
}
```

### Q37

```c++
#include <iostream>
#include <fstream>
#include <string>

int main() {
    std::fstream is("./data/17-37");
    char temp[20];
    is.getline(temp, 4, '\n');
    std::cout << temp << std::endl;
    std::cout << is.gcount() << std::endl;
    is.getline(temp, 4, '\n');
    std::cout << temp << std::endl;
    std::cout << is.gcount() << std::endl;
	std::cout << std::boolalpha << (is.rdstate() == std::ios_base::failbit) << std::endl;
    is.getline(temp, 4, '\n');
    std::cout << temp << std::endl;
    std::cout << is.gcount() << std::endl;
    is.getline(temp, 4, '\n');
    std::cout << temp << std::endl;
    std::cout << is.gcount() << std::endl;
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
    std::fstream inOut("./data/17-39", std::fstream::ate | std::fstream::in | std::fstream::out);
    if (!inOut) {
        std::cerr << "Unable to open file!" << std::endl;
        return EXIT_FAILURE;
    }
    auto end_mark = inOut.tellg();
    inOut.seekg(0, std::fstream::beg);
    size_t cnt = 0;
    std::string line;
    while (inOut && inOut.tellg() != end_mark && getline(inOut, line)) {
        cnt += line.size() + 1;
        auto mark = inOut.tellg();
        inOut.seekp(0, std::fstream::end);
        inOut << cnt;
        if (mark != end_mark) inOut << " ";
        inOut.seekg(mark);
    }
    inOut.seekp(0, std::fstream::end);
    inOut << "\n";
    return 0;
}
```