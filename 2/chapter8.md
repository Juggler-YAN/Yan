# Chapter 8

### Q1

```c++
#include <iostream>
#include <string>

using namespace std;

istream& myread(istream&);

int main() {
    myread(cin);
    return 0;
}

istream& myread(istream& in) {
    string s;
    while (in >> s) {
        cout << s << " ";
    }
    cout << endl;
    in.clear();
    return in;
}
```

### Q2

见Q1

### Q3

badbit、failbit和eofbit中任何一个被置位，则检测流状态的条件会失败，循环终止。

### Q4

```c++
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

void myread(ifstream&, vector<string>&);

int main() {
    vector<string> v;
    ifstream in("./test.txt");
    if (in) {
        myread(in, v);
    }
    for (auto i : v) {
        cout << i << endl;
    }
    in.close();
    return 0;
}

void myread(ifstream& in, vector<string>& v) {
    string s;
    while (getline(in, s)) {
        v.push_back(s);
    }
}
```

### Q5

```c++
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

void myread(ifstream&, vector<string>&);

int main() {
    vector<string> v;
    ifstream in("./test.txt");
    if (in) {
        myread(in, v);
    }
    for (auto i : v) {
        cout << i << " ";
    }
    cout << endl;
    in.close();
    return 0;
}

void myread(ifstream& in, vector<string>& v) {
    string s;
    while (in >> s) {
        v.push_back(s);
    }
}
```

### Q6

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
// Sales_data.h
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <string>

using namespace std;

struct Sales_data {

    string bookNo;
    unsigned units_sold;
    double revenue;

    string isbn() const { return bookNo; }
    Sales_data& combine(const Sales_data&);
    double avg_price() const;

};

Sales_data& Sales_data::combine(const Sales_data& rhs) {
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}

double Sales_data::avg_price() const {
	if (units_sold) {
		return revenue / units_sold;
    }
	else {
		return 0;
    }
}

istream &read(istream &is, Sales_data &item) {
	double price = 0;
	is >> item.bookNo >> item.units_sold >> price;
	item.revenue = price * item.units_sold;
	return is;
}

ostream &print(ostream &os, const Sales_data &item) {
	os << item.isbn() << " " << item.units_sold << " " << item.revenue << " " << item.avg_price();
	return os;
}

Sales_data add(const Sales_data &lhs, const Sales_data &rhs) {
	Sales_data sum = lhs;
	sum.combine(rhs);
	return sum;
}

#endif
```

```c++
#include <iostream>
#include <fstream>
#include "Sales_data.h"

using namespace std;

int main(int argc, char* argv[]) {
    ifstream input(argv[1]);
    Sales_data total;
    if (read(input, total)) {
        Sales_data trans;
        while (read(input, trans)) {
            if (total.isbn() == trans.isbn())
                total.combine(trans);
            else {
                print(cout, total) << endl;
                total = trans;
            }
        }
        print(cout, total) << endl;
    } else
        cerr << "No data?!" << endl;
    input.close();
    return 0;
}
```

### Q7

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
// Sales_data.h
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <string>

using namespace std;

struct Sales_data {

    string bookNo;
    unsigned units_sold;
    double revenue;

    string isbn() const { return bookNo; }
    Sales_data& combine(const Sales_data&);
    double avg_price() const;

};

Sales_data& Sales_data::combine(const Sales_data& rhs) {
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}

double Sales_data::avg_price() const {
	if (units_sold) {
		return revenue / units_sold;
    }
	else {
		return 0;
    }
}

istream &read(istream &is, Sales_data &item) {
	double price = 0;
	is >> item.bookNo >> item.units_sold >> price;
	item.revenue = price * item.units_sold;
	return is;
}

ostream &print(ostream &os, const Sales_data &item) {
	os << item.isbn() << " " << item.units_sold << " " << item.revenue << " " << item.avg_price();
	return os;
}

Sales_data add(const Sales_data &lhs, const Sales_data &rhs) {
	Sales_data sum = lhs;
	sum.combine(rhs);
	return sum;
}

#endif
```

```c++
#include <iostream>
#include <fstream>
#include "Sales_data.h"

using namespace std;

int main(int argc, char* argv[]) {
    ifstream input(argv[1]);
    ofstream output(argv[2]);
    Sales_data total;
    if (read(input, total)) {
        Sales_data trans;
        while (read(input, trans)) {
            if (total.isbn() == trans.isbn())
                total.combine(trans);
            else {
                print(output, total) << endl;
                total = trans;
            }
        }
        print(output, total) << endl;
    } else
        cerr << "No data?!" << endl;
    input.close();
    output.close();
    return 0;
}
```

### Q8

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
// Sales_data.h
#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <string>

using namespace std;

struct Sales_data {

    string bookNo;
    unsigned units_sold;
    double revenue;

    string isbn() const { return bookNo; }
    Sales_data& combine(const Sales_data&);
    double avg_price() const;

};

Sales_data& Sales_data::combine(const Sales_data& rhs) {
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}

double Sales_data::avg_price() const {
	if (units_sold) {
		return revenue / units_sold;
    }
	else {
		return 0;
    }
}

istream &read(istream &is, Sales_data &item) {
	double price = 0;
	is >> item.bookNo >> item.units_sold >> price;
	item.revenue = price * item.units_sold;
	return is;
}

ostream &print(ostream &os, const Sales_data &item) {
	os << item.isbn() << " " << item.units_sold << " " << item.revenue << " " << item.avg_price();
	return os;
}

Sales_data add(const Sales_data &lhs, const Sales_data &rhs) {
	Sales_data sum = lhs;
	sum.combine(rhs);
	return sum;
}

#endif
```

```c++
#include <iostream>
#include <fstream>
#include "Sales_data.h"

using namespace std;

int main(int argc, char* argv[]) {
    ifstream input(argv[1]);
    ofstream output(argv[2], ofstream::app);
    Sales_data total;
    if (read(input, total)) {
        Sales_data trans;
        while (read(input, trans)) {
            if (total.isbn() == trans.isbn())
                total.combine(trans);
            else {
                print(output, total) << endl;
                total = trans;
            }
        }
        print(output, total) << endl;
    } else
        cerr << "No data?!" << endl;
    input.close();
    output.close();
    return 0;
}
```

### Q9

```c++
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

istream& myread(istream&);

int main() {
    istringstream in("Hello world");
    myread(in);
    return 0;
}

istream& myread(istream& in) {
    string s;
    while (in >> s) {
        cout << s << " ";
    }
    cout << endl;
    in.clear();
    return in;
}
```

### Q10

```c++
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

void myread(ifstream&, vector<string>&);

int main() {
    vector<string> v;
    ifstream in("./test.txt");
    if (in) {
        myread(in, v);
    }
    for (auto i : v) {
        istringstream ss(i);
        string s;
        while (ss >> s) {
            cout << s << " ";
        }
        cout << endl;
    }
    in.close();
    return 0;
}

void myread(ifstream& in, vector<string>& v) {
    string s;
    while (getline(in, s)) {
        v.push_back(s);
    }
}
```

### Q11

```c++
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

struct PersonInfo {
    string name;
    vector<string> phones;
};

int main() {
	string line, word;
	vector<PersonInfo> people;
	istringstream record;

	while(getline(cin, line)) {
		record.str(line);
		PersonInfo info;
		record >> info.name;
		while(record >> word)
			info.phones.push_back(word);
		record.clear();
		people.push_back(info);
	}

	for(const auto &person : people) {
		cout << person.name << "  ";
		for(const auto &ph : person.phones) {
			cout << ph << " ";
		}
		cout << endl;
	}

	return 0;
}
```

### Q12

string和vector可以自己初始化

### Q13

```c++
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

struct PersonInfo {
    string name;
    vector<string> phones;
};

int main() {
	string line, word;
	vector<PersonInfo> people;
	istringstream record;
    ifstream in("./test.txt");

	while(getline(in, line)) {
		record.str(line);
		PersonInfo info;
		record >> info.name;
		while(record >> word)
			info.phones.push_back(word);
		record.clear();
		people.push_back(info);
	}

	for(const auto &person : people) {
		cout << person.name << "  ";
		for(const auto &ph : person.phones) {
			cout << ph << " ";
		}
		cout << endl;
	}

	in.close();

	return 0;
}
```

### Q14

使用引用避免拷贝减少开销，不涉及改变值所以采用const