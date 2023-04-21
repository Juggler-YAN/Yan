#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <algorithm>
#include <numeric>
#include <vector>
#include <string>

using namespace std;

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