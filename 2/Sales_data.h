#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <string>

using namespace std;

struct Sales_data {

    friend istream& operator>>(istream&, Sales_data&);
    friend ostream& operator<<(ostream&, const Sales_data&);
    friend Sales_data operator+(const Sales_data&, const Sales_data&);

public:
    Sales_data(string s, unsigned n, double p) :
                bookNo(s), units_sold(n), revenue(p*n) {};
    Sales_data() : Sales_data("", 0, 0) {}
    Sales_data(string s) : Sales_data(s, 0, 0) {}
    Sales_data(istream &is) : Sales_data() { is >> *this; }
    string isbn() const { return bookNo; }
    Sales_data& operator+=(const Sales_data&);
    Sales_data& operator=(const string&);
    explicit operator string() const { return bookNo; }
    explicit operator double() const { return avg_price(); }

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

Sales_data& Sales_data::operator+=(const Sales_data &rhs) {
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}

Sales_data& Sales_data::operator=(const string &s) {
    *this = Sales_data(s);
    return *this;
}

#endif