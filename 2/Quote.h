#ifndef QUOTE_H
#define QUOTE_H

#include <iostream>
#include <string>

using namespace std;

class Quote {
    friend double print_total(ostream&, const Quote&, size_t);
public:
    virtual Quote* clone() const & { return new Quote(*this); }
    virtual Quote* clone() && { return new Quote(std::move(*this)); }
    Quote() = default;
    Quote(const string &book, double sales_price) : bookNo(book), price(sales_price) {}
    string isbn() const { return bookNo; }
    virtual double net_price(size_t n) const { return n * price; }
    virtual ~Quote() = default;
    virtual void debug() const;
private:
    string bookNo;
protected:
    double price = 0.0;
};

double print_total(ostream &os, const Quote &item, size_t n) {
    double ret = item.net_price(n);
    os << "ISBN: " << item.isbn() << " # sold: " << n << " total due: " << ret << endl;
    return ret;
}

void Quote::debug() const {
    cout << bookNo << " " << price;
}

#endif