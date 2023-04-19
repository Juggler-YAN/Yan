#ifndef BULK_QUOTE_H
#define BULK_QUOTE_H

#include <iostream>
#include "Quote.h"

using namespace std;

class Bulk_quote : public Quote {
public:
    Bulk_quote* clone() const & { return new Bulk_quote(*this); }
    Bulk_quote* clone() && { return new Bulk_quote(std::move(*this)); }
    Bulk_quote() = default;
    Bulk_quote(const string& book, double p, size_t qty, double disc) : 
               Quote(book, p), min_qty(qty), discount(disc) {}
    double net_price(size_t) const override;
    void debug() const override;
private:
    size_t min_qty = 0;
    double discount = 0.0;
};

double Bulk_quote::net_price(size_t cnt) const {
    if (cnt >= min_qty) {
        return cnt * (1 - discount) * price;
    }
    else {
        return cnt * price;
    }
}

void Bulk_quote::debug() const {
    Quote::debug();
    cout << " " << min_qty << " " << discount;
}

#endif