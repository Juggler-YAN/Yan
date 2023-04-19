#ifndef FEW_QUOTE_H
#define FEW_QUOTE_H

#include <string>
#include "Disc_quote.h"

using namespace std;

class Few_quote : public Disc_quote {
public:
    Few_quote() = default;
    Few_quote(const string& book, double p, size_t qty, double disc) : 
               Disc_quote(book, p, qty, disc) {}
    double net_price(size_t) const override;
};

double Few_quote::net_price(size_t cnt) const {
    if (cnt <= quantity) {
        return cnt * (1 - discount) * price;
    }
    else {
        return quantity * (1 - discount) * price + (cnt - quantity) * price;
    }
}

#endif