#ifndef BOOK_H
#define BOOK_H

#include <string>

class Book {

public:
    Book(unsigned int a, std::string b, std::string c) : 
        no(a), name(b), author(c) {}
    Book() : Book(0,"","") {};
    Book(std::istream &is) : Book() { is >> no >> name >> author; }

private:
    unsigned int no;
    std::string name;
    std::string author;

};

#endif