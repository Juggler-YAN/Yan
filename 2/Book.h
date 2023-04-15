#ifndef BOOK_H
#define BOOK_H

#include <string>

using namespace std;

class Book {

    friend istream& operator>>(istream&, Book&);
    friend ostream& operator<<(ostream&, const Book&);
    friend bool operator==(const Book&, const Book&);
    friend bool operator!=(const Book&, const Book&);
    friend bool operator<(const Book&, const Book&);
    friend bool operator>(const Book&, const Book&);
    friend bool operator<=(const Book&, const Book&);
    friend bool operator>=(const Book&, const Book&);

public:
    Book() = default;
    Book(unsigned int a, string b, string c) : 
        no(a), name(b), author(c) {}
    Book(istream &is) { is >> *this; }
    Book& operator=(const Book &);
    Book& operator=(Book&&) noexcept;
    explicit operator bool() const { return no; }

private:
    unsigned int no;
    string name;
    string author;

};

istream& operator>>(istream &is, Book &book) {
    is >> book.no >> book.name >> book.author;
    if (!is) {
        book = Book();
    }
    return is;
}

ostream& operator<<(ostream &os, const Book &book) {
    os << book.no << " " << book.name << " " << book.author;
    return os;
}

bool operator==(const Book &book1, const Book &book2) {
    return book1.no == book2.no;
}

bool operator!=(const Book &book1, const Book &book2) {
    return !(book1 == book2);
}

bool operator<(const Book &book1, const Book &book2) {
    return book1.no < book2.no;
}

bool operator>(const Book &book1, const Book &book2) {
    return book2 < book1;
}

bool operator<=(const Book &book1, const Book &book2) {
    return !(book2 < book1);
}

bool operator>=(const Book &book1, const Book &book2) {
    return !(book1 < book2);
}

Book& Book::operator=(const Book &book) {
    no = book.no;
    name = book.name;
    author = book.author;
    return *this;
}

Book& Book::operator=(Book &&book) noexcept {
    if (this != &book) {
        no = move(book.no);
        name = move(book.name);
        author = move(book.author);
    }
    return *this;
}
#endif