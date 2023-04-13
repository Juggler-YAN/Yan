#ifndef PERSON_H
#define PERSON_H

#include <string>

using namespace std;

struct Person {
    friend istream& read(istream&, Person&);
    friend ostream& print(ostream&, const Person&);
private:
    string name{""};
    string address{""};
public:
    string getName() const { return name; }
    string getAddress() const { return address; }
    Person() = default;
    Person(const string & n, const string & a) : name(n), address(a) {}
    explicit Person(istream &is) { read(is, *this); }
};

istream& read(istream& is, Person& item) {
    is >> item.name >> item.address;
    return is;
}

ostream& print(ostream& os, const Person& item) {
    os << item.name << " " << item.address;
    return os;
}

#endif