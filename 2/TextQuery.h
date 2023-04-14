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
#include "StrBlob.h"

using namespace std;

class QueryResult;
class TextQuery {
public:
    using line_no = StrBlob::size_type;
    TextQuery(ifstream&);
    QueryResult query(const string&) const;
private:
    StrBlob file;
    map<string, shared_ptr<set<line_no>>> wm;
};

class QueryResult {
friend ostream& print(ostream&, const QueryResult&);
public:
    QueryResult(string s,
                shared_ptr<set<TextQuery::line_no>> p,
                StrBlob f) :
        sought(s), lines(p), file(f) {}
    set<StrBlob::size_type>::iterator begin() const { return lines->begin(); }
    set<StrBlob::size_type>::iterator end() const { return lines->end(); }
    shared_ptr<StrBlob> get_file() const { return make_shared<StrBlob>(file); }
private:
    string sought;
    shared_ptr<set<TextQuery::line_no>> lines;
    StrBlob file;
};

TextQuery::TextQuery(ifstream &is) {
    string text;
    while (getline(is, text)) {
        file.push_back(text);
        int n = file.size() - 1;
        istringstream line(text);
        string word;
        while (line >> word) {
            auto &lines = wm[word];
            if (!lines)
                lines.reset(new set<line_no>);
            lines->insert(n);
        }
    }
}

QueryResult TextQuery::query(const string &sought) const {
    static shared_ptr<set<line_no>> nodata(new set<line_no>);
    auto loc = wm.find(sought);
    if (loc == wm.end())
        return QueryResult(sought, nodata, file);
    else
        return QueryResult(sought, loc->second, file);
}

string make_plural(size_t ctr, const string &word, const string &ending) {
    return (ctr > 1) ? word + ending : word;
}

ostream &print(ostream & os, const QueryResult &qr) {
    os << qr.sought << " occurs " << qr.lines->size() << " "
        << make_plural(qr.lines->size(), "times", "s") << endl;
    for (auto num : *qr.lines) {
        ConstStrBlobPtr p(qr.file, num);
        os << "\t(line " << num+1 << ") " << p.deref() << endl;
    }
    return os;
}

#endif