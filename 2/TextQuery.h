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
#include <tuple>

using namespace std;

class QueryResult;
class TextQuery {
public:
    using line_no = vector<string>::size_type;
    TextQuery(ifstream&);
    tuple<string, shared_ptr<set<TextQuery::line_no>>, shared_ptr<vector<string>>> query(const string&) const;
private:
    shared_ptr<vector<string>> file;
    map<string, shared_ptr<set<line_no>>> wm;
};

class QueryResult {
friend ostream& print(ostream&, const QueryResult&);
public:
    QueryResult(string s,
                shared_ptr<set<TextQuery::line_no>> p,
                shared_ptr<vector<string>> f) :
        sought(s), lines(p), file(f) {}
    auto begin() const { return lines->cbegin(); }
    auto end() const { return lines->cend(); }
    auto get_file() const { return file; }
private:
    string sought;
    shared_ptr<set<TextQuery::line_no>> lines;
    shared_ptr<vector<string>> file;
};

TextQuery::TextQuery(ifstream &is) : file(new vector<string>) {
    string text;
    while (getline(is, text)) {
        file->push_back(text);
        int n = file->size() - 1;
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

tuple<string, shared_ptr<set<TextQuery::line_no>>, shared_ptr<vector<string>>> TextQuery::query(const string &sought) const {
    static shared_ptr<set<line_no>> nodata(new set<line_no>);
    auto loc = wm.find(sought);
    if (loc == wm.end())
        return tuple<string, shared_ptr<set<TextQuery::line_no>>, shared_ptr<vector<string>>>(sought, nodata, file);
    else
        return tuple<string, shared_ptr<set<TextQuery::line_no>>, shared_ptr<vector<string>>>(sought, loc->second, file);
}

string make_plural(size_t ctr, const string &word, const string &ending) {
    return (ctr > 1) ? word + ending : word;
}

ostream &print(ostream & os, tuple<string, shared_ptr<set<TextQuery::line_no>>, shared_ptr<vector<string>>> qr) {
    os << get<0>(qr) << " occurs " << get<1>(qr)->size() << " "
        << make_plural(get<1>(qr)->size(), "times", "s") << endl;
    for (auto num : *get<1>(qr))
        os << "\t(line " << num+1 << ") " << *(get<2>(qr)->begin()+num) << endl;
    return os;
}

#endif