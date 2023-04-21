#ifndef QUERY_H
#define QUERY_H

#include <iostream>
#include <memory>
#include <set>
#include <string>
#include "TextQuery.h"

using namespace std;

namespace chapter15 {
    class Query_base {
        friend class Query;
    protected:
        using line_no = chapter10::TextQuery::line_no;
        virtual ~Query_base() = default;
    private:
        virtual chapter10::QueryResult eval(const chapter10::TextQuery&) const = 0;
        virtual string rep() const = 0;
    };

    class Query {
        friend Query operator~(const Query&);
        friend Query operator|(const Query&, const Query&);
        friend Query operator&(const Query&, const Query&);
    public:
        Query(const string&);
        chapter10::QueryResult eval(const chapter10::TextQuery &t) const {
            cout << "Query::eval()" << endl;
            return q->eval(t);
        }
        string rep() const { 
            cout << "Query::rep()" << endl;
            return q->rep();
        }
    private:
        Query(shared_ptr<Query_base> query): q(query) {
            cout << "Query(shared_ptr<Query_base>)" << endl;
        }
        shared_ptr<Query_base> q;
    };
    inline ostream& operator<<(ostream &os, const Query &query) {
        return os << query.rep();
    }

    class WordQuery : public Query_base {
        friend class Query;
        WordQuery(const string &s) : query_word(s) {
            cout << "WordQuery(const string &)" << endl;
        }
        chapter10::QueryResult eval(const chapter10::TextQuery &t) const {
            cout << "WordQuery::eval()" << endl;
            return t.query(query_word);
        }
        string rep() const {
            cout << "WordQuery::rep()" << endl;
            return query_word;
        }
        string query_word;
    };
    inline Query::Query(const string &s) : q(new WordQuery(s)) {
        cout << "Query(const string &)" << endl;
    }

    class NotQuery : public Query_base {
        friend Query operator~(const Query&);
        NotQuery(const Query &q) : query(q) {
            cout << "NotQuery(const Query &)" << endl;
        }
        string rep() const {
            cout << "NotQuery::rep()" << endl;
            return "~(" + query.rep() + ")";
        }
        chapter10::QueryResult eval(const chapter10::TextQuery&) const;
        Query query;
    };
    inline Query operator~(const Query &operand) {
        return shared_ptr<Query_base>(new NotQuery(operand));
    }
    chapter10::QueryResult NotQuery::eval(const chapter10::TextQuery &text) const {
        cout << "NotQuery::eval()" << endl;
        auto result = query.eval(text);
        auto ret_lines = make_shared<set<line_no>>();
        auto beg = result.begin(), end = result.end();
        auto sz = result.get_file()->size();
        for (size_t n = 0; n != sz; ++n) {
            if (beg == end || *beg != n) {
                ret_lines->insert(n);
            }
            else if (beg != end) {
                ++beg;
            }
        }
        return chapter10::QueryResult(rep(), ret_lines, result.get_file());
    }

    class BinaryQuery : public Query_base {
    protected:
        BinaryQuery(const Query &l, const Query &r, string s) : lhs(l), rhs(r), opSym(s) {
            cout << "BinaryQuery(const Query &, const Query &, string)" << endl;
        }
        string rep() const {
            cout << "BinaryQuery::rep()" << endl;
            return "(" + lhs.rep() + " " + opSym + " " + rhs.rep() + ")";
        }
        Query lhs, rhs;
        string opSym;
    };

    class AndQuery : public BinaryQuery {
        friend Query operator&(const Query&, const Query&);
        AndQuery(const Query &left, const Query &right) : BinaryQuery(left, right, "&") {
            cout << "AndQuery(const Query &, const Query &, string)" << endl;
        }
        chapter10::QueryResult eval(const chapter10::TextQuery&) const;
    };
    inline Query operator&(const Query &lhs, const Query &rhs) {
        return shared_ptr<Query_base>(new AndQuery(lhs, rhs));
    }
    chapter10::QueryResult AndQuery::eval(const chapter10::TextQuery &text) const {
        cout << "AndQuery::eval()" << endl;
        auto right = rhs.eval(text), left = lhs.eval(text);
        auto ret_lines = make_shared<set<line_no>>();
        set_intersection(left.begin(), left.end(), right.begin(), right.end(), inserter(*ret_lines, ret_lines->begin()));
        return chapter10::QueryResult(rep(), ret_lines, left.get_file());
    }

    class OrQuery : public BinaryQuery {
        friend Query operator|(const Query&, const Query&);
        OrQuery(const Query &left, const Query &right) : BinaryQuery(left, right, "|") {
            cout << "OrQuery(const Query &, const Query &, string)" << endl;
        }
        chapter10::QueryResult eval(const chapter10::TextQuery&) const;
    };
    inline Query operator|(const Query &lhs, const Query &rhs) {
        return shared_ptr<Query_base>(new OrQuery(lhs, rhs));
    }
    chapter10::QueryResult OrQuery::eval(const chapter10::TextQuery &text) const {
        cout << "OrQuery::eval()" << endl;
        auto right = rhs.eval(text), left = lhs.eval(text);
        auto ret_lines = make_shared<set<line_no>>(left.begin(), left.end());
        ret_lines->insert(right.begin(), right.end());
        return chapter10::QueryResult(rep(), ret_lines, left.get_file());
    }
}

#endif