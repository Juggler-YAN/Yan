#include <iostream>
#include <fstream>
#include "TextQuery.h"

using namespace std;

void runQueries(ifstream &);

int main() {
    ifstream in("./test.txt");
    runQueries(in);
    return 0;
}

void runQueries(ifstream &infile) {
    TextQuery tq(infile);
    while (true) {
        cout << "enter word to look for, or q to quit: ";
        string s;
        if (!(cin >> s) || s == "q") break;
        print(cout, tq.query(s)) << endl;
    }
}