#include <iostream>
#include <algorithm>
#include <list>
#include <string>

using namespace std;

void elimDups(list<string>&);

int main() {
    list<string> words;
    string s;
    while (cin >> s) {
        words.push_back(s);
    }
    elimDups(words);
    for (const auto &s : words) {
        cout << s << " ";
    }
    cout << endl;
    
    return 0;
}

void elimDups(list<string> &words) {
    words.sort();
    words.unique();
}