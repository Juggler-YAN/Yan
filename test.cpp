#include <iostream>
#include <fstream>

#define N 17
#define NUM 8

using namespace std;

int main() {
    ofstream out("chapter"+to_string(N)+".md");
    out << "# Chapter " << N << endl;
    out << endl;
    for (int i = 1; i <= NUM; ++i) {
        out << "### Q" << i << endl;
        out << endl;
        out << "```c" << endl;
        out << endl;
        out << "```" << endl;
        out << endl;
    }
    out.close();
    return 0;
}