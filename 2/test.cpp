#include <iostream>
#include <fstream>
#include "TextQuery.h"
#include "Query.h"

int main() {
    Query q = Query("fiery") & Query("bird") | Query("wind");
    ifstream in("test.txt");
    print(cout, q.eval(TextQuery(in)));
    return 0;
}