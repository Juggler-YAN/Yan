#include <iostream>
#include <omp.h>

using namespace std;

int main() {
    
#pragma omp parallel
{
    cout << "hellowolrd" << endl;
}
    return 0;
}