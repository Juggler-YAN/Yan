#include <iostream>
#include <omp.h>

using namespace std;

int main() {
    int num_threads, thread_id;
#pragma omp parallel private(num_threads, thread_id)
{
    thread_id = omp_get_thread_num();
    cout << "This thread is: " << thread_id << endl;
    if (thread_id == 0) {
        num_threads = omp_get_num_threads();
        cout << "Total pf threads is: " << num_threads << endl;
    }
}
    return 0;
}