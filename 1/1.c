#include <stdio.h>

int fibonacci(int);

int main(void) {
    printf("%d", fibonacci(5));
    return 0;
}

int fibonacci(int n) {
    int a1 = 1,
        a2 = 1,
        a3 = a1 + a2;
    if (n == 1) {
        return a1;
    }
    if (n == 2) {
        return a2;
    }
    n -= 2;
    while (n) {
        a3 = a1 + a2;
        a1 = a2;
        a2 = a3;
        n--;
    }
    return a3;
}