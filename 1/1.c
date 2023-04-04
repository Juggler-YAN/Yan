#include <stdio.h>

void one(void);
void two(void);

int main(void) {
    printf("starting now:\n");
    one();
    printf("done!");
    return 0;
}

void one(void) {
    printf("one\n");
    two();
    printf("three\n");
}

void two(void) {
    printf("two\n");
}