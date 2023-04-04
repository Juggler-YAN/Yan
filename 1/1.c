#include <stdio.h>

int main(void) {
    int val = 5,
        week = 0;
    while (val <= 150) {
        week++;
        val = (val - week) * 2;
        printf("Week%d:%d\n", week, val);
    }
    return 0;
}