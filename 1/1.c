#include <stdio.h>

#define SIZE 4

void menu(void);
void printA(void);
void printB(void);
void printC(void);
void printD(void);

int main(void) {
    void (*ptf[SIZE])(void) = {printA, printB, printC, printD};
    menu();
    char ch;
    while ((ch = getchar()) != 'q') {
        switch(ch) {
            case 'a': {
                ptf[0]();
                break;
            }
            case 'b': {
                ptf[1]();
                break;
            }
            case 'c': {
                ptf[2]();
                break;
            }
            case 'd': {
                ptf[3]();
                break;
            }
            default: {
                printf("Enter \'a\', \'b\', \'c\', \'d\' or  \'q\'\n");
            }
        }
        menu();
        while (getchar() != '\n') ;
    }
	return 0;
}

void menu(void) {
    printf("**************************\n");
    printf("a)A         b)B\n");
    printf("c)C         d)D\n");
    printf("q)quit\n");
    printf("**************************\n");
}

void printA(void) {
    printf("A\n");
}

void printB(void) {
    printf("B\n");
}

void printC(void) {
    printf("C\n");
}

void printD(void) {
    printf("D\n");
}