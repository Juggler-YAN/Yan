#include <stdio.h>
#include <ctype.h>
#include <string.h>

int main(int argc, char * argv[]) {
    int ch;
    if (argc == 1 || (argc == 2 && !strcmp(argv[1], "-p"))) {
        while ((ch = getchar()) != EOF) {
            putchar(ch);
        }
    }
    if(!strcmp(argv[1], "-u")) {
        while ((ch = getchar()) != EOF) {
            putchar(toupper(ch));
        }
    }
    if(!strcmp(argv[1], "-l")) {
        while ((ch = getchar()) != EOF) {
            putchar(tolower(ch));
        }
    }
    return 0;
}