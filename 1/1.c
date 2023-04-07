#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SIZE 40
const char s[] = " .':~*= %#";

int getpos(const char *, char);
char correctVal(int, int, int, int, char [*][*]);

int main(void) {
    int m = 20, n = 31;
    char filename[SIZE];
    printf("Enter the file name: ");
    scanf("%s", filename);
    FILE *fp;
    if ((fp = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "Can't open file %s.", filename);
        exit(EXIT_FAILURE);
    }
    char ch;
    char a[m][n], b[m][n];
    for (int i = 0; i < m; i++) {
        ch = getc(fp);
        for (int j = 0; j < n - 1; j++) {
            if (ch - '0' < 0 || ch - '0' > 9) {
                a[i][j] = ' ';
            }
            else {
                a[i][j] = s[ch-'0'];
            }
            if (j == n - 2) {
                break;
            }
            ch = getc(fp);
            ch = getc(fp);
        }
        ch = getc(fp);
        a[i][n - 1] = '\0';
    }
    for (int i = 0; i < m; i++) {
        for(int j = 0; j < n-1; j++){
            b[i][j] = correctVal(i, j, m, n, a);
        }
        b[i][n-1] = '\0';
    }
    for (int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++){
            printf("%c", b[i][j]);
        }
        printf("\n");
    }
    if (fclose(fp) != 0){
        fprintf(stderr, "Can't close file %s.", filename);
        exit(EXIT_FAILURE);
    }
    return 0;
}

int getpos(const char * str, char c) {
    const char *p = str;
    while (*p) {
        if (*p == c) {
            break;
        }
        p++;
    }
    if (*p == '\0') {
        return -1;
    }
    return p-str;
}

char correctVal(int m, int n, int rows, int cols, char a[rows][cols]) {
    int up = 0,
        down = 0,
        left = 0,
        right = 0,
        edge = 0;
    if (m != 0) {
        up = 1;
        edge++;
    }
    if (m != rows-1) {
        down = 1;
        edge++;
    }
    if (n != 0) {
        left = 1;
        edge++;
    }
    if (n != cols-2) {
        right = 1;
        edge++;
    }
    if (
        (up?(abs(getpos(s, a[m-1][n]) - getpos(s, a[m][n])) > 1):1) &&
        (down?(abs(getpos(s, a[m+1][n]) - getpos(s, a[m][n])) > 1):1) &&
        (left?(abs(getpos(s, a[m][n-1]) - getpos(s, a[m][n])) > 1):1) &&
        (right?(abs(getpos(s, a[m][n+1]) - getpos(s, a[m][n])) > 1):1)
        ) {
        int sum = (up?getpos(s, a[m-1][n]):0) + (down?getpos(s, a[m+1][n]):0) +
                  (left?getpos(s, a[m][n-1]):0) + (right?getpos(s, a[m][n+1]):0);
        return s[(int)round((double)sum/edge)];
    }
    return a[m][n];
}