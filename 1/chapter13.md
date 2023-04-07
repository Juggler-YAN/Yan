# Chapter 13

### Q1

```c
#include <stdio.h>
#include <stdlib.h>

#define LEN 20

int main(void) {
    int ch;
    FILE * fp;
    unsigned long count = 0;
    char filename[LEN];

    printf("Input filename: ");
    scanf("%s", filename);
    if ((fp = fopen(filename, "r")) == NULL) {
        printf("Can't open %s\n", filename);
        exit(EXIT_FAILURE);
    }
    while ((ch = getc(fp)) != EOF) {
        putc(ch, stdout);
        count++;
    }
    fclose(fp);
    printf("\nFile %s has %lu characters\n", filename, count);
    return 0;
}
```

### Q2

```c
#include <stdio.h>
#include <stdlib.h>

#define LEN 256

int main(int argc, char * argv[]) {
    int ch;
    char buf[LEN];
    size_t bytes;
    FILE * in, * out;
    if (argc != 3) {
        fprintf(stderr, "Usage: %s filename filename\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    if ((in = fopen(argv[1], "rb")) == NULL) {
        printf("Can't open %s\n", argv[1]);
        exit(EXIT_FAILURE);
    }
    if ((out = fopen(argv[2], "wb")) == NULL) {
        printf("Can't open %s\n", argv[2]);
        exit(EXIT_FAILURE);
    }
    while ((bytes = fread(buf, sizeof(char), LEN, in)) > 0) {
        fwrite(buf, sizeof(char), bytes, out);
    }
    if (fclose(in) != 0) {
        printf("Can't close %s\n", argv[1]);
        exit(EXIT_FAILURE);
    }
    if (fclose(out) != 0) {
        printf("Can't close %s\n", argv[2]);
        exit(EXIT_FAILURE);
    }
    return 0;
}
```

### Q3

```c
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#define LEN 20

int main(void) {
    int ch;
    FILE * fp, * temp;
    char filename[LEN];

    printf("Input filename: ");
    scanf("%s", filename);
    if ((fp = fopen(filename, "r+")) == NULL) {
        printf("Can't open %s\n", filename);
        exit(EXIT_FAILURE);
    }
    while ((ch = getc(fp)) != EOF) {
        fseek(fp, -1L, SEEK_CUR);
        putc(toupper(ch), fp);
        // fseek(fp, 0L, SEEK_CUR);    // 改变文件状态以正常读取下一个字符
    }
    rewind(fp);
    while ((ch = getc(fp)) != EOF) {
        putc(ch, stdout);
    }
    if (fclose(fp) != 0) {
        printf("Can't close %s\n", filename);
        exit(EXIT_FAILURE);
    }
    return 0;
}
```

### Q4

```c
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char * argv[]) {
    int ch;
    FILE * fp;
    if (argc < 2) {
        fprintf(stderr, "Usage: %s filename[s]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    for (int i = 1; i < argc; i++) {
        if ((fp = fopen(argv[i], "r")) == NULL) {
            printf("Can't open %s\n", argv[i]);
            exit(EXIT_FAILURE);
        }
        printf("%s\n", argv[i]);
        while ((ch = getc(fp)) != EOF) {
            putc(ch, stdout);
        }
        printf("\n");
        if (fclose(fp) != 0) {
            printf("Can't close %s\n", argv[i]);
            exit(EXIT_FAILURE);
        };
    }
    return 0;
}
```

### Q5

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFSIZE 4096

void append(FILE*, FILE*);

int main (int argc, char * argv []) {
    FILE *fa, *fs;
    int files = 0;
    int ch;
    if (argc < 3) {
        fprintf(stderr, "Usage: %s targetfile sourcefile[s]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    if ((fa = fopen(argv[1], "a+")) == NULL) {
        fprintf(stderr, "Can't open %s.", argv[1]);
        exit(EXIT_FAILURE);
    }
    if (setvbuf(fa, NULL, _IOFBF, BUFSIZE) != 0) {
        fputs("Can't create input buffer\n", stderr);
        exit(EXIT_FAILURE);
    }
    for (int i = 2; i < argc; i++){
        if (strcmp(argv[1], argv[i]) == 0) {
            fputs("Can't append file to itself.\n", stderr);
            exit(EXIT_FAILURE);
        }
        else if ((fs = fopen(argv[i],"r")) == NULL){
            fprintf(stderr, "Can't open %s.", argv[i]);
            exit(EXIT_FAILURE);
        }
        else {
            if (setvbuf(fs, NULL, _IOFBF, BUFSIZE) != 0) {
                fputs("Can't create input buffer\n", stderr);
                continue;
            }
            append(fs, fa);
            if (ferror(fs) != 0) {
                fprintf(stderr, "Error in reading file %s.\n", argv[1]);
                exit(EXIT_FAILURE);
            }
            if (ferror(fa) != 0) {
                fprintf(stderr, "Error in reading file %s.\n", argv[i]);
                exit(EXIT_FAILURE);
            }
            fclose(fs);
            files++;
        }
    }
    printf("Done appending. %d files appended.\n", files);
    rewind(fa);
    printf("%s contents:\n", argv[1]);
    while ((ch = getc(fa)) != EOF){
        putchar(ch);
    }
    puts("\nDone displaying.");
    fclose(fa);
    return 0;
}

void append(FILE * source, FILE * dest) {
    size_t bytes;
    static char temp[BUFSIZE];

    while ((bytes = fread(temp, sizeof(char), BUFSIZE, source)) > 0) {
        fwrite(temp, sizeof(char), bytes, dest);
    }
}
```

### Q6

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LEN 40

int main() {
    FILE *fp1, *fp2;
    int ch;
    int count = 0;
    char a[LEN], b[LEN];
    printf("Enter the filename you want to reduced: ");
    scanf("%s", a);
    if ((fp1 = fopen(a,"r")) == NULL) {
        fprintf(stderr, "Can't open %s.", a);
        exit(EXIT_FAILURE);
    }
    strncpy(b, a, LEN - 5);
    b[LEN-5] = '\0';
    strcat(b, ".red");
    if ((fp2 = fopen(b, "w")) == NULL) {
        fprintf(stderr, "Can't write in %s.", b);
        exit(EXIT_FAILURE);
    }
    while ((ch = getc(fp1)) != EOF){
        if(count++%3 == 0){
            putc(ch, fp2);
        }
    }
    if (fclose(fp1) != 0 || fclose(fp2) != 0){
        fprintf(stderr,"Error in closing files %s or %s.", a, b);
        exit(EXIT_FAILURE);
    }
    return 0;
}
```

### Q7

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main (int argc, char *argv[]) {
    FILE *fp1, *fp2;
    int ch1, ch2;
    int flag1 = 1, flag2 = 1;
    if (argc != 3) {
        fprintf(stderr, "Usage: %s file1 file12", argv[0]);
        exit(EXIT_FAILURE);
    }
    if ((fp1 = fopen(argv[1], "r")) == NULL || (fp2 = fopen(argv[2], "r")) == NULL) {
        fputs("Error in reading files.", stderr);
        exit(EXIT_FAILURE);
    }
    ch1 = getc(fp1);
    ch2 = getc(fp2);
    if (ch1 == EOF) {
        flag1 = 0;
    }
    if (ch2 == EOF) {
        flag2 = 0;
    }
    while (ch1 != EOF || ch2 != EOF) {
        while (ch1 != EOF && ch1 != '\n') {
            putchar(ch1);
            ch1 = getc(fp1);
        }
        if (ch1 == '\n') {
            putchar(ch1);
            ch1 = getc(fp1);
        }
        if (ch1 == EOF && flag1) {
            putchar('\n');
            flag1 = 0;
        }
        while (ch2 != EOF && ch2 != '\n') {
            putchar(ch2);
            ch2 = getc(fp2);
        }
        if (ch2 == '\n') {
            putchar(ch2);
            ch2 = getc(fp2);
        }
        if (ch2 == EOF && flag2) {
            putchar('\n');
            flag2 = 0;
        }
    }
    if (fclose(fp1) != 0 || fclose(fp2) != 0){
        fputs("Error in writing files.", stderr);
        exit(EXIT_FAILURE);
    }
    return 0;
}
```

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

int main (int argc, char *argv[]) {
    FILE *fp1, *fp2;
    int ch1, ch2;
    int flag1 = 1, flag2 = 1;
    if (argc != 3) {
        fprintf(stderr, "Usage: %s file1 file12", argv[0]);
        exit(EXIT_FAILURE);
    }
    if ((fp1 = fopen(argv[1], "r")) == NULL || (fp2 = fopen(argv[2], "r")) == NULL) {
        fputs("Error in reading files.", stderr);
        exit(EXIT_FAILURE);
    }
    ch1 = getc(fp1);
    ch2 = getc(fp2);
    while (ch1 != EOF || ch2 != EOF) {
        while (ch1 != EOF && ch1 != '\n') {
            putchar(ch1);
            ch1 = getc(fp1);
        }
        while (ch2 != EOF && ch2 != '\n') {
            putchar(ch2);
            ch2 = getc(fp2);
        }
        if (ch1 == '\n' || ch2 == '\n') {
            putchar('\n');
            if (ch1 != EOF) {
                ch1 = getc(fp1);
            }
            if (ch2 != EOF) {
                ch2 = getc(fp2);
            }
        }
    }
    if (fclose(fp1) != 0 || fclose(fp2) != 0){
        fputs("Error in writing files.", stderr);
        exit(EXIT_FAILURE);
    }
    return 0;
}
```

### Q8

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int nums(char, FILE *);

int main (int argc, char *argv[]) {
    FILE *fp;
    char ch;
    int count;
    if (argc < 2) {
        fprintf(stderr, "Usage: %s character filename[s]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    if (strlen(argv[1]) != 1) {
        fprintf(stderr, "%s must be a character!\n", argv[1]);
        exit(EXIT_FAILURE);
    }
    if (argc == 2) {
        fp = stdin;
        printf("Please Enter a string: ");
        count = nums(argv[1][0], fp);
        printf("%c appeared %d time[s].", argv[1][0], count);
    }
    else {
        for (int i = 2; i < argc; i++) {
            if ((fp = fopen(argv[i], "r")) == NULL) {
                fprintf(stderr, "Can't open file %s.\n", argv[i]);
                continue;
            }
            count = nums(argv[1][0], fp);
            printf("%c appeared %d time[s] in file %s.\n", argv[1][0], count, argv[i]);
            if (fclose(fp) != 0) {
                fprintf(stderr, "Can't close file %s.\n", argv[i]);
                continue;
            }
        }
    }
    return 0;
}

int nums(char c, FILE * fp) {
    int ch;
    int n = 0;
    while ((ch = getc(fp)) != EOF) {
        if (c == ch) {
            n++;
        }
    }
    return n;
}
```

### Q9

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX 41

int main(void) {
    FILE *fp;
    char words[MAX];
    if ((fp = fopen("test.txt", "a+")) == NULL) {
        fprintf(stdout, "Can't open \"data/13-9.txt\" file.");
        exit(EXIT_FAILURE);
    }
    int c;
    int cnt = 0;
    while ((c = getc(fp)) != EOF) {
        if (c == '\n') {
            cnt++;
        }
    }
    rewind(fp);
    puts("Enter words to add to the file; press the #");
    puts("key at the beginning of a line to terminate.");
    while ((fscanf(stdin, "%40s", words) == 1) && (words[0] != '#')) {
        fprintf(fp, "%d:%s\n", ++cnt, words);
    }
    puts("File contents: ");
    rewind(fp);
    while (fscanf(fp, "%s", words) == 1) {
        puts(words);
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "Error closing file.");
    }
    return 0;
}
```

### Q10

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 100

int main(void) {
    char filename[SIZE];
    printf("Enter the file name: ");
    scanf("%99s", filename);
    FILE *fp;
    if ((fp = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "Can't open file %s.", filename);
        exit(EXIT_FAILURE);
    }
    long pos;
    printf("Enter a file location: ");
    while (scanf("%ld", &pos) == 1){
        if (pos <= 0) {
            break;
        }
        fseek(fp, pos, SEEK_SET);
        int ch;
        while ((ch = getc(fp)) != '\n' && ch != EOF){
            putchar(ch);
        }
        printf("\n");
        printf("Enter a file location: ");
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "Can't close file %s.", filename);
        exit(EXIT_FAILURE);
    }
    return 0;
}
```

### Q11

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LEN 256

int main(int argc, char * argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s string filename", argv[0]);
        exit(EXIT_FAILURE);
    }
    char s[LEN];
    FILE *fp;
    if ((fp = fopen(argv[2], "r")) == NULL) {
        fprintf(stderr, "Can't open file %s.", argv[2]);
        exit(EXIT_FAILURE);
    }
    while ((fgets(s, LEN, fp)) != NULL){
        if (strstr(s, argv[1])) {
            fputs(s, stdout);
        }
    }
    if (fclose(fp) != 0){
        fprintf(stderr, "Can't close the file.");
        exit(EXIT_FAILURE);
    }
    return 0;
}
```

### Q12

```
0 0 9 0 0 0 0 0 0 0 0 0 5 8 9 9 8 5 2 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 9 0 0 0 0 0 0 0 5 8 9 9 8 5 5 2 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 5 8 1 9 8 5 4 5 2 0 0 0 0 0 0 0 0 0
0 0 0 0 9 0 0 0 0 0 0 0 5 8 9 9 8 5 0 4 5 2 0 0 0 0 0 0 0 0
0 0 9 0 0 0 0 0 0 0 0 0 5 8 9 9 8 5 0 0 4 5 2 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 5 8 9 1 8 5 0 0 0 4 5 2 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 5 8 9 9 8 5 0 0 0 0 4 5 2 0 0 0 0 0
5 5 5 5 5 5 5 5 5 5 5 5 5 8 9 9 8 5 5 5 5 5 5 5 5 5 5 5 5 5
8 8 8 8 8 8 8 8 8 8 8 8 5 8 9 9 8 5 8 8 8 8 8 8 8 8 8 8 8 8
9 9 9 9 0 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 3 9 9 9 9 9 9 9
8 8 8 8 8 8 8 8 8 8 8 8 5 8 9 9 8 5 8 8 8 8 8 8 8 8 8 8 8 8
5 5 5 5 5 5 5 5 5 5 5 5 5 8 9 9 8 5 5 5 5 5 5 5 5 5 5 5 5 5
0 0 0 0 0 0 0 0 0 0 0 0 5 8 9 9 8 5 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 5 8 9 9 8 5 0 0 0 0 6 6 0 0 0 0 0 0
0 0 0 0 2 2 0 0 0 0 0 0 5 8 9 9 8 5 0 0 5 6 0 0 6 5 0 0 0 0
0 0 0 0 3 3 0 0 0 0 0 0 5 8 9 9 8 5 0 5 6 1 1 1 1 6 5 0 0 0
0 0 0 0 4 4 0 0 0 0 0 0 5 8 9 9 8 5 0 0 5 6 0 0 6 5 0 0 0 0
0 0 0 0 5 5 0 0 0 0 0 0 5 8 9 9 8 5 0 0 0 0 6 6 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 5 8 9 9 8 5 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 5 8 9 9 8 5 0 0 0 0 0 0 0 0 0 0 0 0
```

```c
#include <stdio.h>
#include <stdlib.h>

#define SIZE 40
#define ROWS 20
#define COLS 31

int main(void) {
    char filename[SIZE];
    printf("Enter the file name: ");
    scanf("%s", filename);
    FILE *fp;
    if ((fp = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "Can't open file %s.", filename);
        exit(EXIT_FAILURE);
    }
    char ch;
    char a[ROWS][COLS];
    char s[] = " .':~*= %#";
    for (int i = 0; i < ROWS; i++) {
        // ch = getc(fp);
        for (int j = 0; j < COLS-1; j++) {
            ch = getc(fp);
            a[i][j] = ch;
            if (ch - '0' < 0 || ch - '0' > 9) {
                a[i][j] = ' ';
            }
            else {
                a[i][j] = s[ch-'0'];
            }
            ch = getc(fp);
        }
        a[i][COLS - 1] = '\0';
    }
    for (int i = 0; i < ROWS; i++) {
        for(int j = 0; j < COLS; j++){
            printf("%c", a[i][j]);
        }
        printf("\n");
    }
    if (fclose(fp) != 0){
        fprintf(stderr, "Can't close file %s.", filename);
        exit(EXIT_FAILURE);
    }
    return 0;
}
```

### Q13

```c
#include <stdio.h>
#include <stdlib.h>

#define SIZE 40

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
    char a[m][n];
    char s[] = " .':~*= %#";
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
        for(int j = 0; j < n; j++){
            printf("%c", a[i][j]);
        }
        printf("\n");
    }
    if (fclose(fp) != 0){
        fprintf(stderr, "Can't close file %s.", filename);
        exit(EXIT_FAILURE);
    }
    return 0;
}
```

### Q14

```c
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
```

