# Chapter 11

### Q1

```c
#include <stdio.h>

#define LEN 10

void getnchar(char [], int);

int main(void) {
    char input[LEN];
    getnchar(input, LEN);
    printf("%s", input);
    return 0;
}

void getnchar(char str[], int n) {
    int i = 0;
    while (i < n - 1) {
        str[i] = getchar();
        i++;
    }
    str[i] = '\0';
}
```

### Q2

```c
#include <stdio.h>

#define LEN 10

void getnchar(char [], int);

int main(void) {
    char input[LEN];
    getnchar(input, LEN);
    printf("%s", input);
    return 0;
}

void getnchar(char str[], int n) {
    int i = 0;
    while (i < n - 1) {
        str[i] = getchar();
        if (str[i] == ' ' || str[i] == '\t' || str[i] == '\n') {
            break;
        }
        i++;
    }
    str[i] = '\0';
}
```

### Q3

```c
#include <stdio.h>
#include <ctype.h>

#define LEN 10

void getword(char []);

int main(void) {
    char input[LEN];
    getword(input);
    printf("%s", input);
    return 0;
}

void getword(char str[]) {
    char c;
    int i = 0;
    c = getchar();
    while (isspace(c)) {
        c = getchar();
        continue;
    }
    while ((!isspace(c)) && i < LEN - 1) {
        *str++ = c;
        i++;
        c = getchar();
    }
    *str = '\0';
    while (c != '\n') {
        c = getchar();
        continue;
    }
}
```

### Q4

```c
#include <stdio.h>
#include <ctype.h>

#define LEN 10

void getword(char [], int);

int main(void) {
    char input[LEN];
    getword(input, LEN-1);
    printf("%s", input);
    return 0;
}

void getword(char str[], int n) {
    char c;
    int i = 0;
    c = getchar();
    while (isspace(c)) {
        c = getchar();
        continue;
    }
    while ((!isspace(c)) && i < n) {
        *str++ = c;
        i++;
        c = getchar();
    }
    *str = '\0';
    while (c != '\n') {
        c = getchar();
        continue;
    }
}
```

### Q5

```c
#include <stdio.h>

#define LEN 10

char * s_gets(char *, int);
char * mystrchr(char [], char);

int main(void) {
    char str[LEN], c;
    char * res;
    while (s_gets(str, LEN)) {
        c = getchar();
        while (getchar() != '\n') {
            continue;
        }
        res = mystrchr(str, c);
        if (res) {
            printf("%c\n", *res);
        }
        else {
            printf("Null pointer\n");
        }
    }
    return 0;
}

char * s_gets(char * st, int n) {
    char * ret_val;
    int i = 0;

    ret_val = fgets(st, n, stdin);
    if (ret_val) {
        while (st[i] != '\n' && st[i] != '\0') {
            i++;
        }
        if (st[i] == '\n') {
            st[i] = '\0';
        }
        else {
            while (getchar() != '\n') {
                continue;
            }
        }
    }
    return ret_val;
}

char * mystrchr(char str[], char c) {
    while (*str) {
        if (*str == c) {
            return str;
        }
        str++;
    }
    return NULL;
}
```

### Q6

```c
#include <stdio.h>

#define LEN 10

char * s_gets(char *, int);
int is_within(char, const char *);

int main(void) {
    char str[LEN], c;
    while (s_gets(str, LEN)) {
        c = getchar();
        while (getchar() != '\n') {
            continue;
        }
        printf("%d\n", is_within(c, str));
    }
    return 0;
}

char * s_gets(char * st, int n) {
    char * ret_val;
    int i = 0;

    ret_val = fgets(st, n, stdin);
    if (ret_val) {
        while (st[i] != '\n' && st[i] != '\0') {
            i++;
        }
        if (st[i] == '\n') {
            st[i] = '\0';
        }
        else {
            while (getchar() != '\n') {
                continue;
            }
        }
    }
    return ret_val;
}

int is_within(char c, const char * s) {
    while (*s) {
        if (*s == c) {
            return 1;
        }
        s++;
    }
    return 0;
}
```

### Q7

```c
#include <stdio.h>

#define LEN 10

char * s_gets(char *, int);
char * mystrncpy(char *, const char *, int);

int main(void) {
    int n;
    char s1[LEN], s2[LEN+1];
    while (s_gets(s2, LEN+1)) {
        scanf("%d", &n);
        while (getchar() != '\n') {
            continue;
        }
        mystrncpy(s1, s2, n);
        printf("%s %s\n", s1, s2);
    }
    return 0;
}

char * s_gets(char * st, int n) {
    char * ret_val;
    int i = 0;

    ret_val = fgets(st, n, stdin);
    if (ret_val) {
        while (st[i] != '\n' && st[i] != '\0') {
            i++;
        }
        if (st[i] == '\n') {
            st[i] = '\0';
        }
        else {
            while (getchar() != '\n') {
                continue;
            }
        }
    }
    return ret_val;
}

char * mystrncpy(char * s1, const char * s2, int n) {
    int i = 0;
    char * res = s1;
    while (i < n && *s2 != '\0') {
        *s1++ = *s2++;
        i++;
    }
    *s1 = '\0';
    return res;
}
```

### Q8

```c
#include <stdio.h>

#define LEN 10

char * s_gets(char *, int);
char * string_in(char *, char *);

int main(void) {
    char s1[LEN], s2[LEN];
    while (s_gets(s1, LEN)) {
        if (s_gets(s2, LEN)) {
            if (string_in(s1, s2)) {
                printf("%s is in %s\n", s2, s1);
            }
            else {
                printf("Null Pointer!!!\n");
            }
        }
    }
    return 0;
}

char * s_gets(char * st, int n) {
    char * ret_val;
    int i = 0;

    ret_val = fgets(st, n, stdin);
    if (ret_val) {
        while (st[i] != '\n' && st[i] != '\0') {
            i++;
        }
        if (st[i] == '\n') {
            st[i] = '\0';
        }
        else {
            while (getchar() != '\n') {
                continue;
            }
        }
    }
    return ret_val;
}

char * string_in(char * s1, char * s2) {
    while (*s1) {
        if (*s1 == *s2) {
            char * temp1 = s1,
                 * temp2 = s2;
            while (*temp2) {
                if (*temp1 != *temp2) {
                    break;
                }
                temp1++, temp2++;
            }
            if (!(*temp2)) {
                return s1;
            }
        }
        s1++;
    }
    return NULL;
}
```

### Q9

```c
#include <stdio.h>

#define LEN 10

char * s_gets(char *, int);
char * reverse(char *);

int main(void) {
    char s[LEN];
    while (s_gets(s, LEN)) {
        puts(reverse(s));
    }
    return 0;
}

char * s_gets(char * st, int n) {
    char * ret_val;
    int i = 0;

    ret_val = fgets(st, n, stdin);
    if (ret_val) {
        while (st[i] != '\n' && st[i] != '\0') {
            i++;
        }
        if (st[i] == '\n') {
            st[i] = '\0';
        }
        else {
            while (getchar() != '\n') {
                continue;
            }
        }
    }
    return ret_val;
}

char * reverse(char * s) {
    char * start = s,
         * end = s;
    while (*end) {
        end++;
    }
    end--;
    while (start < end) {
        char temp;
        temp = *start;
        *start = *end;
        *end = temp;
        start++, end--;
    }
    return s;
}
```

### Q10

```c
#include <stdio.h>

#define LEN 10

char * s_gets(char *, int);
void removeSpace(char *);

int main(void) {
    char s[LEN];
    while (s_gets(s, LEN)) {
        if (*s == '\0') {
            break;
        }
        removeSpace(s);
        puts(s);
    }
    return 0;
}

char * s_gets(char * st, int n) {
    char * ret_val;
    int i = 0;

    ret_val = fgets(st, n, stdin);
    if (ret_val) {
        while (st[i] != '\n' && st[i] != '\0') {
            i++;
        }
        if (st[i] == '\n') {
            st[i] = '\0';
        }
        else {
            while (getchar() != '\n') {
                continue;
            }
        }
    }
    return ret_val;
}

void removeSpace(char * s) {
    while (*s) {
        while (*s == ' ') {
            char * temp = s;
            while (*temp) {
                *temp = *(temp + 1);
                temp++;
            }
        }
        s++;
    }
}
```

### Q11

```c
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#define LIM 10
#define SIZE 20

void menu(void);
char * s_gets(char *, int);
void fun1(char * [], int);
void fun2(char * [], int);
void fun3(char * [], int);
int word_len(char *);
void fun4(char * [], int);

int main(void) {
    // 读取字符串
    char input[LIM][SIZE];
    char * ptstr[LIM];
    int ct = 0;
    while (ct < LIM && s_gets(input[ct], SIZE) != NULL && input[ct][0] != EOF) {
        ptstr[ct] = input[ct];
        ct++;
    }
    // 菜单
    int choice;
    menu();
    scanf("%d", &choice);
    while (getchar() != '\n') {
        continue;
    }
    while (choice != 5) {
        switch (choice) {
            case 1: {
                fun1(ptstr, ct);
                break;
            }
            case 2: {
                fun2(ptstr, ct);
                break;
            }
            case 3: {
                fun3(ptstr, ct);
                break;
            }
            case 4: {
                fun4(ptstr, ct);
                break;
            }
            default: {
                printf("输入错误!\n");
            }
        }
        menu();
        scanf("%d", &choice);
        while (getchar() != '\n') {
            continue;
        }
    }
    return 0;
}

void menu(void) {
    printf("*******************************************************************************\n");
    printf("选择数字以执行相应功能:\n");
    printf("1) 打印源字符串列表            2) 以ASCII中的顺序打印字符串\n");
    printf("3) 按长度递增顺序打印字符串    4) 按字符串中第一个单词的长度打印字符串\n");
    printf("5) 退出\n");
    printf("*******************************************************************************\n");
}

char * s_gets(char * st, int n) {
    char * ret_val;
    int i = 0;

    ret_val = fgets(st, n, stdin);
    if (ret_val) {
        while (st[i] != '\n' && st[i] != '\0') {
            i++;
        }
        if (st[i] == '\n') {
            st[i] = '\0';
        }
        else {
            while (getchar() != '\n') {
                continue;
            }
        }
    }
    return ret_val;
}

void fun1(char * strs[], int n) {
    for (int i = 0; i < n; i++) {
        puts(strs[i]);
//        printf("%s\n", strs[i]);
    }
}

void fun2(char * strs[], int n) {
    char * temp;
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (strcmp(strs[i], strs[j]) > 0) {
                temp = strs[i];
                strs[i] = strs[j];
                strs[j] = temp;
            }
        }
    }
    fun1(strs, n);
}

void fun3(char * strs[], int n) {
    char * temp;
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (strlen(strs[i]) > strlen(strs[j])) {
                temp = strs[i];
                strs[i] = strs[j];
                strs[j] = temp;
            }
        }
    }
    fun1(strs, n);
}

int word_len(char * s) {
    int len = 0;
    while (isblank(*s)) {
        s++;
    }
    while (*s) {
        if (isblank(*s)) {
            break;
        }
        len++;
        s++;
    }
    return len;
}

void fun4(char * strs[], int n) {
    char * temp;
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (word_len(strs[i]) > word_len(strs[j])) {
                temp = strs[i];
                strs[i] = strs[j];
                strs[j] = temp;
            }
        }
    }
    fun1(strs, n);
}
```

### Q12

```c
#include <stdio.h>
#include <ctype.h>

int main(void) {
    int ch;
    int len1 = 0,
        len2 = 0,
        len3 = 0,
        len4 = 0,
        len5 = 0;
    int flag = 1;
    while ((ch = getchar()) != EOF) {
        if (isspace(ch)) {
            flag = 1;
            continue;
        }
        else {
            if (flag == 1) {
                len1++;
                flag = 0;
            }
            if (isupper(ch)) {
                len2++;
            }
            if (islower(ch)) {
                len3++;
            }
            if (ispunct(ch)) {
                len4++;
            }
            if (isdigit(ch)) {
                len5++;
            }
        }
    }
    printf("单词数:%d\n", len1);
    printf("大写字母数:%d\n", len2);
    printf("小写字母数:%d\n", len3);
    printf("标点符号数:%d\n", len4);
    printf("数字数:%d\n", len5);
    return 0;
}
```

### Q13

```c
#include <stdio.h>

int main(int argc, char * argv[]) {
    int i = argc;
    while(--i) {
        printf("%s ", argv[i]);
    }
    printf("\b\n");
    return 0;
}
```

### Q14

```c
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char * argv[]) {
    double i = atof(argv[1]);
    int j = atoi(argv[2]);
    double res = 1.0;
    while(j--) {
        res *= i;
    }
    printf("%f", res);
    return 0;
}
```

### Q15

```c
#include <stdio.h>
#include <ctype.h>

int myatoi(char *);

int main() {
    printf("%d\n", myatoi("456132"));
    printf("%d\n", myatoi("456132c"));
    return 0;
}

int myatoi(char * s) {
    int num = 0, flag = 1;
    while (*s) {
        if (!(isdigit(*s))) {
            return 0;
        }
        else {
            num = num * 10 +(*s - '0');
        }
        s++;
    }
    return num;
}
```

### Q16

```c
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
```

