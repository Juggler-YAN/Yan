# Chapter 9

### Q1

```c
#include <stdio.h>

double min(double, double);

int main(void) {
    printf("The smallest of %.2f and %.2f is %.2f.", 2.0, 3.0, min(2.0, 3.0));
    return 0;
}

double min(double x, double y) {
    return (x < y ? x : y);
}
```

### Q2

```c
#include <stdio.h>

void chline(char, int, int);

int main(void) {
    chline('a', 2, 3);
    return 0;
}

void chline(char ch, int i, int j) {
    int m, n;
    for (m = 0; m < j; m++) {
        for (n = 0; n < i; n++) {
            printf("%c", ch);
        }
        printf("\n");
    }
}
```

### Q3

```c
#include <stdio.h>

void chline(char, int, int);

int main(void) {
    chline('a', 2, 3);
    return 0;
}

void chline(char ch, int i, int j) {
    int m, n;
    for (m = 0; m < j; m++) {
        for (n = 0; n < i; n++) {
            printf("%c", ch);
        }
        printf("\n");
    }
}
```

### Q4

```c
#include <stdio.h>

double findHarmon(double, double);

int main(void) {
    printf("%f", findHarmon(2.0, 2.0));
    return 0;
}

double findHarmon(double a,double b) {
    return 2 / (1 / a + 1 / b);
}
```

### Q5

```c
#include <stdio.h>

void larger_of(double *, double *);

int main(void) {
    double x = 2.0,
           y = 3.0;
    larger_of(&x, &y);
    printf("%f %f", x, y);
    return 0;
}

void larger_of(double * x, double * y) {
    (*x < *y) ? (*x = *y) : (*y = *x);
}
```

### Q6

```c
#include <stdio.h>

void sortSToL (double *, double *, double *);

int main(void) {
    double x = 2.0,
           y = 3.0,
           z = 4.0;
    sortSToL(&x, &y, &z);
    printf("%f %f %f", x, y, z);
    return 0;
}

void sortSToL(double * x, double * y, double * z) {
    double temp;
    if (*x > *y) {
        temp = *x, *x = *y, *y = temp;
    }
    if (*y > *z) {
        temp = *y, *y = *z, *z = temp;
    }
    if (*x > *y) {
        temp = *x, *x = *y, *y = temp;
    }
}
```

### Q7

```c
#include <stdio.h>

int isLetter(char);
void getLetterVal(void);

int main(void) {
    getLetterVal();
    return 0;
}

void getLetterVal(void) {
    int ch;
    while ((ch = getchar()) != EOF) {
        if (isLetter(ch) != -1) {
            printf("%c is a letter, its position in the "
                   "alphabet is %d.\n", ch, isLetter(ch));
        }
        else {
            printf("%c is not a letter.\n", ch);
        }
    }
}

int isLetter(char ch) {
    if (ch >= 'a' && ch <= 'z') {
        return ch-'a'+1;
    }
    if (ch >= 'A' && ch <= 'Z') {
        return ch-'A'+1;
    }
    return -1;
}
```

### Q8

```c
#include <stdio.h>

double power(double, int);

int main(void) {
    double x;
    int n;
    while (scanf("%lf %d", &x, &n) == 2) {
        printf("%f\n", power(x, n));
    }
    return 0;
}

double power(double x, int n) {
    double res = 1;
    if (n > 0) {
        for (int i = 0; i < n; i++) {
            res *= x;
        }
    }
    else if (n < 0) {
        if (x == 0.0) {
            res = 0.0;
        }
        else {
            for (int i = 0; i < -n; i++) {
                res *= x;
            }
            res = 1.0 / res;
        }
    }
    else {
        res = 1.0;
    }
    return res;
}
```

### Q9

```c
#include <stdio.h>

double power(double, int);
double powerCore(double, int);

int main(void) {
    double x;
    int n;
    while (scanf("%lf %d", &x, &n) == 2) {
        printf("%f\n", power(x, n));
    }
    return 0;
}

double powerCore(double x, int n) {
    if (n == 1) {
        return x;
    }
    else {
        return x * powerCore(x, n - 1);
    }
}

double power(double x, int n) {
    if (n > 0) {
        return powerCore(x, n);
    }
    else if (n < 0) {
        if (x == 0.0) {
            return 0.0;
        }
        return 1 / powerCore(x, -n);
    }
    else {
        return 1.0;
    }
}
```

### Q10

```c
#include <stdio.h>

void to_base_n(int, int);

int main(void) {
    to_base_n(129, 8);
    return 0;
}

void to_base_n(int x, int n) {
    int r;
    r = x % n;
    if (x >= n) {
        to_base_n(x/n, n);
    }
    printf("%d", r);
}
```

### Q11

```c
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
```

