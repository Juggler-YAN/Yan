# Chapter 16

### Q1

```c
// fun.h
#ifndef FUN_H_
#define FUN_H_
#define SIZE 20
#endif
```

```c
// main.c
#include <stdio.h>
#include "fun.h"

int main(void) {
    printf("%d", SIZE);
    return 0;
}
```

### Q2

```c
#include <stdio.h>

#define HARMEAN(X,Y) (2/(1/(X)+1/(Y)))

int main(void) {
    printf("%f\n", HARMEAN(1.0,2.0));
    return 0;
}
```

### Q3

```c
#include <stdio.h>
#include <math.h>

#define RAD_TO_DEG (180/(4*atan(1)))

typedef struct vector {
    double length;
    double angle;
} vector;
typedef struct point {
    double x;
    double y;
} point;

point transform(vector);

int main(void) {
    vector val = {10, 45};
    point res;
    res = transform(val);
    printf("(%f,%f)\n", res.x, res.y);
    return 0;
}

point transform(vector val) {
    point res;
    res.x = val.length * cos(val.angle/RAD_TO_DEG);
    res.y = val.length * sin(val.angle/RAD_TO_DEG);
    return res;
}
```

### Q4

```c
#include <stdio.h>
#include <time.h>
#include <unistd.h>

void timeTest(double);

int main(void) {
    double timeDelay;
    printf("Enter a delay time(s):");
    scanf("%lf", &timeDelay);
    timeTest(timeDelay);
    return 0;
}

void timeTest(double timeDelay) {
    clock_t start, end;
    start = clock();
    sleep(timeDelay);
    end = clock();
    printf("Delay %.2lfs, spend %.2lfs", timeDelay, (double)
           (end-start)/(double)CLOCKS_PER_SEC);
}
```

### Q5

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 5
#define MEMBERS 4

void rands(int [], int, int);

int main(void) {
    int a[SIZE] = {8, 4, 5, 2, 7};
    rands(a, SIZE, MEMBERS);
    return 0;
}

void rands(int a[], int len, int times) {
    int * p = (int *)malloc(times*sizeof(int));
    srand((unsigned)time(0));
    for (int i = 0; i < times; i++) {
        p[i] = rand() % len;
        for (int j = i-1; j >= 0; j--) {
            if (p[i] == p[j]) {
                i--;
                break;
            }
        }
    }
    for (int i = 0; i < times; i++) {
        printf("%d:%d\n", i+1, a[p[i]]);
    }
    free(p);
}
```

### Q6

```c
#include <stdio.h>
#include <stdlib.h>

struct VAL {
    double val;
};

#define NUM 4
void showarray(const struct VAL ar[], int n);
int mycomp(const void * p1, const void * p2);

int main(void) {
    struct VAL vals[NUM] = {
        {5.0},
        {2.0},
        {3.0},
        {4.0}
    };
    puts("\nUnsorted list:");
    showarray(vals, NUM);
    qsort(vals, NUM, sizeof(struct VAL ), mycomp);
    puts("\nSorted list:");
    showarray(vals, NUM);
    return 0;
}

void showarray(const struct VAL ar[], int n) {
    int index;
    for (index = 0; index < n; index++) {
        printf("%9.4f ", ar[index].val);
    }
    putchar('\n');
}

int mycomp(const void * p1, const void * p2) {
    const struct VAL * a1 = (const struct VAL *) p1;
    const struct VAL * a2 = (const struct VAL *) p2;
    if (a1->val < a2->val) {
        return -1;
    }
    else if (a1->val == a2->val) {
        return 0;
    }
    else {
        return 1;
    }
}
```

### Q7

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

void show_array(const double ar[], int n);
double * new_d_array(int n, ...);

int main() {
    double *p1;
    double *p2;

    p1 = new_d_array(5, 1.2, 2.3, 3.4, 4.5, 5.6);
    p2 = new_d_array(4, 100.0, 20.00, 8.08, -1890.0);
    show_array(p1, 5);
    show_array(p2, 4);
    free(p1);
    free(p2);

    return 0;
}

void show_array(const double ar[], int n) {
    printf("The %d elements are :", n);
    for (int i = 0; i < n; i++) {
        printf("%.2lf", ar[i]);
        (i != n-1) ? printf(",") : printf(".");
    }
    printf("\n");
    return;
}

double * new_d_array(int n, ...) {
    va_list ap;
    va_start(ap, n);
    double *pt;
    pt = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        pt[i] = va_arg(ap, double);
    }
    va_end(ap);
    return pt;
}
```

