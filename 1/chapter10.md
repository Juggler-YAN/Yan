# Chapter 10

### Q1

```c
#include <stdio.h>

#define MONTHS 12
#define YEARS 5

int main(void) {
    const float rain[YEARS][MONTHS] = {
        {4.3, 4.3, 4.3, 3.0, 2.0, 1.2, 0.2, 0.2, 0.4, 2.4, 3.5, 6.6},
        {8.5, 8.2, 1.2, 1.6, 2.4, 0.0, 5.2, 0.9, 0.3, 0.9, 1.4, 7.3},
        {9.1, 8.5, 6.7, 4.3, 2.1, 0.8, 0.2, 0.2, 1.1, 2.3, 6.1, 8.4},
        {7.2, 9.9, 8.4, 3.3, 1.2, 0.8, 0.4, 0.0, 0.6, 1.7, 4.3, 6.2},
        {7.6, 5.6, 3.8, 2.8, 3.8, 0.2, 0.0, 0.0, 0.0, 1.3, 2.6, 5.2}
    };
    int year, month;
    float subtot, total;
    printf(" YEAR   RAINFALL  (inches)\n");
    for (year = 0, total = 0; year < YEARS; year++) {
        for (month = 0, subtot = 0; month < MONTHS; month++)
            subtot += *(*(rain + year) + month);
        printf("%5d %15.1f\n", 2010 + year, subtot);
        total += subtot;
    }
    printf("\nThe yearly average is %.1f inches.\n\n", total / YEARS);
    printf("MONTHLY AVERAGES:\n\n");
    printf(" Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Seq  Oct  Mov  Dec\n");

    for (month = 0; month < MONTHS; month++) {
        for (year = 0, subtot = 0; year < YEARS; year++)
            subtot += *(*(rain + year) + month);
        printf("%4.1f ", subtot / YEARS);
    }
    printf("\n");

    return 0;
}
```

### Q2

```c
#include <stdio.h>

void copy_arr(double [], const double [], int);
void copy_ptr(double *, const double *, int);
void copy_ptrs(double *, const double *, const double *);

int main(void) {
    double source[5] = {1.1, 2.2, 3.3, 4.4, 5.5};
    double target1[5];
    double target2[5];
    double target3[5];
    copy_arr(target1, source, 5);
    copy_ptr(target2, source, 5);
    copy_ptrs(target3, source, source + 5);
    for (int i = 0; i < 5; i++) {
        printf("%f ", source[i]);
    }
    printf("\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", target1[i]);
    }
    printf("\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", target2[i]);
    }
    printf("\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", target3[i]);
    }
    printf("\n");
    return 0;
}

void copy_arr(double target[], const double source[], int n) {
    for (int i = 0; i < n; i++) {
        target[i] = source[i];
    }
}

void copy_ptr(double * target, const double * source, int n) {
    for (int i = 0; i < n; i++) {
        *target++ = *source++;
    }
}

void copy_ptrs(double * target, const double * source, const double * sourceEnd) {
    while (source < sourceEnd) {
        *target++ = *source++;
    }
}
```

### Q3

```c
#include <stdio.h>

int findMax(const int [], int);

int main(void) {
    int arr[] = {1, 5, 2, 3, 4};
    printf("%d", findMax (arr, 5));
    return 0;
}

int findMax(const int arr[], int len) {
    int max = arr[0];
    for (int i = 0; i < len; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}
```

### Q4

```c
#include <stdio.h>

int findMax(const double [], int);

int main(void) {
    double arr[] = {1.0, 5.0, 2.0, 3.0, 4.0};
    printf("%d", findMax (arr, 5));
    return 0;
}

int findMax(const double arr[], int len) {
    int index = 0;
    double max = arr[0];
    for (int i = 0; i < len; i++) {
        if (arr[i] > max) {
            index = i;
            max = arr[i];
        }
    }
    return index;
}
```

### Q5

```c
#include <stdio.h>

int findMaxMin(const double [], int);

int main(void) {
    double arr[] = {1.0, 5.0, 2.0, 3.0, 4.0};
    printf("%d", findMaxMin (arr, 5));
    return 0;
}

int findMaxMin(const double arr[], int len) {
    double max = arr[0], min = arr[0];
    for (int i = 0; i < len; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
        if (arr[i] < min) {
            min = arr[i];
        }
    }
    return (max - min);
}
```

### Q6

```c
#include <stdio.h>

void reverseArr(double [], int);

int main(void) {
    double arr[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    reverseArr(arr, 5);
    for (int i = 0; i < 5; i++) {
        printf("%f ", arr[i]);
    }
    return 0;
}

void reverseArr(double arr[], int len) {
    double temp;
    for (int i = 0; i * 2 < len; i++) {
        temp = arr[i];
        arr[i] = arr[len-i-1];
        arr[len-i-1] = temp;
    }
}
```

### Q7

```c
#include <stdio.h>

#define ROWS 2
#define COLS 3

void copy_arr(double [], const double [], int);
void copy_ptr(double *, const double *, int);
void copy_ptrs(double *, const double *, const double *);

int main(void) {
    double source[ROWS][COLS] = {
        {1.1, 2.2, 3.3},
        {3.3, 4.4, 5.5}
    };
    double target[ROWS][COLS];
    for (int i = 0; i < ROWS; i++) {
        copy_arr(target[i], source[i], COLS);
    }
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%f ", target[i][j]);
        }
        printf("\n");
    }
    for (int i = 0; i < ROWS; i++) {
        copy_ptr((double *)(target+i), (double *)(source+i), COLS);
    }
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%f ", target[i][j]);
        }
        printf("\n");
    }
    for (int i = 0; i < ROWS; i++) {
        copy_ptrs((double *)(target+i), (double *)(source+i), (double *)(source+i)+COLS);
    }
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%f ", target[i][j]);
        }
        printf("\n");
    }
    return 0;
}

void copy_arr(double target[], const double source[], int n) {
    for (int i = 0; i < n; i++) {
        target[i] = source[i];
    }
}

void copy_ptr(double * target, const double * source, int n) {
    for (int i = 0; i < n; i++) {
        *target++ = *source++;
    }
}

void copy_ptrs(double * target, const double * source, const double * sourceEnd) {
    while (source < sourceEnd) {
        *target++ = *source++;
    }
}
```

### Q8

```c
#include <stdio.h>

void copy_arr(double [], const double [], int);
void copy_ptr(double *, const double *, int);
void copy_ptrs(double *, const double *, const double *);

int main(void) {
    double source[7] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7};
    double target[3];
    copy_arr(target, source + 2, 3);
    for (int i = 0; i < 3; i++) {
        printf("%f ", target[i]);
    }
    copy_ptr(target, source + 2, 3);
    for (int i = 0; i < 3; i++) {
        printf("%f ", target[i]);
    }
    copy_ptrs(target, source + 2, source + 5);
    for (int i = 0; i < 3; i++) {
        printf("%f ", target[i]);
    }
    return 0;
}

void copy_arr(double target[], const double source[], int n) {
    for (int i = 0; i < n; i++) {
        target[i] = source[i];
    }
}

void copy_ptr(double * target, const double * source, int n) {
    for (int i = 0; i < n; i++) {
        *target++ = *source++;
    }
}

void copy_ptrs(double * target, const double * source, const double * sourceEnd) {
    while (source < sourceEnd) {
        *target++ = *source++;
    }
}
```

### Q9

```c
#include <stdio.h>

#define ROWS 3
#define COLS 5

void copy(int, int, double [*][*], const double [*][*]);
void show(int, int, const double [*][*]);

int main(void) {
    double source[ROWS][COLS] = {
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {2.0, 3.0, 4.0, 5.0, 6.0},
        {3.0, 4.0, 5.0, 6.0, 7.0}
    };
    double target[ROWS][COLS];
    copy(ROWS, COLS, target, source);
    show(ROWS, COLS, target);
    show(ROWS, COLS, source);
    return 0;
}

void copy(int m, int n, double target[m][n], const double source[m][n]) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            target[i][j] = source[i][j];
        }
    }
}

void show(int m, int n, const double ar[m][n]) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", ar[i][j]);
        }
        printf("\n");
    }
}
```

### Q10

```c
#include <stdio.h>

#define LEN 5

void add(int m, const double [], const double [], double []);

int main(void) {
    double ar1[LEN] = {1.0, 2.0, 3.0, 4.0, 5.0},
           ar2[LEN] = {1.0, 2.0, 3.0, 4.0, 5.0},
           sum[LEN];
    add(LEN, ar1, ar2, sum);
    for (int i = 0; i < LEN; i++) {
        printf("%f ", sum[i]);
    }
    printf("\n");
    return 0;
}

void add(int m, const double ar1[m], const double ar2[m], double sum[m]) {
    for (int i = 0; i < m; i++) {
        sum[i] = ar1[i] + ar2[i];
    }
}
```

### Q11

```c
#include <stdio.h>

#define ROWS 3
#define COLS 5

void show(const int [][COLS], int);
void redouble(int [][COLS], int);

int main(void) {
    int source[ROWS][COLS] = {
        {1, 2, 3, 4, 5},
        {2, 3, 4, 5, 6},
        {3, 4, 5, 6, 7}
    };
    int target[ROWS][COLS];
    show(source, 3);
    redouble(source, 3);
    show(source, 3);
    return 0;
}

void show(const int ar[][COLS], int rows) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%d ", ar[i][j]);
        }
        printf("\n");
    }
}

void redouble(int ar[][COLS], int rows) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < COLS; j++) {
            ar[i][j] *= 2;
        }
    }
}
```

### Q12

```c
#include <stdio.h>
#define MONTHS 12
#define YEARS 5

void year(const float [][MONTHS], int);
void month(const float [][MONTHS], int);

int main(void) {
    const float rain[YEARS][MONTHS] = {
        {4.3, 4.3, 4.3, 3.0, 2.0, 1.2, 0.2, 0.2, 0.4, 2.4, 3.5, 6.6},
        {8.5, 8.2, 1.2, 1.6, 2.4, 0.0, 5.2, 0.9, 0.3, 0.9, 1.4, 7.3},
        {9.1, 8.5, 6.7, 4.3, 2.1, 0.8, 0.2, 0.2, 1.1, 2.3, 6.1, 8.4},
        {7.2, 9.9, 8.4, 3.3, 1.2, 0.8, 0.4, 0.0, 0.6, 1.7, 4.3, 6.2},
        {7.6, 5.6, 3.8, 2.8, 3.8, 0.2, 0.0, 0.0, 0.0, 1.3, 2.6, 5.2}
    };
    year(rain, YEARS);
    month(rain, YEARS);

    return 0;
}

void year(const float rain[][MONTHS], int years) {
    int year, month;
    float subtot, total;
    printf(" YEAR   RAINFALL  (inches)\n");
    for (year = 0, total = 0; year < years; year++) {
        for (month = 0, subtot = 0; month < MONTHS; month++)
            subtot += rain[year][month];
        printf("%5d %15.1f\n", 2010 + year, subtot);
        total += subtot;
    }
    printf("\nThe yearly average is %.1f inches.\n\n", total / YEARS);
}

void month(const float rain[][MONTHS], int years) {
    int year, month;
    float subtot;
    printf("MONTHLY AVERAGES:\n\n");
    printf(" Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Seq  Oct  Mov  Dec\n");
    for (month = 0; month < MONTHS; month++) {
        for (year = 0, subtot = 0; year < YEARS; year++)
            subtot += rain[year][month];
        printf("%4.1f ", subtot / YEARS);
    }
    printf("\n");
}
```

### Q13

```c
#include <stdio.h>

#define ROWS 3
#define COLS 5

void getData(double [][COLS], int);
double getAverage(const double [], int);
void getRowAverage(const double [][COLS], int, double []);
double getAllAverage(const double [][COLS], int);
double getMax(const double [][COLS], int);
void printRes(const double [][COLS], int, const double [], double, double);

int main(void) {
    double arr[ROWS][COLS],
           rowAverage[ROWS],
           allAverage,
           max;
    getData(arr, ROWS);
    getRowAverage(arr, ROWS, rowAverage);
    allAverage = getAllAverage(arr, ROWS);
    max = getMax(arr, ROWS);
    printRes(arr, ROWS, rowAverage, allAverage, max);
    return 0;
}

void getData(double arr[][COLS], int rows) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < COLS; j++) {
            scanf("%lf", &arr[i][j]);
        }
    }
}

double getAverage(const double arr[], int len) {
    double total = 0;
    for (int i = 0; i < len; i++) {
        total += arr[i];
    }
    return total / len;
}

void getRowAverage(const double arr1[][COLS], int rows, double arr2[]) {
    for (int i = 0; i < rows; i++) {
        arr2[i] = getAverage(arr1[i], COLS);
    }
}

double getAllAverage(const double arr[][COLS], int rows) {
    double total = 0,
           allAverage;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < COLS; j++) {
            total += arr[i][j];
        }
    }
    allAverage = total / (rows * COLS);
    return allAverage;
}

double getMax(const double arr[][COLS], int rows) {
    double max = arr[0][0];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < COLS; j++) {
            if (arr[i][j] > max) {
                max = arr[i][j];
            }
        }
    }
    return max;
}

void printRes(const double arr1[][COLS], int rows, const double rowAverage[], double allAverage, double max) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%f ", arr1[i][j]);
        }
        printf("\n");
    }
    for (int i = 0; i < rows; i++) {
        printf("%f\n", rowAverage[i]);
    }
    printf("%f\n", allAverage);
    printf("%f\n", max);
}
```

### Q14

```c
#include <stdio.h>

#define ROWS 3
#define COLS 5

void getData(int, int, double [*][*]);
double getAverage(int, const double []);
void getRowAverage(int, int, const double [*][*], double []);
double getAllAverage(int, int, const double [*][*]);
double getMax(int, int, const double [*][*]);
void printRes(int, int, const double [*][*], const double [], double, double);

int main(void) {
    double arr[ROWS][COLS],
           rowAverage[ROWS],
           allAverage,
           max;
    getData(ROWS, COLS, arr);
    getRowAverage(ROWS, COLS, arr, rowAverage);
    allAverage = getAllAverage(ROWS, COLS, arr);
    max = getMax(ROWS, COLS, arr);
    printRes(ROWS, COLS, arr, rowAverage, allAverage, max);
    return 0;
}

void getData(int rows, int cols, double arr[rows][cols]) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            scanf("%lf", &arr[i][j]);
        }
    }
}

double getAverage(int len, const double arr[]) {
    double total = 0;
    for (int i = 0; i < len; i++) {
        total += arr[i];
    }
    return total / len;
}

void getRowAverage(int rows, int cols, const double arr1[rows][cols], double arr2[rows]) {
    for (int i = 0; i < rows; i++) {
        arr2[i] = getAverage(cols, arr1[i]);
    }
}

double getAllAverage(int rows, int cols, const double arr[rows][cols]) {
    double total = 0,
           allAverage;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            total += arr[i][j];
        }
    }
    allAverage = total / (rows * COLS);
    return allAverage;
}

double getMax(int rows, int cols, const double arr[rows][cols]) {
    double max = arr[0][0];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (arr[i][j] > max) {
                max = arr[i][j];
            }
        }
    }
    return max;
}

void printRes(int rows, int cols, const double arr[rows][cols], const double rowAverage[], double allAverage, double max) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", arr[i][j]);
        }
        printf("\n");
    }
    for (int i = 0; i < rows; i++) {
        printf("%f\n", rowAverage[i]);
    }
    printf("%f\n", allAverage);
    printf("%f\n", max);
}
```

