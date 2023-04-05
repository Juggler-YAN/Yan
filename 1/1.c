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