# Chapter 12

### Q1

```c
#include <stdio.h>

void critic(int *);

int main(void) {
    int units;
    printf("How many pounds to a firkin of butter?\n");
    scanf("%d", &units);
    while (units != 56)
        critic(&units);
    printf("You must have looked it up!\n");

    return 0;
}

void critic(int * units) {
    printf("No luck, my friend. Try again.\n");
    scanf("%d", units);
}
```

### Q2

```c
// pel2-2a.h
void set_mode(int);
void get_info();
void show_info();
```

```c
// pel2-2a.c
#include <stdio.h>
#include "pel2-2a.h"

static int mode;
static double distance;
static double fuel;

void set_mode(int modeVal) {
    if (modeVal == 0 || modeVal == 1) {
        mode = modeVal;
    }
    else {
        printf("Invalid mode specified. More %s used.\n", mode?"1(US)":"0(metric)");
    }
}

void get_info() {
    if (mode == 0) {
        printf("Enter distance traveled in kilometers: ");
        scanf("%lf", &distance);
        printf("Enter fuel consumed in liters: ");
        scanf("%lf", &fuel);
    }
    if (mode == 1) {
        printf("Enter distance traveled in miles: ");
        scanf("%lf", &distance);
        printf("Enter fuel consumed in gallons: ");
        scanf("%lf", &fuel);
    }
}

void show_info() {
    if (mode == 0) {
        printf("Fuel consumption is %.2lf liters per 100km.\n", fuel / distance * 100);
    }
    if (mode == 1) {
        printf("Fuel consumption is %.1lf miles per gallon.\n", distance / fuel);
    }
}
```

```c
// pel2-2b.c
#include <stdio.h>
#include "pel2-2a.h"

int main(void) {
    int mode;

    printf("Enter 0 for metric mode, 1 for US mode: ");
    scanf("%d", &mode);
    while (mode >= 0) {
        set_mode(mode);
        get_info();
        show_info();
        printf("Enter 0 for metric mode, 1 for US mode: ");
        printf(" (-1 to quit): ");
        scanf("%d", &mode);
    }
    printf("Done.\n");
    return 0;
}
```

### Q3

```c
// pel2-2a.h
void set_mode(int*, int*);
void get_info(int, double *, double *);
void show_info(int, double, double);
```

```c
// pel2-2a.c
#include <stdio.h>
#include "pel2-2a.h"

void set_mode(int * modeVal, int * lmodeVal) {
    if (*modeVal == 0 || *modeVal == 1) {
        *lmodeVal = *modeVal;
    }
    else {
        *modeVal = *lmodeVal;
        printf("Invalid mode specified. More %s used.\n", (*modeVal==0)?"(0)metric":"(1)US");
    }
}

void get_info(int mode, double * distance, double * fuel) {
    if (mode == 0) {
        printf("Enter distance traveled in kilometers: ");
        scanf("%lf", distance);
        printf("Enter fuel consumed in liters: ");
        scanf("%lf", fuel);
    }
    if (mode == 1) {
        printf("Enter distance traveled in miles: ");
        scanf("%lf", distance);
        printf("Enter fuel consumed in gallons: ");
        scanf("%lf", fuel);
    }
}

void show_info(int mode, double distance, double fuel) {
    if (mode == 0) {
        printf("Fuel consumption is %.2lf liters per 100km.\n", fuel / distance * 100);
    }
    if (mode == 1) {
        printf("Fuel consumption is %.1lf miles per gallon.\n", distance / fuel);
    }
}
```

```c
// pel2-2b.c
#include <stdio.h>
#include "pel2-2a.h"

int main(void) {
    int mode, lmode = 0;
    double distance, fuel;

    printf("Enter 0 for metric mode, 1 for US mode: ");
    scanf("%d", &mode);
    while (mode >= 0) {
        set_mode(&mode, &lmode);
        get_info(mode, &distance, &fuel);
        show_info(mode, distance, fuel);
        printf("Enter 0 for metric mode, 1 for US mode: ");
        printf(" (-1 to quit): ");
        scanf("%d", &mode);
    }
    printf("Done.\n");
    return 0;
}
```

### Q4

```c
#include <stdio.h>

int fun(void);

int main(void) {
    int n;
    scanf("%d", &n);
    while (n--) {
        printf("%d\n", fun());
    }
    return 0;
}

int fun(void) {
    static int times = 0;
    times++;
    return times;
}
```

### Q5

```c
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define SIZE 100

int main(void) {
    int arr[SIZE];
    srand((unsigned int)time(0));
    for (int i = 0; i < SIZE; i++) {
        arr[i] = rand() % 10 + 1;
    }
    for (int i = 0; i < SIZE; i++) {
        if (i % 10 == 0 && i != 0)
            printf("\n");
        printf("%2d ", arr[i]);
    }
    for (int i = 0; i < SIZE - 1; i++) {
        for (int j = i + 1; j < SIZE; j++) {
            if (arr[i] < arr[j]) {
                int temp;
                temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
    }
    printf("\n\n");
    for (int i = 0; i < SIZE; i++) {
        if (i % 10 == 0 && i != 0)
            printf("\n");
        printf("%2d ", arr[i]);
    }
    return 0;
}
```

### Q6

```c
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define SIZE 1000
#define RANGE 10
#define TIMES 10

void getNums(int);

int main(void) {
    int seeds[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    for (int i = 0; i < TIMES; i++) {
        getNums(seeds[i]);
    }
    return 0;
}

void getNums(int seed) {
    int arr[SIZE],
        nums[RANGE] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    srand(seed);
    for (int i = 0; i < SIZE; i++) {
        arr[i] = rand() % 10 + 1;
        nums[arr[i]-1]++;
    }
    for (int i = 0; i < RANGE; i++) {
        printf("%4d ", nums[i]);
    }
    printf("\n");
}
```

### Q7

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void)
{
    int sets;
    int sides, dice;
    srand((unsigned int)time(0));
    printf("Enter the number of sets. enter q to stop: ");
    while (scanf("%d", &sets) == 1) {
        printf("How many sides and how many dice? ");
        scanf("%d %d", &sides, &dice);
        printf("Here are %d sets of %d %d-sided throws.\n", sets, dice, sides);
        for (int i = 0; i < sets; i++) {
            int val = 0;
            if (i % 15 == 0 && i != 0) {
                printf("\n");
            }
            for (int j = 0; j < dice; j++) {
                val += rand() % sides + 1;
            }
            printf("%d ", val);
        }
        printf("\n");
        printf("Enter the number of sets. enter q to stop: ");
    }
    return 0;
}
```

### Q8

```c
#include <stdio.h>
#include <stdlib.h>

int *make_array(int elem, int val);
void show_array(const int ar[], int n);

int main(void)
{
    int *pa;
    int size;
    int value;

    printf("Enter the number of elements: ");
    while (scanf("%d", &size) == 1 && size > 0)
    {
        printf("Enter the initialization value: \n");
        scanf("%d", &value);
        pa = make_array(size, value);
        if (pa)
        {
            show_array(pa, size);
            free(pa);
        }
        printf("Enter the number of elements (<1 to quit): ");
    }
    printf("Done.\n");

    return 0;
}

int *make_array(int elem, int val)
{
    int * pt = (int *) malloc(elem * sizeof(int));
    int i = 0;

    if (pt == NULL)
    {
        printf("Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    while (i < elem)
    {
        pt[i] = val;
        i++;
    }
    return pt;
}

void show_array(const int ar[], int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("%d ", ar[i]);
        if ((i + 1) % 8 == 0 && i+1 != n)
        {
            printf("\n");
        }
    }
    printf("\n");
}
```

### Q9

```c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define LEN 100

int main(void) {
    char **pt;
    char temp[LEN];
    int n, len;

    printf("How many words do you wish to enter? ");
    scanf("%d", &n);
    pt = (char **) malloc(n * sizeof(char *));
    if (pt == NULL) {
        printf("Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    printf("Enter %d words now: \n", n);
    for (int i = 0; i < n; i++) {
        scanf("%99s", temp);
        len = strlen(temp) + 1;
        pt[i] = (char *)malloc(len * sizeof(char));
        if (pt[i] == NULL) {
            printf("Memory allocation failed!\n");
            exit(EXIT_FAILURE);
        }
        strcpy(pt[i], temp);
    }
    printf("Here are your words:\n");
    for (int i = 0; i < n; i++) {
        puts(pt[i]);
        free(pt[i]);
    }
    free(pt);
    return 0;
}
```

