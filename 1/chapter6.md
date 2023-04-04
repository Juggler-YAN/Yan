# Chapter 6

### Q1

```c
#include <stdio.h>

#define SIZE 26

int main(void) {
    char letter[SIZE] = "abcdefghijklmnopqrstuvwxyz";
    for (int i = 0; i < SIZE; i++) {
        printf("%c ", letter[i]);
    }
    printf("\b\n");
    return 0;
}
```

### Q2

```c
#include <stdio.h>

int main(void) {
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j <= i; j++) {
            printf("$");
        }
        printf("\n");
    }
    return 0;
}
```

### Q3

```c
#include <stdio.h>

int main(void) {
    char ch = 'F';
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j <= i; j++) {
            printf("%c", ch-j);
        }
        printf("\n");
    }
    return 0;
}
```

### Q4

```c
#include <stdio.h>

int main(void) {
    int i, j;
    char ch = 'A';
    for (i = 0; i < 6; i++) {
        for (j = 0; j <= i; j++) {
            printf("%c", ch);
            ch++;
        }
        printf("\n");
    }
    return 0;
}
```

### Q5

```c
#include <stdio.h>

int main(void) {
    char ch;
    printf("Please enter a capital letter: ");
    scanf("%c", &ch);
    int rows = ch - 'A' + 1,
        cols = (ch - 'A') * 2 + 1;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < rows-i-1; j++) {
            printf(" ");
        }
        char c;
        for (c = 'A'; c < 'A'+i; c++) {
            printf("%c", c);
        }
        for (; c >= 'A'; c--) {
            printf("%c", c);
        }
        printf("\n");
    }
    return 0;
}
```

### Q6

```c
#include <stdio.h>

int main(void) {
    int limit1, limit2;
    printf("Please enter the lower and upper limits of the form respectively: ");
    scanf("%d %d", &limit1, &limit2);
    printf("              square      cube\n");
    for (int i = limit1; i <= limit2; i++) {
        printf("%10d %10d %10d\n", i, i*i, i*i*i);
    }
    return 0;
}
```

### Q7

```c
#include <stdio.h>
#include <string.h>

int main(void) {
    char s[20];
    printf("Please enter a word: ");
    scanf("%s", s);
    for (int i = strlen(s) - 1; i >= 0; i--) {
        printf("%c", s[i]);
    }
    return 0;
}
```

### Q8

```c
#include <stdio.h>
#include <math.h>

int main(void) {
    double num1, num2;
    printf("Please enter two floating point numbers: ");
    while(scanf("%lf %lf", &num1, &num2) == 2) {
        printf("%f\n", fabs(num1-num2)/(num1*num2));
        printf("Please enter two floating point numbers: ");
    }
    return 0;
}
```

### Q9

```c
#include <stdio.h>
#include <math.h>

double fun(double, double);

int main(void) {
    double num1, num2;
    printf("Please enter two floating point numbers: ");
    while(scanf("%lf %lf", &num1, &num2) == 2) {
        printf("%lf\n", fun(num1, num2));
        printf("Please enter two floating point numbers: ");
    }
    return 0;
}

double fun(double num1, double num2) {
    return fabs(num1-num2)/(num1*num2);
}
```

### Q10

```c
#include <stdio.h>

int main(void) {
    int limit1, limit2;
    printf("Enter lower and upper integer limits: ");
    scanf("%d %d", &limit1, &limit2);
    while (limit1 < limit2) {
        int i,
            sum = 0;
        for (i = limit1; i <= limit2; i++) {
            sum += i*i;
        }
        printf("The sums of the squares from %d to %d is %d\n",
               limit1*limit1, limit2*limit2, sum);
        printf("Enter next set of limits: ");
        scanf("%d %d", &limit1, &limit2);
    }
    printf("Done");
    return 0;
}
```

### Q11

```c
#include <stdio.h>

#define SIZE 8

int main(void) {
    int i, arr[SIZE];
    printf("Please enter eight integers: ");
    for (i = 0; i < SIZE; i++) {
        scanf("%d", &arr[i]);
    }
    for (i--; i >= 0; i--) {
        printf("%d ", arr[i]);
    }
    printf("\b\n");
    return 0;
}
```

### Q12

只有第二个序列是收敛的。

```c
#include <stdio.h>

int main(void) {
    int times;
    printf("Please enter the number of runs: ");
    scanf("%d", &times);
    while (times > 0) {
        int i;
        double dividend = 1.0,
               divisor = 1.0,
               sum1 = 0.0,
               sum2 = 0.0;
        int sign = 1;
        for (i = 1; i <= times; i++) {
            sum1 += dividend / divisor;
            sum2 += sign * dividend / divisor;
            divisor += 1.0;
            sign *= -1;
        }
        printf("%f\n", sum1);
        printf("%f\n", sum2);
        printf("Please enter the number of runs: ");
        scanf("%d", &times);
    }
    return 0;
}
```

### Q13

```c
#include <stdio.h>

#define SIZE 8

int main(void) {
    int i,
        j,
        val = 1,
        arr[SIZE];
    for (i = 0; i < SIZE; i++) {
        val *= 2;
        arr[i] = val;
    }
    j = 0;
    do
    {
        printf("%d\n", arr[j]);
        j++;
    } while (j < SIZE);
    return 0;
}
```

### Q14

```c
#include <stdio.h>

#define SIZE 8

int main(void) {
    double arr1[SIZE],
           arr2[SIZE];
    for (int i = 0; i < SIZE; i++) {
        scanf("%lf", &arr1[i]);
    }
    arr2[0] = arr1[0];
    for (int j = 0; j+1 < SIZE; j++) {
        arr2[j+1] = arr2[j] + arr1[j+1];
    }
    for (int i = 0; i < SIZE; i++) {
        printf("%0.2f ", arr1[i]);
    }
    printf("\b\n");
    for (int j = 0; j < SIZE; j++) {
        printf("%0.2f ", arr2[j]);
    }
    printf("\b\n");
    return 0;
}
```

### Q15

```c
#include <stdio.h>

#define SIZE 255

int main(void) {
    int i;
    char arr[SIZE];
    for (i = 0; arr[i-1] != '\n' && i < SIZE; i++) {
        scanf("%c", &arr[i]);
    }
    for (i -= 2; i >= 0; i--) {
        printf("%c", arr[i]);
    }
    return 0;
}
```

### Q16

```c
#include <stdio.h>

#define RATE1 0.1
#define RATE2 0.05

int main(void) {
    int year = 0;
    double init = 100,
           money1 = init,
           money2 = init;
    while (money1 >= money2) {
        money1 += (init * RATE1);
        money2 *= (1 + RATE2);
        year++;
    }
    printf("Year:%d, Daphne:%0.2lf, Deirdre:%0.2lf",
           year, money1, money2);
    return 0;
}
```

### Q17

```c
#include <stdio.h>

#define RATE 0.08

int main(void) {
    int year = 0;
    double val = 100;
    while (val > 0) {
        val = val * (1 + RATE) - 10;
        year++;
    }
    printf("Year:%d", year);
    return 0;
}
```

### Q18

```c
#include <stdio.h>

int main(void) {
    int val = 5,
        week = 0;
    while (val <= 150) {
        week++;
        val = (val - week) * 2;
        printf("Week%d:%d\n", week, val);
    }
    return 0;
}
```

