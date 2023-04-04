# Chapter 5

### Q1

```c
#include <stdio.h>

#define S_PER_M 60

int main(void) {
    int time;
    printf("Note: Exit when the input is not more than 0.\n");
    printf("Enter the time(min): ");
    scanf("%d", &time);
    while(time > 0) {
        printf("%d hour %d minute\n", time / S_PER_M,  time % S_PER_M);
        printf("Enter the time(min): ");
        scanf("%d", &time);
    }
    return 0;
}
```

### Q2

```c
#include <stdio.h>

int main(void) {
    int num, limit;
    printf("Enter an integer:");
    scanf("%d", &num);
    limit = num + 10;
    while(num <= limit) {
        printf("%d ", num);
        num++;
    }
    printf("\b\n");
    return 0;
}
```

### Q3

```c
#include <stdio.h>

#define S_PER_D 7

int main(void) {
    int day;
    printf("Note: Exit when enter a non positive value.\n");
    printf("Please enter the number of days (day): ");
    scanf("%d", &day);
    while(day > 0) {
        printf("%d days are %d weeks, %d days.\n", day, day / S_PER_D,  day % S_PER_D);
        printf("Please enter the number of days (day): ");
        scanf("%d", &day);
    }
    return 0;
}
```

### Q4

```c
#include <stdio.h>

#define S_PER_C 2.54
#define S_PER_I 12

int main(void) {
    float height;
    printf("Enter a height in centimeters(<=0 to quit):");
    scanf("%f", &height);
    while(height > 0) {
        printf("%0.1f cm = %d feet, %0.1f inches.\n", height, (int)(height/S_PER_C/S_PER_I),
               height/S_PER_C-(int)(height/S_PER_C/S_PER_I)*S_PER_I);
        printf("Enter a height in centimeters(<=0 to quit):");
        scanf("%f", &height);
    }
    printf("bye");
    return 0;
}
```

### Q5

```c
#include <stdio.h>

int main(void) {
    int count = 0, sum = 0;
    int days;
    printf("Please enter the number of days: ");
    scanf("%d", &days);
    while (count++ < days) {
        sum += count;
    }
    printf("sum = %d\n", sum);
    return 0;
}
```

### Q6

```c
#include <stdio.h>

int main(void) {
    int count = 0, sum = 0;
    int days;
    printf("Please enter the number of days: ");
    scanf("%d", &days);
    while (count++ < days) {
        sum += count * count;
    }
    printf("sum = %d\n", sum);
    return 0;
}
```

### Q7

```c
#include <stdio.h>

double cube(double);

int main(void) {
    double val;
    printf("Please enter the value of double type: ");
    scanf("%lf", &val);
    printf("%f", cube(val));
    return 0;
}

double cube (double val) {
    return val * val * val;
}
```

### Q8

```c
#include <stdio.h>

int main(void) {
    int dividend, divisor;
    printf("This program computes moduli.\n");
    printf("Enter a integer to serve as the second operand: ");
    scanf("%d", &divisor);
    printf("Now enter the first operand: ");
    scanf("%d", &dividend);
    while(dividend > 0) {
        printf("%d %% %d is %d.\n", dividend, divisor, dividend % divisor);
        printf("Enter next number for first operand (<=0 to quit): ");
        scanf("%d", &dividend);
    }
    printf("Done");
    return 0;
}
```

### Q9

```c
#include <stdio.h>

void Temperatures(double);

int main(void) {
    double val;
    printf("Please enter the temperature (Fahrenheit): ");
    while (scanf("%lf", &val)) {
        Temperatures(val);
        printf("Please enter the temperature (Fahrenheit): ");
    }
    printf("bye");
    return 0;
}

void Temperatures(double val) {
    const double tran1 = 5.0;
    const double tran2 = 9.0;
    const double tran3 = 32.0;
    const double tran4 = 273.16;
    printf("Fahrenheit: %0.2f\n", val);
    printf("Celsius: %0.2f\n", tran1/tran2*(val-tran3));
    printf("Kelvin: %0.2f\n", tran1/tran2*(val-tran3)+tran4);
}
```

