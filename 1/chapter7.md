# Chapter 7

### Q1

```c
#include <stdio.h>

int main(void) {
    char ch;
    int shapes = 0,
        lineBreaks = 0,
        otherChs = 0;
    while ((ch = getchar()) != '#') {
        if (ch == ' ') {
            shapes++;
        }
        else if (ch == '\n') {
            lineBreaks++;
        }
        else {
            otherChs++;
        }
    }
    printf("%d %d %d", shapes, lineBreaks, otherChs);
    return 0;
}
```

### Q2

```c
#include <stdio.h>

int main(void) {
    char ch;
    int num = 0;
    while ((ch = getchar()) != '#') {
        if (ch != '\n') {
            num++;
            printf("%c:%d ", ch, ch);
            if (num % 8 == 0) {
                printf("\n");
            }
        }
    }
    if (num % 8 != 0) {
        printf("\b\n");
    }
    return 0;
}
```

### Q3

```c
#include <stdio.h>

int main(void) {
    int num;
    int oddSum = 0,
        evenSum = 0,
        oddNum = 0,
        evenNum = 0;
    scanf("%d", &num);
    while (num != 0) {
        if (num % 2) {
            evenNum++;
            evenSum += num;
        }
        else {
            oddNum++;
            oddSum += num;
        }
        scanf("%d", &num);
    }
    printf("%d:%.2f %d:%.2f", oddNum, ((double)oddSum)/oddNum,
            evenNum, ((double)evenSum)/evenNum);
    return 0;
}
```

### Q4

```c
#include <stdio.h>

int main(void) {
    char ch;
    int num = 0;
    while ((ch = getchar()) != '#') {
        if (ch == '.') {
            putchar('!');
            num++;
        }
        else if (ch == '!') {
            putchar('!');
            putchar('!');
            num++;
        }
        else {
            putchar(ch);
        }
    }
    printf("\n%d\n", num);
    return 0;
}
```

### Q5

```c
#include <stdio.h>

int main(void) {
    char ch;
    int num = 0;
    while ((ch = getchar()) != '#') {
        switch (ch) {
            case '.': {
                putchar('!');
                num++;
                break;
            }
            case '!': {
                putchar('!');
                putchar('!');
                num++;
                break;
            }
            default: {
                putchar(ch);
                break;
            }
        }
    }
    printf("\n%d\n", num);
    return 0;
}
```

### Q6

```c
#include <stdio.h>

int main(void) {
    char ch,
         prev = ' ';
    int num = 0;
    while ((ch = getchar()) != '#') {
        if (prev == 'e' && ch == 'i') {
            num++;
        }
        prev = ch;
    }
    printf("%d\n", num);
    return 0;
}
```

### Q7

```c
#include <stdio.h>

#define SALARY 10
#define WORKTIME 40
#define OVERWORKTIMERATE 1.5
#define RATE1 0.15
#define RATE2 0.2
#define RATE3 0.25
#define BASE1 300
#define BASE2 150

int main(void) {
    double time;
    printf("Please Enter the number of working hours: ");
    scanf("%lf", &time);
    if (time > WORKTIME) {
        time = WORKTIME + (time - WORKTIME) * OVERWORKTIMERATE;
    }
    double salary;
    salary = time * SALARY;
    printf("%.2f\n", salary);
    double money, tax;
    if (salary <= BASE1) {
        tax = salary * RATE1;
    }
    else if (salary <= BASE1 + BASE2) {
        tax = BASE1 * RATE1 + (salary - BASE1) * RATE2;
    }
    else {
        tax = BASE1 * RATE1 + BASE2 * RATE2 + (salary - (BASE1 + BASE2)) * RATE3;
    }
    money = salary - tax;
    printf("%.2f\n", tax);
    printf("%.2f\n", money);
    return 0;
}
```

### Q8

```c
#include <stdio.h>

#define OVERWORKTIME 40
#define OVERWORKTIMERATE 1.5
#define RATE1 0.15
#define RATE2 0.2
#define RATE3 0.25
#define BASE1 300
#define BASE2 150

void menu(void);
void calMoney(double);

int main(void) {
    int choice;
    menu();
    scanf("%d", &choice);
    while (choice != 5) {
        switch (choice) {
            case 1: {
                calMoney(8.75);
                break;
            }
            case 2: {
                calMoney(9.33);
                break;
            }
            case 3: {
                calMoney(10.00);
                break;
            }
            case 4: {
                calMoney(11.20);
                break;
            }
            default: {
                printf("Input Error!\n");
            }
        }
        menu();
        scanf("%d", &choice);
    }
    return 0;
}

void menu(void) {
    printf("*****************************************************************\n");
    printf("Enter the number corresponding to the desired pay rate or action:\n");
    printf("1) $8.75/hr                       2) $9.33/hr\n");
    printf("3) $10.00/hr                      4) $11.20/hr\n");
    printf("5) quit\n");
    printf("*****************************************************************\n");
}

void calMoney(double hourlyWage) {
    double time;
    double totalWage, tax, realWage;
    printf("Please enter the number of working hours:\n");
    scanf("%lf", &time);
    if (time > OVERWORKTIME) {
        time = (time - OVERWORKTIME) * OVERWORKTIMERATE + OVERWORKTIME;
    }
    totalWage = hourlyWage * time;
    if (totalWage <  BASE1) {
        tax = totalWage * RATE1;
    }
    else if (totalWage < BASE1 + BASE2) {
        tax = (totalWage - BASE1) * RATE2 + BASE1 * RATE1;
    }
    else {
        tax = (totalWage - (BASE1 + BASE2)) * RATE3 + BASE2 * RATE2 + BASE1 * RATE1;
    }
    realWage = totalWage - tax;
    printf("totalWage:%.2f\n", totalWage);
    printf("tax:%.2f\n", tax);
    printf("realWage:%.2f\n", realWage);
}
```

### Q9

```c
#include <stdio.h>

int main(void) {
    int num;
    scanf("%d", &num);
    for (int val = 2; val <= num; val++) {
        int isPrime = 1;
        for (int i = 2; i * i <= val; i++) {
            if (!(val % i)) {
                isPrime = 0;
                break;
            }
        }
        if (isPrime) {
            printf("%d ", val);
        }
    }
    return 0;
}
```

### Q10

```c
#include <stdio.h>

#define RATE1 0.15
#define RATE2 0.28
#define BASE1 17850.00
#define BASE2 23900.00
#define BASE3 29750.00
#define BASE4 14875.00

void menu(void);
void calTax(double);

int main(void) {
    int choice;
    menu();
    scanf("%d", &choice);
    while (choice != 5) {
        switch (choice) {
            case 1: {
                calTax(BASE1);
                break;
            }
            case 2: {
                calTax(BASE2);
                break;
            }
            case 3: {
                calTax(BASE3);
                break;
            }
            case 4: {
                calTax(BASE4);
                break;
            }
            default: {
                printf("Input Error!\n");
            }
        }
        menu();
        scanf("%d", &choice);
    }
    return 0;
}

void menu(void) {
    printf("*****************************************************************\n");
    printf("选择类别:\n");
    printf("1)单身                       2)户主\n");
    printf("3)已婚，共有                 4)已婚，离异\n");
    printf("5)quit\n");
    printf("*****************************************************************\n");
}

void calTax(double BASE) {
    double wage, tax;
    printf("Please enter your salary: \n");
    scanf("%lf", &wage);
    printf("%.2f\n", wage);
    if (wage < BASE) {
        tax = wage * RATE1;
    }
    else {
        tax = BASE * RATE1 + (wage - BASE) * RATE2;
    }
    printf("tax:%.2f\n", tax);
}
```

### Q11

```c
#include <stdio.h>

#define PRICE1 2.05
#define PRICE2 1.15
#define PRICE3 1.09
#define PRICEVAL 100
#define PRICERATE 0.05
#define BASE1 5
#define BASE2 20
#define MONEY1 6.5
#define MONEY2 14
#define MONEY3 0.5

void menu(void);
void sumPrice(double, double, double);

int main(void) {
    char choice;
    menu();
    scanf("%c", &choice);
    double a = 0.0,
           b = 0.0,
           c = 0.0,
           tem;
    while (choice != 'q') {
        switch (choice) {
            case 'a': {
                printf("Please select the purchase weight: ");
                scanf("%lf", &tem);
                a += tem;
                break;
            }
            case 'b': {
                printf("Please select the purchase weight: ");
                scanf("%lf", &tem);
                b += tem;
                break;
            }
            case 'c': {
                printf("Please select the purchase weight: ");
                scanf("%lf", &tem);
                c += tem;
                break;
            }
            default: {
                printf("Input Error!\n");
            }
        }
        menu();
        while (getchar() != '\n') {
            continue;
        }
        scanf("%c", &choice);
    }
    sumPrice(a, b, c);
    return 0;
}

void menu(void) {
    printf("*****************************************************************\n");
    printf("Please select the type of vegetables: \n");
    printf("a)Artichoke                   b)Beet\n");
    printf("c)Carrot                      q)quit\n");
    printf("*****************************************************************\n");
}

void sumPrice(double a, double b, double c) {
    printf("Artichoke: %.2f $/lb\n", PRICE1);
    printf("Beet: %.2f $/lb\n", PRICE2);
    printf("Carrot: %.2f $/lb\n", PRICE3);
    printf("Artichoke: %.2f lb\n", a);
    printf("Beet: %.2f lb\n", b);
    printf("Carrot: %.2f lb\n", c);
    double allPrice, discount;
    allPrice = a * PRICE1 + b * PRICE2 + c * PRICE3;
    printf("Total order cost: %.2f\n", allPrice);
    if (allPrice > PRICEVAL) {
        discount = allPrice * PRICERATE;
        printf("Discount: %.2f\n", discount);
        allPrice *= (1 - PRICERATE);
        printf("Total order cost(After discount): %.2f\n", allPrice);
    }
    double otherPrice;
    if (a + b + c == 0) {
        otherPrice = 0;
    }
    else if (a + b + c < BASE1) {
        otherPrice = MONEY1;
    }
    else if (a + b + c < BASE2) {
        otherPrice = MONEY2;
    }
    else {
        otherPrice = ((a + b + c) - BASE2) * MONEY3 + MONEY2;
    }
    printf("Freight and packaging: %.2f\n", otherPrice);
    double sumPrice;
    if (allPrice != 0) {
        sumPrice = allPrice + otherPrice;
    }
    printf("Total cost: %.2f\n", sumPrice);
}
```

