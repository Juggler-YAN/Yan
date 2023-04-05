# Chapter 8

### Q1

```c
#include <stdio.h>

int main(void) {
    int num = 0;
    while (getchar() != EOF) {
        num++;
    }
    printf("%d", num);
    return 0;
}
```

### Q2

生成测试数据

```c
#include <stdio.h>

int main(void) {
    FILE* fp = fopen("./test.txt" , "w");
	for (int i = 0; i <= 127; i++)
	{
        if (i == 26) continue;
        fprintf(fp, "%c", i);
	}
	fclose(fp);
    return 0;
}
```

```c
#include <stdio.h>

int main(void) {
    int c,
        num = 0;
    while ((c = getchar()) != EOF) {
        if (c == '\n') {
            printf("\\n:%d ", c);
        }
        else if (c == '\t') {
            printf("\\t:%d ", c);
        }
        else if (c < ' ') {
            printf("^%c:%d ", c+64, c);
        }
        else if (c == 127) {
            printf("^?:%d\n", c);
        }
        else {
            printf("%c:%d ", c, c);
        }
        num++;
        if (!(num % 10)) {
            printf("\n");
        }
    }
    return 0;
}
```

### Q3

```c
#include <stdio.h>
#include <ctype.h>

int main(void) {
    int c,
        numL = 0,
        numU = 0;
    while ((c = getchar()) != EOF) {
        if (islower(c)) {
            numL++;
        }
        if (isupper(c)) {
            numU++;
        }
    }
    printf("L:%d;U:%d", numL, numU);
    return 0;
}
```

### Q4

```c
#include <stdio.h>
#include <ctype.h>
#include <stdbool.h>

int main(void) {
    int c,
        wordsNum = 0,   // 单词数
        allNum = 0;     // 所有单词字母总数
    bool inword = false;
    while ((c = getchar()) != EOF) {
        if (ispunct(c)) {
            continue;
        }
        if (isalpha(c)) {
            allNum++;
        }
        if (!isspace(c) && !inword) {
            inword = true;
            wordsNum++;
        }
        if (isspace(c) && inword) {
            inword = false;
        }
    }
    printf("%f", (double)allNum/wordsNum);
    return 0;
}
```

### Q5

```c
#include <stdio.h>

int main(void) {
    int c,
        min = 0,
        max = 100;
    printf("%d\n", (min + max) / 2);
    printf("Response \"g\" for \"greater\", \"s\" for "
           " \"smaller\" and \"r\" for \"right\"\n");
    while ((c = getchar()) != 'r') {
        if (c == 'g') {
            max = (max + min) / 2;
        }
        if (c == 's') {
            min = (max + min) / 2;
        }
        printf("%d-%d\n", min, max);
        printf("%d\n", (min + max) / 2);
        while(getchar() != '\n') continue;
    }
    return 0;
}
```

### Q6

```c
#include <stdio.h>
#include <ctype.h>

char get_first(void);

int main(void) {
    putchar(get_first());
    return 0;
}

char get_first(void) {
    int ch;
    while (isspace(ch = getchar())) {
        continue;
    }
    return ch;
}
```

### Q7

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
    char choice;
    menu();
    scanf("%c", &choice);
    while (choice != 'q') {
        switch (choice) {
            case 'a': {
                calMoney(8.75);
                break;
            }
            case 'b': {
                calMoney(9.33);
                break;
            }
            case 'c': {
                calMoney(10.00);
                break;
            }
            case 'd': {
                calMoney(11.20);
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
    return 0;
}

void menu(void) {
    printf("*****************************************************************\n");
    printf("Enter the number corresponding to the desired pay rate or action:\n");
    printf("a) $8.75/hr                       b) $9.33/hr\n");
    printf("c) $10.00/hr                      d) $11.20/hr\n");
    printf("q) quit\n");
    printf("*****************************************************************\n");
}

void calMoney(double hourlyWage) {
    int time;
    double totalWage, tax, realWage;
    printf("Please enter the number of working hours:\n");
    scanf("%d", &time);
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

### Q8

```c
#include <stdio.h>

void menu(void);
float getNum(void);
void choose(char, float, float);

int main(void) {
    char ch;
    menu();
    scanf("%c", &ch);
    while (ch != 'q') {
        float num1, num2;
        printf("Enter first number:");
        num1 = getNum();
        printf("Enter Second number:");
        num2 = getNum();
        while (ch == 'd' && num2 == 0.0) {
            printf("Enter a number other than 0:");
            num2 = getNum();
        }
        choose(ch, num1, num2);
        menu();
        while (getchar() != '\n') {
            continue;
        }
        scanf("%c", &ch);
    }
    return 0;
}

void menu(void) {
    printf("************************************************************\n");
    printf("Enter the operation of your choice\n");
    printf("a) add                       b) subtract\n");
    printf("c) multiply                  d) divide\n");
    printf("q) quit\n");
    printf("************************************************************\n");
}

float getNum(void) {
    float num;
    char error[10];
    while (!(scanf("%f", &num))) {
        scanf("%s", error);
        printf("%s is not a number.\n", error);
        printf("Please enter a number, such as 2.5, -1.78E8, or 3:");
    }
    return num;
}

void choose(char ch, float num1, float num2) {
    switch (ch) {
        case 'a': {
            printf("%.2f + %.2f = %.2f\n", num1, num2, num1 + num2);
            break;
        }
        case 's': {
            printf("%.2f - %.2f = %.2f\n", num1, num2, num1 - num2);
            break;
        }
        case 'm': {
            printf("%.2f * %.2f = %.2f\n", num1, num2, num1 * num2);
            break;
        }
        case 'd': {
            printf("%.2f / %.2f = %.2f\n", num1, num2, num1 / num2);
            break;
        }
        default: {
            printf("Input Error!\n");
        }
    }
}
```

