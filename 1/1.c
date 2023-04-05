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