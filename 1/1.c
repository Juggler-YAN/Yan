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