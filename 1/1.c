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