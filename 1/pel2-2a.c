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