#include <stdio.h>

int main(void) {
    float nums;
    printf("Enter the number of cups: ");
    scanf("%f", &nums);
    printf("Pint: %f\n", nums * 0.5);
    printf("Ounce: %f\n", nums * 8);
    printf("Soupspoon: %f\n", nums * 8 * 2);
    printf("Teaspoon: %f\n", nums * 8 * 2 * 3);
    return 0;
}