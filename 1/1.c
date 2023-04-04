#include <stdio.h>

#define transformA 3.785
#define transformB 1.609

int main(void) {
    float distance, gasoline;
    printf("Enter the distance(miles): ");
    scanf("%f", &distance);
    printf("Enter gasoline consumption(gallons)): ");
    scanf("%f", &gasoline);
    printf("%0.1f mpg", distance / gasoline);
    printf("\n");
    printf("%0.1f L/100km", (gasoline * transformA) / (distance * transformB / 100));
    return 0;
}