# Chapter 4

### Q1

```c
#include <stdio.h>

int main(void) {
    char name[20], surname[20];
    printf("Enter your name: ");
    scanf("%s", name);
    printf("Enter your surname: ");
    scanf("%s", surname);
    printf("%s,%s", name, surname);
    return 0;
}
```

### Q2

```c
#include <stdio.h>
#include <string.h>

int main(void) {
    char name[20], surname[20];
    printf("Enter your name: ");
    scanf("%s", name);
    printf("Enter your surname: ");
    scanf("%s", surname);
    printf("\"%s\",\"%s\"\n", name, surname);
    printf("\"%20s\",\"%20s\"\n", name, surname);
    printf("\"%-20s\",\"%-20s\"\n", name, surname);
    printf("%*s,%*s\n", (int)strlen(name)+3, name, (int)strlen(surname)+3, surname);
    return 0;
} 
```

### Q3

```c
#include <stdio.h>

int main(void) {
    float f;
    scanf("%f", &f);
    printf("%f %e", f, f);
    return 0;
}
```

### Q4

```c
#include <stdio.h>

int main(void) {
    char name[20];
    float height;
    printf("Enter your name:");
    scanf("%s", name);
    printf("Enter your height(inches):");
    scanf("%f", &height);
    printf("%s, you are %f feet tall", name, height / 12);
    return 0;
} 
```

```c
#include <stdio.h>

int main(void) {
    char name[20];
    float height;
    printf("Enter your name:");
    scanf("%s", name);
    printf("Enter your height(cm):");
    scanf("%f", &height);
    printf("%s, you are %0.2f meter tall", name, height / 100);
    return 0;
}
```

### Q5

```c
#include <stdio.h>

int main(void) {
    float speed, size;
    printf("Enter download speed (Mb/s):");
    scanf("%f", &speed);
    printf("Enter file size (MB):");
    scanf("%f", &size);
    printf("At %.2f megabits per second, a file of %.2f megabytes\n",
           speed , size);
    printf("downloads in %.2f seconds.", size * 8 / speed);
    return 0;
}
```

### Q6

```c
#include <stdio.h>
#include <string.h>

int main(void) {
    char name[20], surname[20];
    printf("Enter your name: ");
    scanf("%s", name);
    printf("Enter your surname: ");
    scanf("%s", surname);
    printf("%s %s", name, surname);
    printf("\n");
    printf("%*zd %*zd", (int)strlen(name), strlen(name), (int)strlen(surname), strlen(surname));
    return 0;
} 
```

```c
#include <stdio.h>
#include <string.h>

int main(void) {
    char name[20], surname[20];
    printf("Enter your name: ");
    scanf("%s", name);
    printf("Enter your surname: ");
    scanf("%s", surname);
    printf("%s %s", name, surname);
    printf("\n");
    printf("%-*zd %-*zd", (int)strlen(name), strlen(name), (int)strlen(surname), strlen(surname));
    return 0;
} 
```

### Q7

精度小是一致的，精度大则不一致。

```c
#include <stdio.h>
#include <float.h>

int main(void) {
    double a = 1.0 / 3.0;
    float b = 1.0 / 3.0;
    printf("%d\n", FLT_DIG);
    printf("%d\n", DBL_DIG);
    printf("%0.6f\n", a);
    printf("%0.6f\n", b);
    printf("%0.12f\n", a);
    printf("%0.12f\n", b);
    printf("%0.16f\n", a);
    printf("%0.16f\n", b);
    return 0;
}
```

### Q8

```c
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
```

