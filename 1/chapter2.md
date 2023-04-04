# Chapter 2

### Q1

```c
#include <stdio.h>

int main(void) {
    printf("Gustav Mahler\n");
    printf("Gustav\nMahler\n");
    printf("Gustav ");
    printf("Mahler");
    return 0;
}
```

### Q2

```c
#include <stdio.h>

int main(void) {
    printf("Name:XXX\n");
    printf("Address:XXXXXXXXXX");
    return 0;
}
```

### Q3

```c
#include <stdio.h>

int main(void) {
    int years;
    printf("Enter your ages:");
    scanf("%d", &years);
    printf("Age:%d -> Days:%d", years, years*365);
    return 0;
}
```

### Q4

```c
#include <stdio.h>

void jolly(void);
void deny(void);

int main(void) {
    jolly();
    printf("\n");
    jolly();
    printf("\n");
    jolly();
    printf("\n");
    deny();
    return 0;
}

void jolly(void) {
    printf("For he's s jolly good fellow!");
}

void deny(void) {
    printf("Which no body can deny!");
}
```

### Q5

```c
#include <stdio.h>

void br(void);
void ic(void);

int main(void) {
    br();
    printf(", ");
    ic();
    printf("\n");
    ic();
    printf(",\n");
    br();
    return 0;
}

void br(void) {
    printf("Brazil, Russia");
}

void ic(void) {
    printf("India, China");
}
```

### Q6

```c
#include <stdio.h>

int main(void) {
    int toes = 10;
    printf("toes: %d\n", toes);
    printf("Twice as much as toes: %d\n", toes * 2);
    printf("Square of toes: %d", toes * toes);
    return 0;
}
```

### Q7

```c
#include <stdio.h>

void smile(void);

int main(void) {
    smile();
    smile();
    smile();
    printf("\n");
    smile();
    smile();
    printf("\n");
    smile();
    return 0;
}

void smile(void) {
    printf("Smile!");
}
```

### Q8

```c
#include <stdio.h>

void one(void);
void two(void);

int main(void) {
    printf("starting now:\n");
    one();
    printf("done!");
    return 0;
}

void one(void) {
    printf("one\n");
    two();
    printf("three\n");
}

void two(void) {
    printf("two\n");
}
```