# Chapter 3

### Q1

整数溢出会从另一个端点开始继续计算，浮点数上溢会返回INF，浮点数下溢会损失浮点数精度。

```c
#include <stdio.h>

int main(void) {
    // 整数溢出
    int a = 2147483647;
    printf("%d\n", a);
    a += 1;
    printf("%d\n", a);
    a -= 1;
    printf("%d\n", a);
    // 浮点数上溢
    float b = 3.4e38f;
    printf("%e\n", b);
    b *= 10.0f;
    printf("%e\n", b);
    // 浮点数下溢
    float c = 0.1234e-38f;
    printf("%e\n", c);
    c /= 10.0f;
    printf("%e\n", c);
    return 0;
}
```

### Q2

```c
#include <stdio.h>

int main(void) {
    int c;
    printf("Enter ASCII value:");
    scanf("%d", &c);
    printf("%c", c);
    return 0;
}
```

### Q3

```c
#include <stdio.h>

int main(void) {
    printf("\aStartled by the sudden sound, Sally shouted,\n"
           "\"By the Great Pumpkin, what was that!\"");
    return 0;
}
```

### Q4

```c
#include <stdio.h>

int main(void) {
    float f;
    printf("Enter a floating-point value: ");
    scanf("%f", &f);
    printf("fixed-point notation: %f\n", f);
    printf("exponential notation: %e\n", f);
    printf("p notation: %a", f);
    return 0;
}
```

### Q5

```c
#include <stdio.h>

int main(void) {
    int age;
    printf("Enter your age: ");
    scanf("%d", &age);
    printf("%e", age * 3.156e7);
    return 0;
}
```

### Q6

```c
#include <stdio.h>

int main(void) {
    float num;
    printf("Enter: ");
    scanf("%f", &num);
    printf("%e", num * 950 / 3e-23);
    return 0;
}
```

### Q7

```c
#include <stdio.h>

int main(void) {
    float height;
    printf("Enter your height (inches): ");
    scanf("%f", &height);
    printf("%fcm", height * 2.54);
    return 0;
}
```

### Q8

因为1杯等于0.5品脱，所以浮点类型会更合适

```c
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
```

