# Chapter 15

### Q1

```c
#include <stdio.h>
#include <string.h>

int transform(char *);

int main(void) {
    int res;
    char * pbin = "01001001";
    res = transform(pbin);
    printf("%d", res);
	return 0;
}

int transform(char * s) {
    int num = 0 ;
	for (int i = 0; s[i]; i++) {
		num <<= 1 ;
		num |= (s[i]-'0');
	}
	return num;
}
```

### Q2

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LEN 33

unsigned decToBin(char *);
void binToDec(unsigned);

int main(int argc, char * argv[]) {
    if (argc != 3) {
        printf("Please enter the correct number of parameters.");
        exit(0);
    }
    binToDec(~decToBin(argv[1]));
    binToDec(~decToBin(argv[2]));
    binToDec(decToBin(argv[1])&decToBin(argv[2]));
    binToDec(decToBin(argv[1])|decToBin(argv[2]));
    binToDec(decToBin(argv[1])^decToBin(argv[2]));
	return 0;
}

unsigned decToBin(char * s) {
	unsigned num = 0 ;
	for (int i = 0; s[i]; i++) {
		num <<= 1 ;
		num |= (s[i]-'0');
	}
	return num;
}

void binToDec(unsigned num) {
    char s[LEN];
    int i = 0;
    while (num) {
        s[i] = num % 2 + '0';
        num /= 2;
        i++;
    }
    s[i] = '\0';
    int len = strlen(s);
    for (int j = 0; j < len-j-1; j++) {
        char temp;
        temp = s[j];
        s[j] = s[len-j-1];
        s[len-j-1] = temp;
    }
    printf("%s\n", s);
}
```

### Q3

```c
#include <stdio.h>

void sum(int);

int main() {
    sum(7);
    return 0;
}

void sum(int num) {
    int count = 0;
    while (num) {
        if (num % 2) count++;
        num /= 2;
    }
    printf("%d", count);
}
```

### Q4

```c
#include <stdio.h>

int find(int, int);

int main() {
    printf("%d", find(17,1));
    printf("%d", find(17,2));
    printf("%d", find(17,3));
    printf("%d", find(17,4));
    printf("%d", find(17,5));
    printf("%d", find(17,6));
    printf("%d", find(17,7));
    printf("%d", find(17,8));
    return 0;
}

int find(int num, int pos) {
    for (int i = 0; i < pos-1; i++) {
        num >>= 1;
    }
    return num & 01;
}
```

### Q5

```c
#include <stdio.h>

#define LEN 32

unsigned rotate_l_one(unsigned);
unsigned rotate_l(unsigned, int);

int main(void) {
    printf("%u\n", rotate_l(12, 7));
    printf("%u\n", rotate_l(12, 31));
    return 0;
}

unsigned rotate_l_one(unsigned num) {
    unsigned temp = num >> (LEN - 1);
    num = (num << 1) + temp;
    return num;
}

unsigned rotate_l(unsigned num, int times) {
    for (int i = 0; i < times; i++) {
        num = rotate_l_one(num);
    }
    return num;
}
```

### Q6

```c
#include <stdio.h>

#define left 0
#define center 1
#define right 2
#define on 1
#define off 0

typedef struct wordInfo {
    unsigned ID        :8;
    unsigned           :1;
    unsigned SIZE      :7;
    unsigned           :2;
    unsigned ALIGNMENT :2;
    unsigned           :1;
    unsigned B         :1;
    unsigned I         :1;
    unsigned U         :1;
} wordInfo;

void menu();
void print(wordInfo *);
void changeFont(wordInfo *);
void changeSize(wordInfo *);
void changeAlignment(wordInfo *);
void toggleBold(wordInfo *);
void toggleItalic(wordInfo *);
void toggleUnderline(wordInfo *);

int main(void) {
    wordInfo word = {1, 12, left, off, off, off};
    print(&word);
    char ch;
    menu();
    while ((ch = getchar()) != 'q') {
        switch(ch) {
            case 'f': {
                changeFont(&word);
                print(&word);
                break;
            }
            case 's': {
                changeSize(&word);
                print(&word);
                break;
            }
            case 'a': {
                changeAlignment(&word);
                print(&word);
                break;
            }
            case 'b': {
                toggleBold(&word);
                print(&word);
                break;
            }
            case 'i': {
                toggleItalic(&word);
                print(&word);
                break;
            }
            case 'u': {
                toggleUnderline(&word);
                print(&word);
                break;
            }
            default: {
                printf("Enter \'f\', \'s\', \'a\', "
                       "\'b\', \'i\', \'u\' or  \'q\'.");
            }
        }
        menu();
        while (getchar() != '\n') ;
    }
    puts("Bye!");
    return 0;
}

void menu() {
    printf("\nf)change font\ts)change size\ta)change alignment\n");
    printf("b)toggle bold\ti)toggle italic\tu)toggle underline\n");
    printf("q)quit\n");
}

void print(wordInfo * p) {
    printf("ID\tSIZE\tALIGNMENT\tB\tI\tU\n");
    printf("%d\t", p->ID);
    printf("%d\t", p->SIZE);
    switch(p->ALIGNMENT){
        case left:
            printf("left\t\t");
            break;
        case center:
            printf("center\t\t");
            break;
        case right:
            printf("right\t\t");
            break;
    }
    printf("%s\t", p->B==0?"off":"on");
    printf("%s\t", p->I==0?"off":"on");
    printf("%s\t", p->U==0?"off":"on");
}

void changeFont(wordInfo * p) {
    printf("Enter font ID:");
    int temp;
    scanf("%d", &temp);
    p->ID = temp;
}

void changeSize(wordInfo * p) {
    printf("Enter font size(0-127):");
    int temp;
    scanf("%d", &temp);
    p->SIZE = temp;
}

void changeAlignment(wordInfo * p) {
    printf("Select alignment:\n");
    printf("l)left\tc)center\tr)right\n");
    while (getchar() != '\n') ;
    switch(getchar()) {
        case 'l': {
            p->ALIGNMENT = 0;
            break;
        }
        case 'c': {
            p->ALIGNMENT = 1;
            break;
        }
        case 'r': {
            p->ALIGNMENT = 2;
            break;
        }
    }
}

void toggleBold(wordInfo * p) {
    (p->B)? (p->B=0): (p->B=1);
}

void toggleItalic(wordInfo * p) {
    (p->I)? (p->I=0): (p->I=1);
}

void toggleUnderline(wordInfo * p) {
    (p->U)? (p->U=0): (p->U=1);
}
```

### Q7

```c
#include <stdio.h>

#define FONT_ID 0x010000
#define ID_MASK 0xFF0000
#define FONT_SIZE 0x0C00
#define SIZE_MASK 0xFE00
#define LEFT 0x00
#define CENTER 0x10
#define RIGHT 0x20
#define POS_MASK 0x30
#define BOLD 0x4
#define ITALIC 0x2
#define UNDERLINE 0x1

void menu();
void print(unsigned long);
void changeFont(unsigned long *);
void changeSize(unsigned long *);
void changeAlignment(unsigned long *);
void toggleBold(unsigned long *);
void toggleItalic(unsigned long *);
void toggleUnderline(unsigned long *);

int main(void) {
    unsigned long word = FONT_ID | FONT_SIZE | LEFT;
    print(word);
    char ch;
    menu();
    while ((ch = getchar()) != 'q') {
        switch(ch) {
            case 'f': {
                changeFont(&word);
                print(word);
                break;
            }
            case 's': {
                changeSize(&word);
                print(word);
                break;
            }
            case 'a': {
                changeAlignment(&word);
                print(word);
                break;
            }
            case 'b': {
                toggleBold(&word);
                print(word);
                break;
            }
            case 'i': {
                toggleItalic(&word);
                print(word);
                break;
            }
            case 'u': {
                toggleUnderline(&word);
                print(word);
                break;
            }
            default: {
                printf("Enter \'f\', \'s\', \'a\', "
                       "\'b\', \'i\', \'u\' or  \'q\'.");
            }
        }
        menu();
        while (getchar() != '\n') ;
    }
    puts("Bye!");
    return 0;
}

void menu() {
    printf("\nf)change font\ts)change size\ta)change alignment\n");
    printf("b)toggle bold\ti)toggle italic\tu)toggle underline\n");
    printf("q)quit\n");
}

void print(unsigned long word) {
    printf("ID\tSIZE\tALIGNMENT\tB\tI\tU\n");
    printf("%ld\t", (word&ID_MASK)>>16);
    printf("%ld\t", (word&SIZE_MASK)>>8);
    switch(word&POS_MASK){
        case LEFT:
            printf("left\t\t");
            break;
        case CENTER:
            printf("center\t\t");
            break;
        case RIGHT:
            printf("right\t\t");
            break;
    }
    printf("%s\t", (word&BOLD)>>2?"on":"off");
    printf("%s\t", (word&ITALIC)>>1?"on":"off");
    printf("%s\t", (word&UNDERLINE)?"on":"off");
}

void changeFont(unsigned long * p) {
    *p &= ~ID_MASK;
    printf("Enter font ID:");
    int temp;
    scanf("%d", &temp);
    *p |= temp<<16;
}

void changeSize(unsigned long * p) {
    *p &= ~SIZE_MASK;
    printf("Enter font size(0-127):");
    int temp;
    scanf("%d", &temp);
    *p |= temp<<8;
}

void changeAlignment(unsigned long * p) {
    *p &= ~POS_MASK;
    printf("Select alignment:\n");
    printf("l)left\tc)center\tr)right\n");
    while (getchar() != '\n') ;
    switch(getchar()) {
        case 'l': {
            *p |= LEFT;
            break;
        }
        case 'c': {
            *p |= CENTER;
            break;
        }
        case 'r': {
            *p |= RIGHT;
            break;
        }
    }
}

void toggleBold(unsigned long * p) {
    ((*p&BOLD)>>2)? (*p&=~BOLD): (*p|=BOLD);
}

void toggleItalic(unsigned long * p) {
    ((*p&ITALIC)>>1)? (*p&=~ITALIC): (*p|=ITALIC);
}

void toggleUnderline(unsigned long * p) {
    (*p&UNDERLINE)? (*p&=~UNDERLINE): (*p|=UNDERLINE);
}
```

