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