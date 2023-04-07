# Chapter 14

### Q1

```c
#include <stdio.h>
#include <string.h>

#define SIZE 12
#define LEN 10

struct month {
    char name[10];
    char abbr[4];
    int days;
    int no;
};

const struct month months[SIZE] = {
    {"January" , "jan" , 31 , 1},
    {"February" , "feb" , 28 , 2},
    {"March" , "mar" , 31 , 3},
    {"April" , "apr" , 30 , 4},
    {"May" , "may" , 31 , 5},
    {"June" , "jun" , 30 , 6},
    {"July" , "jul" , 31 , 7},
    {"August" , "aug" , 31 , 8},
    {"September" , "sep" , 30 , 9},
    {"October" , "oct" , 31 , 10},
    {"November" , "nov" , 30 , 11},
    {"December" , "dec" , 31 , 12}
};

int days(char *);

int main(void) {
    char monthName[LEN];
    printf("Enter a name of a month(Ctrl+Z to quit) :");
    scanf("%s", monthName);
    printf("%d", days(monthName));
    return 0;
}

int days(char * s) {
    int index, total;
    for (index = 0, total = 0; index < SIZE; index++) {
        total += months[index].days;
        if (!strcmp(s, months[index].abbr)) {
            break;
        }
    }
    if (index == SIZE) {
        return -1;
    }
    return total;
}
```

### Q2

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 12
#define LEN 10

struct month {
    char name[10];
    char abbr[4];
    int days;
    int no;
};

int days(int *, char *, int *, struct month *);

int main(void) {
    struct month months[SIZE] = {
        {"January" , "jan" , 31 , 1},
		{"February" , "feb" , 28 , 2},
		{"March" , "mar" , 31 , 3},
		{"April" , "apr" , 30 , 4},
		{"May" , "may" , 31 , 5},
		{"June" , "jun" , 30 , 6},
		{"July" , "jul" , 31 , 7},
		{"August" , "aug" , 31 , 8},
		{"September" , "sep" , 30 , 9},
		{"October" , "oct" , 31 , 10},
		{"November" , "nov" , 30 , 11},
		{"December" , "dec" , 31 , 12}
    };
    int year, day;
    char month[LEN];
    scanf("%d %s %d", &year, month, &day);
    printf("%d", days(&year, month, &day, months));
    return 0;
}

int days(int * year, char * month, int * day, struct month * months) {
    int days = 0;
    if (*year % 4 == 0 && *year % 100 != 0 || *year % 400 == 0) {
        (months+1)->days = 29;
    }
    for (int i = 0; i < SIZE; i++) {
        if (strcmp((months+i)->name, month) == 0 || strcmp((months+i)->abbr, month) == 0 || (months+i)->no == atoi(month)) {
            break;
        }
        days += (months+i)->days;
    }
    days += *day;
    return days;
}
```

### Q3

```c
#include <stdio.h>
#include <string.h>

#define MAXTITL 40
#define MAXAUTL 40
#define MAXBKS 100

struct book {
	char title[MAXTITL];
	char author[MAXAUTL];
	float value;
};

void sort_title(struct book * [], int n);
void sort_value(struct book * [], int n);
char * s_gets(char * st, int n);

int main(void) {
	struct book library[MAXBKS];
    struct book * books[MAXBKS];
	int count = 0;
	int index;

	printf("Please enter the book title.\n");
	printf("Press [enter] at the start of a line to stop.\n");
	while (count < MAXBKS && s_gets(library[count].title , MAXTITL) != NULL
			&& library[count].title[0] != '\0') {
		printf("Now enter the author.\n");
		s_gets(library[count].author , MAXAUTL);
		printf("Now enter the value.\n");
		scanf("%f", &library[count].value);
        books[count] = &library[count];
        count++;
		while (getchar() != '\n') {
            continue;
		};
		if (count < MAXBKS) {
			printf("Enter the next title.\n");
		}
	}

	if (count > 0) {
		printf("Here is the list of your books:\n");
		for (index = 0; index < count ; index++) {
			printf("%s by %s: $%.2f\n", library[index].title, library[index].author,
					library[index].value);
		}

        sort_title(books, count);
		for (index = 0; index < count ; index++) {
			printf("%s by %s: $%.2f\n", books[index]->title, books[index]->author,
					books[index]->value);
		}
        
        sort_value(books, count);
		for (index = 0; index < count ; index++) {
			printf("%s by %s: $%.2f\n", books[index]->title, books[index]->author,
					books[index]->value);
		}
		
	}
	else {
		printf("No books? Too bad.\n");
	}

	return 0 ;
}

void sort_title(struct book * books[], int n) {
    struct book *temp;
    for (int i = 0; i < n - 1; i++) {
        for (int j = i; j < n; j++) {
            if (strcmp(books[i]->title, books[j]->title) > 0) {
                temp = books[i];
                books[i] = books[j];
                books[j] = temp;
            }
        }
    }
}

void sort_value(struct book * books[], int n) {
    struct book *temp;
    for (int i = 0; i < n - 1; i++) {
        for (int j = i; j < n; j++) {
            if (books[i]->value > books[j]->value) {
                temp = books[i];
                books[i] = books[j];
                books[j] = temp;
            }
        }
    }
}

char * s_gets(char * st, int n) {
	char * ret_val ;
	char * find ;
	ret_val = fgets(st, n, stdin);
	if (ret_val) {
		find = strchr(st, '\n');
		if (find) {
            *find = '\0';
		}
		else {
			while (getchar() != '\n') {
                continue;
			};
		}
	}
	return ret_val;
}
```

### Q4

```c
#include <stdio.h>
#include <string.h>

#define SIZE 20
#define LEN 5

struct names {
    char firstName[SIZE];
    char middleName[SIZE];
    char lastName[SIZE];
};

struct insurance {
    int no;
    struct names name;
};

void print(int, struct insurance []);

int main(void) {
    struct insurance arr[LEN] = {
        {1, {"san", "Aaaa", "zhang"}},
        {2, {"si", "", "li"}},
        {3, {"wu", "", "wang"}},
        {4, {"liu", "Davd", "zhao"}},
        {5, {"qi", "E", "bai"}}
    };
    print(LEN, arr);
	return 0;
}

void print(int n, struct insurance arr[n]) {
    for (int i = 0; i < n; i++) {
        if (strlen(arr[i].name.middleName)) {
            printf("%s, %s %c. -- %d\n", arr[i].name.firstName, arr[i].name.lastName,
                   arr[i].name.middleName[0], arr[i].no);
        }
        else {
            printf("%s, %s -- %d\n", arr[i].name.firstName, arr[i].name.lastName,
                   arr[i].no);
        }
    }
}
```

```c
#include <stdio.h>
#include <string.h>

#define SIZE 20
#define LEN 5

struct names {
    char firstName[SIZE];
    char middleName[SIZE];
    char lastName[SIZE];
};

struct insurance {
    int no;
    struct names name;
};

void print(struct insurance);

int main(void) {
    struct insurance arr[LEN] = {
        {1, {"san", "Aaaa", "zhang"}},
        {2, {"si", "", "li"}},
        {3, {"wu", "", "wang"}},
        {4, {"liu", "Davd", "zhao"}},
        {5, {"qi", "E", "bai"}}
    };
    for (int i = 0; i < LEN; i++) {
        print(arr[i]);
    }
	return 0;
}

void print(struct insurance a) {
    if (strlen(a.name.middleName)) {
        printf("%s, %s %c. -- %d\n", a.name.firstName, a.name.lastName,
                a.name.middleName[0], a.no);
    }
    else {
        printf("%s, %s -- %d\n", a.name.firstName, a.name.lastName,
                a.no);
    }
}
```

### Q5

```c
#include <stdio.h>
#include <string.h>

#define SIZE 20
#define CSIZE 4

struct name {
    char firstName[SIZE];
    char lastName[SIZE];
};
struct student {
    struct name stuName;
    float grade[3];
    float average;
};

void getGrades(struct student [], int n);
void getGradesAver(struct student [], int n);
void print(struct student [], int n);
void getAver(struct student [], int n);

int main(void) {
    struct student arr[CSIZE] = {
        [0].stuName = {"san", "zhang"},
        [1].stuName = {"si", "li"},
        [2].stuName = {"wu", "wang"},
        [3].stuName = {"liu", "zhao"},
    };
    getGrades(arr, CSIZE);
    getGradesAver(arr, CSIZE);
    print(arr, CSIZE);
    getAver(arr, CSIZE);
	return 0;
}

void getGrades(struct student arr[], int n) {
    char f[SIZE], l[SIZE];
    printf("Enter firstName & lastName:");
    while (scanf("%s %s", f, l) == 2) {
        for (int i = 0; i < n; i++) {
            if(strcmp(f, arr[i].stuName.firstName) == 0 &&
               strcmp(l, arr[i].stuName.lastName) == 0) {
                printf("Enter grades:");
                scanf("%f %f %f", &arr[i].grade[0], &arr[i].grade[1],
                      &arr[i].grade[2]);
            }
        }
        printf("Enter firstName & lastName:");
    }
}

void getGradesAver(struct student arr[], int n) {
    for (int i = 0; i < n; i++) {
        arr[i].average = (arr[i].grade[0] + arr[i].grade[1] +
              arr[i].grade[2]) / 3.0;
    }
}

void print(struct student arr[], int n) {
    for (int i = 0; i < n; i++) {
        printf("%s %s get grades %.2f, %.2f and %.2f, whose average is %.2f\n",
               arr[i].stuName.lastName, arr[i].stuName.firstName,
               arr[i].grade[0], arr[i].grade[1], arr[i].grade[2],
               arr[i].average);
    }
}

void getAver(struct student arr[], int n) {
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += arr[i].average;
    }
    printf("Average is %.2f\n", sum / CSIZE);
}
```

### Q6

```
0 Jessie Joybat 5 2 1 1
1 Mary Json 6 3 7 9
2 Filp Shell 5 3 6 1
3 Francy Card 9 6 1 2
4 Wan cary 8 5 2 1
5 Keassy Coffee 3 3 6 7
0 Jessie Joybat 7 9 8 2
5 Keassy Coffee 4 7 5 8
```

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 20
#define RANGE 19
#define FILENAME "test.txt"

struct player {
    int no;
    char fName[SIZE];
    char lName[SIZE];
    int playTimes;
    int hitTimes;
    int walkTimes;
    int rbi;
    float hitRate;
};

int main(void) {
    FILE * fp;
    if ((fp = fopen(FILENAME, "r")) == NULL) {
        printf("Can't open file %s", FILENAME);
		exit(EXIT_FAILURE);
    }
    struct player players[RANGE];
    for (int i = 0; i < RANGE; i++) {
        players[i].no = i;
        strcpy(players[i].fName, "");
        strcpy(players[i].lName, "");
        players[i].playTimes = 0;
        players[i].hitTimes = 0;
        players[i].walkTimes = 0;
        players[i].rbi = 0.0;
    }
    struct player temp;
    while (fscanf(fp, "%d %s %s %d %d %d %d", &(temp.no),
                  temp.fName, temp.lName, &(temp.playTimes),
                  &(temp.hitTimes), &(temp.walkTimes), &(temp.rbi))
                  != EOF) {
        for (int i = 0; i < RANGE; i++) {
            if (players[i].no == temp.no) {
                strcpy(players[i].fName, temp.fName);
                strcpy(players[i].lName, temp.lName);
                players[i].playTimes += temp.playTimes;
                players[i].hitTimes += temp.hitTimes;
                players[i].walkTimes += temp.walkTimes;
                players[i].rbi += temp.rbi;
            }
        }
    }
    for (int i = 0; i < RANGE; i++) {
        if (players[i].playTimes == 0) {
            players[i].hitRate = 0;
        }
        else {
            players[i].hitRate = (float)players[i].hitTimes / (float)players[i].playTimes;
        }
        if (strcmp(players[i].fName,"") != 0 && strcmp(players[i].lName,"") != 0) {
            printf("No%d: %s, %s. playTimes: %d, hitTimes: %d, "
                "walkTimes: %d, rbi: %d, hitRate: %f\n", players[i].no,
                players[i].fName, players[i].lName, players[i].playTimes,
                players[i].hitTimes, players[i].walkTimes, players[i].rbi,
                players[i].hitRate);
        }
    }
    if (fclose(fp) != 0) {
        printf("Can't close file %s", FILENAME);
		exit(EXIT_FAILURE);
    }
	return 0;
}
```

### Q7

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXTITL 40
#define MAXAUTL 40
#define MAXBKS 3

typedef struct book {
    char title[MAXTITL];
    char author[MAXAUTL];
    float value;
} book;

void menu(void);
int add (book * [], int);
void modify (book * [], int);
int delete (book * [], int);
void show (book * [], int);
void eatline(void);
char *s_gets(char *st, int n);

int main(void) {
    book library[MAXBKS];
    book * plibrary [MAXBKS];
    for (int i = 0; i < MAXBKS; i++) {
        plibrary[i] = &library[i];
    }
    FILE *pbooks;
    int count = 0;

    // 步骤一:读取文件并显示
    if ((pbooks = fopen("test.txt", "r+b")) == NULL) {
        fputs("Can't open file.\n", stderr);
        exit(1);
    }
    while (count < MAXBKS && fread(plibrary[count], sizeof(book), 1, pbooks) == 1)
    {
        if (count == 0) {
            puts("Current contents of :");
        }
        printf("%s by %s: $%.2f\n", plibrary[count]->title,
                plibrary[count]->author, plibrary[count]->value);
        count++;
    }
    fclose(pbooks);

    // 步骤二
    menu();
    char ch;
    printf("Enter your choice:");
    while ((ch = getchar()) != 'q') {
        eatline();
        switch (ch) {
            case 'a': {
                count = add(plibrary, count);
                break;
            }
            case 'm': {
                modify(plibrary, count);
                break;
            }
            case 'd': {
                count = delete(plibrary, count);
                break;
            }
            default: {
                puts("Please Enter \'a\', \'m\', \'d\', or \'q\'.");
            }
        }
        menu();
        printf("Enter your choice:");
    }

    // 步骤三：保存文件并显示
    if ((pbooks = fopen("data/14-7.txt", "w+b")) == NULL) {
        fputs("Can't open file.\n", stderr);
        exit(1);
    }
    if (count > 0) {
        puts("Here is the list of your books:");
        for (int i = 0; i < count; i++) {
            printf("%s by %s: $%.2f\n", plibrary[i]->title,
            plibrary[i]->author, plibrary[i]->value);
            fwrite(plibrary[i], sizeof(book), 1, pbooks);
        }
    }
    else
        puts("No books? Too bad.\n");

    fclose(pbooks);
    puts("Bye.\n");

    return 0;
}

void menu (void) {
    puts("**********************************************");
    puts("a) add books    m) modify books");
    puts("d) delete books q) quit(save)");
    puts("**********************************************");
}

void show (book * plibrary[], int count) {
    if (count > 0) {
        puts("Here is the list of your books:");
        for (int i = 0; i < count; i++) {
            printf("%s by %s: $%.2f\n", plibrary[i]->title,
            plibrary[i]->author, plibrary[i]->value);
        }
    }
    else {
        puts("No books? Too bad.");
    }
}

int add (book * plibrary[], int count) {
    if (count == MAXBKS) {
        fputs("The file is full.\n", stderr);
        return count;
    }
    puts("Please add new book titles.");
    puts("Press [enter] at the start of a line to stop.");
    while (count < MAXBKS && s_gets(plibrary[count]->title, MAXTITL) != NULL
        && plibrary[count]->title[0] != '\0') {
        puts("Now Enter the author.");
        s_gets(plibrary[count]->author, MAXAUTL);
        puts("Now Enter the value.");
        scanf("%f", &(plibrary[count]->value));
        eatline();
        count++;
        show(plibrary, count);
        if (count < MAXBKS) {
            puts("Enter the next title.");
        }
    }
    return count;
}

void modify (book * plibrary[], int count) {
    int choice;
    char title[MAXTITL];
    puts("Please enter the title of the book you want to modify.");
    puts("Press [enter] at the start of a line to stop.");
    while (s_gets(title, MAXTITL) != NULL && title[0] != '\0') {
        for (int i = 0; i < count; i++) {
            if (!strcmp(title, plibrary[i]->title)) {
                puts("Please select what you want to modify: title(0), author(1) and price(2).");
                scanf("%d", &choice);
                eatline();
                if (choice == 0) {
                    char temp[MAXTITL];
                    printf("Please enter the modified title: ");
                    s_gets(temp, MAXTITL);
                    strcpy(plibrary[i]->title, temp);
                    break;
                }
                if (choice == 1) {
                    char temp[MAXAUTL];
                    printf("Please enter the modified author: ");
                    s_gets(temp, MAXAUTL);
                    strcpy(plibrary[i]->author, temp);
                    break;
                }
                if (choice == 2) {
                    float temp;
                    printf("Please enter the modified price: ");
                    scanf("%f", &temp);
                    eatline();
                    plibrary[i]->value = temp;
                    break;
                }
            }
        }
        show(plibrary, count);
        puts("Please enter the title of the book you want to modify.");
    }
}

int delete (book * plibrary[], int count) {
    char title[MAXTITL];
    if (count == 0) {
        fputs("The file is empty.\n", stderr);
        return count;
    }
    puts("Please delete book titles.");
    puts("Press [enter] at the start of a line to stop.");
    while (count > 0 && s_gets(title, MAXTITL) != NULL && title[0] != '\0') {
        for (int i = 0; i < count; i++) {
            if (!strcmp(title, plibrary[i]->title)) {
                for (int j = i; j < count - 1; j++) {
                    strcpy(plibrary[j]->title, plibrary[j+1]->title);
                    strcpy(plibrary[j]->author, plibrary[j+1]->author);
                    plibrary[j]->value = plibrary[j+1]->value;
                }
                count--;
                break;
            }
        }
        show(plibrary, count);
        if (count > 0) {
            puts("Enter the next title.");
        }
    }
    return count;
}

void eatline(void) {
    while (getchar() != '\n') {
        continue;
    }
}

char * s_gets(char * st, int n)
{
    char * ret_val;
    char * find;

    ret_val = fgets(st, n, stdin);
    if (ret_val)
    {
        find = strchr(st, '\n');
        if (find)
        {
            *find = '\0';
        }
        else
        {
            while (getchar() != '\n')
                continue;
        }
    }
    return ret_val;
}
```

### Q8

```c
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>

#define SIZE 20
#define LEN 12

typedef struct seat {
    int no;
    bool reserved;
    char fName[SIZE];
    char lName[SIZE];
} seat;

void menu(void);
void init(seat [], seat * [], int);
void showNumEmpty(seat * [], int);
void showListEmpty(seat * [], int);
int comp(seat *, seat *);
void showAlphaList(seat * [], int);
void seatAssign(seat * [], int);
void seatDel(seat * [], int);
char * s_gets(char *, int);

int main(void) {
    seat seats[LEN];
    seat * pseats[LEN];
    init(seats, pseats, LEN);
    menu();
    char ch;
    printf("Enter your choice:");
    while ((ch = getchar()) != 'f') {
        switch (ch) {
            case 'a': {
                showNumEmpty(pseats, LEN);
                break;
            }
            case 'b': {
                showListEmpty(pseats, LEN);
                break;
            }
            case 'c': {
                showAlphaList(pseats, LEN);
                break;
            }
            case 'd': {
                seatAssign(pseats, LEN);
                break;
            }
            case 'e': {
                seatDel(pseats, LEN);
                break;
            }
            default: {
                printf("Enter \'a\', \'b\', \'c\', \'d\', "
                       "\'e\' or \'f\'\n");
            }
        }
        menu();
        printf("Enter your choice:");
        while (getchar() != '\n') ;
    }
	return 0;
}

void menu(void) {
    puts("To choose a function enter its letter label:");
    puts("a) Show number of empty seats");
    puts("b) Show list of empty seats");
    puts("c) Show alphabetical list of seats");
    puts("d) Assign a customer to a seat assignment");
    puts("e) Delete a seat assignment");
    puts("f) Quit");
}

void init(seat seats[], seat * pseats[], int n) {
    for (int i = 0; i < n; i++) {
        seats[i].no = i + 1;
        seats[i].reserved = false;
        strcpy(seats[i].fName, "");
        strcpy(seats[i].lName, "");
        pseats[i] = &seats[i];
    }
}

void showNumEmpty(seat * pseats[], int n) {
    int num = 0;
    for (int i = 0; i < n; i++) {
        if (pseats[i]->reserved == false) {
            num++;
        }
    }
    printf("The number of empty seats is %d.\n", num);
};

void showListEmpty(seat * pseats[], int n) {
    for (int i = 0; i < n; i++) {
        if (pseats[i]->reserved == false) {
            printf("No.%d, ", pseats[i]->no);
        }
    }
    printf("\b\b seat has been unreserved.\n");
}

int comp(seat * a, seat * b) {
    if (strcmp(a->lName, b->lName) != 0) {
        return strcmp(a->lName, b->lName);
    }
    else if (strcmp(a->fName, b->fName) != 0) {
        return strcmp(a->fName, b->fName);
    }
    else {
        return a->no-b->no;
    }
}

void showAlphaList(seat * pseats[], int n) {
    seat ** ptemp = (seat **) malloc(n * sizeof(seat *));
    for (int i = 0; i < n; i++) {
        ptemp[i] = pseats[i];
    }
    seat * temp;
    for (int i = 0; i < n-1; i++) {
        if (ptemp[i]->reserved == false) {
            continue;
        }
        for (int j = i+1; j < n; j++) {
            if (ptemp[j]->reserved == false) {
                continue;
            }
            if (comp(ptemp[i], ptemp[j]) > 0) {
                temp = ptemp[i];
                ptemp[i] = ptemp[j];
                ptemp[j] = temp;
            }
        }
    }
    bool flag = true;
    for (int i = 0; i < n; i++) {
        if (ptemp[i]->reserved != false) {
            flag = false;
            printf("%d: %s, %s\n", ptemp[i]->no, ptemp[i]->fName, ptemp[i]->lName);
        }
    }
    if (flag) {
        printf("None has benn reserved!\n");
    }
    free(ptemp);
}

void seatAssign(seat * pseats[], int n) {
    char fname[SIZE], lname[SIZE];
    printf("Enter your name: ");
    scanf("%s %s", fname, lname);
    printf("Now you can choice ");
    for (int i = 0; i < n; i++) {
        if (pseats[i]->reserved == false) {
            printf("%d,", pseats[i]->no);
        }
    }
    printf("\b.\n");
    printf("Enter your choice: ");
    int choice;
    scanf("%d", &choice);
    int no;
    for (int i = 0; i < n; i++) {
        if (pseats[i]->no == choice) {
            no = i;
        }
    }
    if (pseats[no]->reserved) {
        printf("The seat has been reserved.\n");
    }
    else {
        int flag = 0;
        printf("Would you want save your choice? 0)conceal, 1)save.");
        scanf("%d", &flag);
        if (flag) {
            for (int i = 0; i < n; i++) {
                if (pseats[i]->no == choice) {
                    pseats[i]->reserved = true;
                    strcpy(pseats[i]->fName, fname);
                    strcpy(pseats[i]->lName, lname);
                }
            }
        }
    }
}

void seatDel(seat * pseats[], int n) {
    int no;
    char fname[SIZE], lname[SIZE];
    printf("Enter your name: ");
    scanf("%s %s", fname, lname);
    printf("Enter your seat no: ");
    scanf("%d", &no);
    int flag = 0;
    printf("Would you want save your choice? 0)conceal, 1)save.");
    scanf("%d", &flag);
    if (flag) {
        int i;
        for (i = 0; i < n; i++) {
            if (strcmp(pseats[i]->fName, fname) == 0 &&
                strcmp(pseats[i]->lName, lname) == 0 &&
                pseats[i]->no == no) {
                pseats[i]->reserved = 0;
                strcpy(pseats[i]->fName, "");
                strcpy(pseats[i]->lName, "");
                break;
            }
        }
        if (i == n) {
            printf("No matching information found.\n");
        }
    }
}

char * s_gets(char * st, int n) {
    char * ret_val;
    int i = 0;

    ret_val = fgets(st, n, stdin);
    if (ret_val) {
        while (st[i] != '\n' && st[i] != '\0') {
            i++;
        }
        if (st[i] == '\n') {
            st[i] = '\0';
        }
        else {
            while (getchar() != '\n') {
                continue;
            }
        }
    }
    return ret_val;
}
```

### Q9

```c
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>

#define SIZE 20
#define LEN 12
#define NUM 4

typedef struct seat {
    int no;
    bool reserved;
    char fName[SIZE];
    char lName[SIZE];
} seat;

typedef struct plane {
    char name[SIZE];
    seat seats[LEN];
} plane;

void menu(void);
void init(seat [], seat * [], int);
void showNumEmpty(seat * [], int);
void showListEmpty(seat * [], int);
int comp(seat *, seat *);
void showAlphaList(seat * [], int);
void seatAssign(seat * [], int);
void seatDel(seat * [], int);
void seatConfirm(seat * [], int);
void eatline(void);
char * s_gets(char *, int);

int main(void) {
    plane planes[NUM] = {
        {.name="102"},
        {.name="311"},
        {.name="444"},
        {.name="519"}
    };
    for (int i = 0; i < NUM; i++) {
        for (int j = 0; j < LEN; j++) {
            planes[i].seats[j].no = j + 1;
            planes[i].seats[j].reserved = false;
            strcpy(planes[i].seats[j].fName, "");
            strcpy(planes[i].seats[j].lName, "");
        }
    }
    int choice;
    puts("please choice a plane? 0)102, 1)311, 2)444, 3)519.");
    while(scanf("%d", &choice) == 1) {
        if (choice < 0 || choice > 3) {
            break;
        }
        eatline();
        seat * pseats[LEN];
        for (int i = 0; i < LEN; i++) {
            pseats[i] = &planes[choice].seats[i];
        }
        menu();
        char ch;
        printf("Enter your choice:");
        while ((ch = getchar()) != 'f') {
            printf("Plane %s:\n", planes[choice].name);
            switch (ch) {
                case 'a': {
                    showNumEmpty(pseats, LEN);
                    break;
                }
                case 'b': {
                    showListEmpty(pseats, LEN);
                    break;
                }
                case 'c': {
                    showAlphaList(pseats, LEN);
                    break;
                }
                case 'd': {
                    seatAssign(pseats, LEN);
                    break;
                }
                case 'e': {
                    seatDel(pseats, LEN);
                    break;
                }
                case 'g': {
                    seatConfirm(pseats, LEN);
                    break;
                }
                default: {
                    printf("Enter \'a\', \'b\', \'c\', \'d\', "
                        "\'e\', \'f\' or \'f\'\n");
                }
            }
            menu();
            printf("Enter your choice:");
            eatline();
        }
        puts("please choice a plane? 0)102, 1)311, 2)444, 3)519.");
    }
    
	return 0;
}

void menu(void) {
    puts("To choose a function enter its letter label:");
    puts("a) Show number of empty seats");
    puts("b) Show list of empty seats");
    puts("c) Show alphabetical list of seats");
    puts("d) Assign a customer to a seat assignment");
    puts("e) Delete a seat assignment");
    puts("f) Quit");
    puts("g) Confirm seat assignment");
}

void init(seat seats[], seat * pseats[], int n) {
    for (int i = 0; i < n; i++) {
        pseats[i] = &seats[i];
    }
}

void showNumEmpty(seat * pseats[], int n) {
    int num = 0;
    for (int i = 0; i < n; i++) {
        if (pseats[i]->reserved == false) {
            num++;
        }
    }
    printf("The number of empty seats is %d.\n", num);
};

void showListEmpty(seat * pseats[], int n) {
    for (int i = 0; i < n; i++) {
        if (pseats[i]->reserved == false) {
            printf("No.%d, ", pseats[i]->no);
        }
    }
    printf("\b\b seat has been unreserved.\n");
}

int comp(seat * a, seat * b) {
    if (strcmp(a->lName, b->lName) != 0) {
        return strcmp(a->lName, b->lName);
    }
    else if (strcmp(a->fName, b->fName) != 0) {
        return strcmp(a->fName, b->fName);
    }
    else {
        return a->no-b->no;
    }
}

void showAlphaList(seat * pseats[], int n) {
    seat ** ptemp = (seat **) malloc(n * sizeof(seat *));
    for (int i = 0; i < n; i++) {
        ptemp[i] = pseats[i];
    }
    seat * temp;
    for (int i = 0; i < n-1; i++) {
        if (ptemp[i]->reserved == false) {
            continue;
        }
        for (int j = i+1; j < n; j++) {
            if (ptemp[j]->reserved == false) {
                continue;
            }
            if (comp(ptemp[i], ptemp[j]) > 0) {
                temp = ptemp[i];
                ptemp[i] = ptemp[j];
                ptemp[j] = temp;
            }
        }
    }
    bool flag = true;
    for (int i = 0; i < n; i++) {
        if (ptemp[i]->reserved != false) {
            flag = false;
            printf("%d: %s, %s\n", ptemp[i]->no, ptemp[i]->fName, ptemp[i]->lName);
        }
    }
    if (flag) {
        printf("None has benn reserved!\n");
    }
    free(ptemp);
}

void seatAssign(seat * pseats[], int n) {
    char fname[SIZE], lname[SIZE];
    printf("Enter your name: ");
    scanf("%s %s", fname, lname);
    printf("Now you can choice ");
    for (int i = 0; i < n; i++) {
        if (pseats[i]->reserved == false) {
            printf("%d,", pseats[i]->no);
        }
    }
    printf("\b.\n");
    printf("Enter your choice: ");
    int choice;
    scanf("%d", &choice);
    int no;
    for (int i = 0; i < n; i++) {
        if (pseats[i]->no == choice) {
            no = i;
        }
    }
    if (pseats[no]->reserved) {
        printf("The seat has been reserved.\n");
    }
    else {
        int flag = 0;
        printf("Would you want save your choice? 0)conceal, 1)save.");
        scanf("%d", &flag);
        if (flag) {
            for (int i = 0; i < n; i++) {
                if (pseats[i]->no == choice) {
                    pseats[i]->reserved = true;
                    strcpy(pseats[i]->fName, fname);
                    strcpy(pseats[i]->lName, lname);
                }
            }
        }
    }
}

void seatDel(seat * pseats[], int n) {
    int no;
    char fname[SIZE], lname[SIZE];
    printf("Enter your name: ");
    scanf("%s %s", fname, lname);
    printf("Enter your seat no: ");
    scanf("%d", &no);
    int flag = 0;
    printf("Would you want save your choice? 0)conceal, 1)save.");
    scanf("%d", &flag);
    if (flag) {
        int i;
        for (i = 0; i < n; i++) {
            if (strcmp(pseats[i]->fName, fname) == 0 &&
                strcmp(pseats[i]->lName, lname) == 0 &&
                pseats[i]->no == no) {
                pseats[i]->reserved = 0;
                strcpy(pseats[i]->fName, "");
                strcpy(pseats[i]->lName, "");
                break;
            }
        }
        if (i == n) {
            printf("No matching information found.\n");
        }
    }
}

void seatConfirm(seat * pseats[], int n) {
    for (int i = 0; i < n; i++) {
        if (pseats[i]->reserved == true) {
            printf("Seat %d has been reserved.\n", pseats[i]->no);
        }
        else {
            printf("Seat %d has not been reserved.\n", pseats[i]->no);
        }
    }
}

void eatline(void) {
    while (getchar() != '\n') ;
}

char * s_gets(char * st, int n) {
    char * ret_val;
    int i = 0;

    ret_val = fgets(st, n, stdin);
    if (ret_val) {
        while (st[i] != '\n' && st[i] != '\0') {
            i++;
        }
        if (st[i] == '\n') {
            st[i] = '\0';
        }
        else {
            while (getchar() != '\n') {
                continue;
            }
        }
    }
    return ret_val;
}
```

### Q10

```c
#include <stdio.h>

#define SIZE 4

void menu(void);
void printA(void);
void printB(void);
void printC(void);
void printD(void);

int main(void) {
    void (*ptf[SIZE])(void) = {printA, printB, printC, printD};
    menu();
    char ch;
    while ((ch = getchar()) != 'q') {
        switch(ch) {
            case 'a': {
                ptf[0]();
                break;
            }
            case 'b': {
                ptf[1]();
                break;
            }
            case 'c': {
                ptf[2]();
                break;
            }
            case 'd': {
                ptf[3]();
                break;
            }
            default: {
                printf("Enter \'a\', \'b\', \'c\', \'d\' or  \'q\'\n");
            }
        }
        menu();
        while (getchar() != '\n') ;
    }
	return 0;
}

void menu(void) {
    printf("**************************\n");
    printf("a)A         b)B\n");
    printf("c)C         d)D\n");
    printf("q)quit\n");
    printf("**************************\n");
}

void printA(void) {
    printf("A\n");
}

void printB(void) {
    printf("B\n");
}

void printC(void) {
    printf("C\n");
}

void printD(void) {
    printf("D\n");
}
```

### Q11

```c
#include <stdio.h>
#include <math.h>

#define SIZE 3

void transform(double [], double [], int, double (*)(double));
double addOne(double);
double minusOne(double);
void print(double [], int);

int main(void) {
    double a[SIZE] = {1, 2, 3};
    double b[SIZE] = {0, 0, 0};
    print(b, SIZE);
    transform(a, b, SIZE, sin);
    print(b, SIZE);
    transform(a, b, SIZE, cos);
    print(b, SIZE);
    transform(a, b, SIZE, addOne);
    print(b, SIZE);
    transform(a, b, SIZE, minusOne);
    print(b, SIZE);
	return 0;
}

void transform(double a[], double b[], int n, double (*fun)(double)) {
    for (int i = 0; i < n; i++) {
        b[i] = fun(a[i]);
    }
}

double addOne(double a) {
    return a+1.0;
}

double minusOne(double a) {
    return a-1.0;
}

void print(double a[], int n) {
    for (int i = 0; i < n; i++) {
        printf("%.2lf ", a[i]);
    }
    printf("\n");
}
```