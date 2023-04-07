# Chapter 17

### Q1

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TSIZE 45

struct film {
    char title[TSIZE];
    int rating;
    struct film *next;
};

char *s_gets(char *st, int n);
void print_r(struct film *);

int main(void)
{
    struct film *head = NULL;
    struct film *prev, *current;
    char input[TSIZE];

    puts("Enter first movie title:");
    while (s_gets(input, TSIZE) != NULL && input[0] != '\0')
    {
        current = (struct film *)malloc(sizeof(struct film));
        if (head == NULL)
        {
            head = current;
        }
        else
        {
            prev->next = current;
        }
        current->next = NULL;
        strcpy(current->title, input);
        puts("Enter your rating <0-10>:");
        scanf("%d", &current->rating);
        while (getchar() != '\n')
            continue;
        puts("Enter next movie title (empty line to stop):");
        prev = current;
    }

    if (head == NULL)
    {
        printf("No data entered. ");
    }
    else
    {
        printf("Here is the movie list:\n");
    }
    current = head;
    while (current != NULL)
    {
        printf("Movie: %s  Rating: %d\n",
               current->title, current->rating);
        current = current->next;
    }
    if (head != NULL) {
        printf("Here is the reversed movie list:\n");
        print_r(head);
    }
    current = head;
    while (current != NULL)
    {
        current = head;
        head = current->next;
        free(current);
    }
    printf("Bye!\n");

    return 0;
}

void print_r(struct film *head)
{
    if (head->next != NULL)
    {
        print_r(head->next);
    }
    printf("Movie: %s  Rating: %d\n",
            head->title, head->rating);
}

char *s_gets(char *st, int n)
{
    char *ret_val;
    char *find;

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

# Q2

```c
// list.h
#ifndef LIST_H_
#define LIST_H_
#include <stdbool.h>
#define TSIZE 45

struct film
{
    char title[TSIZE];
    int rating;
};

typedef struct film Item;

typedef struct node
{
    Item item;
    struct node *next;
} Node;

typedef struct list
{
    Node *head;
    Node *end;
} List;

void InitializeList(List *plist);

bool ListIsEmpty(const List *plist);

bool ListIsFull(const List *plist);

unsigned int ListItemCount(const List *plist);

bool AddItem(Item item, List *plist);

void Traverse(const List *plist, void (*pfun)(Item item));

void EmptyTheList(List *plist);

#endif
```

```c
// list.c
#include <stdio.h>
#include <stdlib.h>
#include "list.h"

static void CopyToNode(Item item, Node * pnode);

void InitializeList(List * plist)
{
    (*plist).head = NULL;
    (*plist).end = NULL;
}

bool ListIsEmpty(const List * plist)
{
    return (*plist).head == NULL;
}

bool ListIsFull(const List * plist)
{
    Node * pt;
    bool full;

    pt = (Node *)malloc(sizeof(Node));
    if (pt == NULL)
    {
        full = true;
    }
    else
    {
        full = false;
    }
    free(pt);

    return full;
}

unsigned int ListItemCount(const List * plist)
{
    unsigned int count = 0;
    Node * pnode = (*plist).head;

    while (pnode != NULL)
    {
        ++count;
        pnode = pnode->next;
    }
    return count;
}

bool AddItem(Item item, List * plist)
{
    Node * pnew;
    Node * scan = (*plist).head;

    pnew = (Node *)malloc(sizeof(Node));
    if (pnew == NULL)
    {
        return false;
    }
    CopyToNode(item, pnew);
    pnew->next = NULL;
    if (scan == NULL)
    {
        (*plist).head = pnew;
        (*plist).end = pnew;
    }
    else
    {
        (*plist).end->next = pnew;
        (*plist).end = pnew;
    }

    return true;
}

void Traverse(const List * plist, void (*pfun)(Item item))
{
    Node * pnode = (*plist).head;

    while (pnode != NULL)
    {
        (*pfun)(pnode->item);
        pnode = pnode->next;
    }
}

void EmptyTheList(List * plist)
{
    Node * psave;

    while ((*plist).head != NULL)
    {
        psave = (*plist).head->next;
        free((*plist).head);
        (*plist).head = psave;
    }
}

static void CopyToNode(Item item, Node * pnode)
{
    pnode->item = item;
}
```

```c
// films.c
#include <stdio.h>
#include <stdlib.h>    /* prototype for exit() */
#include <string.h>
#include "list.h"      /* defines List, Item   */
void showmovies(Item item);
char * s_gets(char * st, int n);
int main(void)
{
    List movies;
    Item temp;
    
    
    /* initialize       */
    InitializeList(&movies);
    if (ListIsFull(&movies))
    {
        fprintf(stderr,"No memory available! Bye!\n");
        exit(1);
    }
    
    /* gather and store */
    puts("Enter first movie title:");
    while (s_gets(temp.title, TSIZE) != NULL && temp.title[0] != '\0')
    {
        puts("Enter your rating <0-10>:");
        scanf("%d", &temp.rating);
        while(getchar() != '\n')
            continue;
        if (AddItem(temp, &movies)==false)
        {
            fprintf(stderr,"Problem allocating memory\n");
            break;
        }
        if (ListIsFull(&movies))
        {
            puts("The list is now full.");
            break;
        }
        puts("Enter next movie title (empty line to stop):");
    }
    
    /* display          */
    if (ListIsEmpty(&movies))
        printf("No data entered. ");
    else
    {
        printf ("Here is the movie list:\n");
        Traverse(&movies, showmovies);
    }
    printf("You entered %d movies.\n", ListItemCount(&movies));
    
    
    /* clean up         */
    EmptyTheList(&movies);
    printf("Bye!\n");
    
    return 0;
}

void showmovies(Item item)
{
    printf("Movie: %s  Rating: %d\n", item.title,
           item.rating);
}

char * s_gets(char * st, int n)
{
    char * ret_val;
    char * find;
    
    ret_val = fgets(st, n, stdin);
    if (ret_val)
    {
        find = strchr(st, '\n');   // look for newline
        if (find)                  // if the address is not NULL,
            *find = '\0';          // place a null character there
        else
            while (getchar() != '\n')
                continue;          // dispose of rest of line
    }
    return ret_val;
}
```

# Q3

```c
// list.h
#ifndef LIST_H_
#define LIST_H_
#include <stdbool.h>

#define TSIZE 45
struct film
{
    char title[TSIZE];
    int rating;
};

typedef struct film Item;

#define MAXSIZE 100
typedef struct list
{
    Item entries[MAXSIZE];
    int items;
} List;

void InitializeList(List *plist);

bool ListIsEmpty(const List *plist);

bool ListIsFull(const List *plist);

unsigned int ListItemCount(const List *plist);

bool AddItem(Item item, List *plist);

void Traverse(const List *plist, void (*pfun)(Item item));

void EmptyTheList(List *plist);

#endif
```

```c
// list.c
#include <stdio.h>
#include <stdlib.h>
#include "list.h"

void InitializeList(List * plist)
{
    plist->items = 0;
}

bool ListIsEmpty(const List * plist)
{
    return plist->items == 0;
}

bool ListIsFull(const List * plist)
{
    return plist->items == MAXSIZE;
}

unsigned int ListItemCount(const List * plist)
{
    return plist->items;
}

bool AddItem(Item item, List * plist)
{
    if (plist->items == MAXSIZE)
    {
        return false;
    }
    else
    {
        (plist->entries)[plist->items] = item;
        (plist->items)++;
    }

    return true;
}

void Traverse(const List * plist, void (*pfun)(Item item))
{
    for (int i = 0; i < plist->items; i++)
    {
        (*pfun)((plist->entries)[i]);
    }
}

void EmptyTheList(List * plist)
{
    plist->items = 0;
}
```

```c
// films.c
#include <stdio.h>
#include <stdlib.h>    /* prototype for exit() */
#include <string.h>
#include "list.h"      /* defines List, Item   */
void showmovies(Item item);
char * s_gets(char * st, int n);
int main(void)
{
    List movies;
    Item temp;
    
    
    /* initialize       */
    InitializeList(&movies);
    if (ListIsFull(&movies))
    {
        fprintf(stderr,"No memory available! Bye!\n");
        exit(1);
    }
    
    /* gather and store */
    puts("Enter first movie title:");
    while (s_gets(temp.title, TSIZE) != NULL && temp.title[0] != '\0')
    {
        puts("Enter your rating <0-10>:");
        scanf("%d", &temp.rating);
        while(getchar() != '\n')
            continue;
        if (AddItem(temp, &movies)==false)
        {
            fprintf(stderr,"Problem allocating memory\n");
            break;
        }
        if (ListIsFull(&movies))
        {
            puts("The list is now full.");
            break;
        }
        puts("Enter next movie title (empty line to stop):");
    }
    
    /* display          */
    if (ListIsEmpty(&movies))
        printf("No data entered. ");
    else
    {
        printf ("Here is the movie list:\n");
        Traverse(&movies, showmovies);
    }
    printf("You entered %d movies.\n", ListItemCount(&movies));
    
    
    /* clean up         */
    EmptyTheList(&movies);
    printf("Bye!\n");
    
    return 0;
}

void showmovies(Item item)
{
    printf("Movie: %s  Rating: %d\n", item.title,
           item.rating);
}

char * s_gets(char * st, int n)
{
    char * ret_val;
    char * find;
    
    ret_val = fgets(st, n, stdin);
    if (ret_val)
    {
        find = strchr(st, '\n');   // look for newline
        if (find)                  // if the address is not NULL,
            *find = '\0';          // place a null character there
        else
            while (getchar() != '\n')
                continue;          // dispose of rest of line
    }
    return ret_val;
}
```

### Q4

```c
// queue.h
#ifndef _QUEUE_H_
#define _QUEUE_H_
#include <stdbool.h>

#define MAXQUEUE 10

typedef struct item
{
    long arrive;
    int processtime;
} Item;
typedef struct node
{
    Item item;
    struct node * next;
} Node;
typedef struct queue
{
    Node * front;
    Node * rear;
    int items;
} Queue;

void InitializeQueue(Queue * pq);
bool QueueIsFull(const Queue * pq);
bool QueueIsEmpty(const Queue *pq);
int QueueItemCount(const Queue * pq);
bool EnQueue(Item item, Queue * pq);
bool DeQueue(Item *pitem, Queue * pq);
void EmptyTheQueue(Queue * pq);

#endif
```

```c
// queue.c
#include <stdio.h>
#include <stdlib.h>
#include "queue.h"

static void CopyToNode(Item item, Node * pn);
static void CopyToItem(Node * pn, Item * pi);

void InitializeQueue(Queue * pq)
{
    pq->front = pq->rear = NULL;
    pq->items = 0;
}

bool QueueIsFull(const Queue * pq)
{
    return pq->items == MAXQUEUE;
}

bool QueueIsEmpty(const Queue * pq)
{
    return pq->items == 0;
}

int QueueItemCount(const Queue * pq)
{
    return pq->items;
}

bool EnQueue(Item item, Queue * pq)
{
    Node * pnew;
    
    if (QueueIsFull(pq))
        return false;
    pnew = (Node *) malloc( sizeof(Node));
    if (pnew == NULL)
    {
        fprintf(stderr,"Unable to allocate memory!\n");
        exit(1);
    }
    CopyToNode(item, pnew);
    pnew->next = NULL;
    if (QueueIsEmpty(pq))
        pq->front = pnew;
    else
        pq->rear->next = pnew;
    pq->rear = pnew;
    pq->items++;
    
    return true;
}

bool DeQueue(Item * pitem, Queue * pq)
{
    Node * pt;
    
    if (QueueIsEmpty(pq))
        return false;
    CopyToItem(pq->front, pitem);
    pt = pq->front;
    pq->front = pq->front->next;
    free(pt);
    pq->items--;
    if (pq->items == 0)
        pq->rear = NULL;
    
    return true;
}

void EmptyTheQueue(Queue * pq)
{
    Item dummy;
    while (!QueueIsEmpty(pq))
        DeQueue(&dummy, pq);
}

static void CopyToNode(Item item, Node * pn)
{
    pn->item = item;
}

static void CopyToItem(Node * pn, Item * pi)
{
    *pi = pn->item;
}
```

```c
// mall.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "queue.h"
#define N 2
#define MIN_PER_HR 60.0

typedef struct mall
{
    Queue line;
    int hours;
    int perhour;
    long cycle, cyclelimit;
    long turnaways;
    long customers;
    long served;
    long sum_line;
    int wait_time;
    double min_per_cust;
    long line_wait; 
} mall;

bool newcustomer(double x);
Item customertime(long when);

int main(void)
{
    mall malls[N];
    Item temp;
    
    srand((unsigned int) time(0));
    puts("Case Study: Sigmund Lander's Advice Booth");
    puts("Enter the number of simulation hours for mall 1:");
    scanf("%d", &malls[0].hours);
    malls[0].cyclelimit = MIN_PER_HR * malls[0].hours;
    puts("Enter the average number of customers per hour for mall 1:");
    scanf("%d", &malls[0].perhour);
    malls[0].min_per_cust = MIN_PER_HR / malls[0].perhour;

    puts("Enter the number of simulation hours for mall 2:");
    scanf("%d", &malls[1].hours);
    malls[1].cyclelimit = MIN_PER_HR * malls[1].hours;
    puts("Enter the average number of customers per hour for mall 2:");
    scanf("%d", &malls[1].perhour);
    malls[1].min_per_cust = MIN_PER_HR / malls[1].perhour;

    for (int i = 0; i < N; i++)
    {
        InitializeQueue(&malls[i].line);
        malls[i].cycle = 0;
        malls[i].turnaways = 0;
        malls[i].customers = 0;
        malls[i].served = 0;
        malls[i].sum_line = 0;
        malls[i].wait_time = 0;
        malls[i].line_wait = 0;
        for (malls[i].cycle = 0; malls[i].cycle < malls[i].cyclelimit; malls[i].cycle++)
        {
            if (newcustomer(malls[i].min_per_cust))
            {
                if (QueueIsFull(&malls[i].line))
                    malls[i].turnaways++;
                else
                {
                    malls[i].customers++;
                    temp = customertime(malls[i].cycle);
                    EnQueue(temp, &malls[i].line);
                }
            }
            if (malls[i].wait_time <= 0 && !QueueIsEmpty(&malls[i].line))
            {
                DeQueue (&temp, &malls[i].line);
                malls[i].wait_time = temp.processtime;
                malls[i].line_wait += malls[i].cycle - temp.arrive;
                malls[i].served++;
            }
            if (malls[i].wait_time > 0)
                malls[i].wait_time--;
            malls[i].sum_line += QueueItemCount(&malls[i].line);
        }
        
        if (malls[i].customers > 0)
        {
            printf("customers accepted: %ld\n", malls[i].customers);
            printf("  customers served: %ld\n", malls[i].served);
            printf("       turnaways: %ld\n", malls[i].turnaways);
            printf("average queue size: %.2f\n",
                (double) malls[i].sum_line / malls[i].cyclelimit);
            printf(" average wait time: %.2f minutes\n",
                (double) malls[i].line_wait / malls[i].served);
        }
        else
            puts("No customers!");
        EmptyTheQueue(&malls[i].line);
    }

    puts("Bye!");
    
    return 0;
}

bool newcustomer(double x)
{
    if (rand() * x / RAND_MAX < 1)
        return true;
    else
        return false;
}

Item customertime(long when)
{
    Item cust;
    
    cust.processtime = rand() % 3 + 1;
    cust.arrive = when;
    
    return cust;
}
```

### Q5

```c
// stack.h
#ifndef STACK_H_
#define STACK_H_
#include <stdbool.h>
#define MAXSIZE 100

typedef char Item;

typedef struct stack
{
    Item items[MAXSIZE];
    int top;
} Stack;

void Init(Stack *st);

bool IsEmpty(Stack *st);

bool IsFull(Stack *st);

bool Push(Stack *st, Item val);

bool Pop(Stack *st, Item *val);

void Empty(Stack *st);

#endif
```

```c
// stack.c
#include <stdio.h>
#include "stack.h"

void Init(Stack *st)
{
    st->top = 0;
}

bool IsEmpty(Stack *st)
{
    return st->top == 0;
}

bool IsFull(Stack *st)
{
    return st->top == MAXSIZE;
}

bool Push(Stack *st, Item val)
{
    if (IsFull(st))
    {
        return false;
    }
    else
    {
        (st->items)[st->top] = val;
        (st->top)++;
    }
    return true;
}

bool Pop(Stack *st, Item *val)
{
    if (IsEmpty(st))
    {
        return false;
    }
    else
    {
        (st->top)--;
        *val = (st->items)[st->top];
    }
    return true;
}

void Empty(Stack *st)
{
    st->top = 0;
}
```

```c
// main.c
#include <stdio.h>
#include <string.h>
#include "stack.h"
#define SLEN 81

char *s_gets(char *st, int n);

int main(void)
{
    Stack st;
    Item ch, temp[SLEN];

    Init(&st);
    puts("Please enter a string (EOF to quit):");
    while (s_gets(temp, SLEN) != NULL)
    {
        int i = 0;
        while (temp[i] != '\0')
        {
            Push(&st, temp[i]);
            i++;
        }
        printf("Reversing order:\n");
        while (!IsEmpty(&st))
        {
            Pop(&st, &ch);
            putchar(ch);
        }
        puts("\nYou can enter a string again (EOF to quit):");
    }
    puts("Done.");

    return 0;
}

char *s_gets(char *st, int n)
{
    char *find;
    char *ret_val;

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

### Q6

```c
#include <stdio.h>

#define M 4
#define N 5

int find(int [], int, int);

int main() {
    int a[M] = {1,2,3,4};
    printf("%d", find(a, M, 1));
    printf("%d", find(a, M, 2));
    printf("%d", find(a, M, 3));
    printf("%d", find(a, M, 4));
    int b[N] = {1,2,3,4,5};
    printf("%d", find(b, N, 1));
    printf("%d", find(b, N, 2));
    printf("%d", find(b, N, 3));
    printf("%d", find(b, N, 4));
    printf("%d", find(b, N, 5));
    return 0;
}

int find(int ar[], int n, int num) {
    int l = 0,
        r = n,
        m = (l+r)/2;
    while (ar[m] != num) {
        if (ar[m] < num) {
            l = m;
        }
        else {
            r = m;
        }
        m = (l+r)/2;
    }
    if (ar[m] == num) {
        return 1;
    }
    return 0;
}
```

### Q7

```c
// tree.h
#ifndef _TREE_H_
#define _TREE_H_
#include <stdbool.h>

#define SLEN 20
typedef struct item {
    char word[SLEN];
    int count;
} Item;

#define MAXITEMS 200

typedef struct trnode {
    Item item;
    struct trnode * left;
    struct trnode * right;
} Trnode;
typedef struct tree {
    Trnode * root;
    int size;
} Tree;

void InitializeTree(Tree * ptree);
bool TreeIsEmpty(const Tree * ptree);
bool TreeIsFull(const Tree * ptree);
int TreeItemCount(const Tree * ptree);
bool AddItem(const Item * pi, Tree * ptree);
bool InTree(const Item * pi, const Tree * ptree);
bool DeleteItem(const Item * pi, Tree * ptree);
void Traverse (const Tree * ptree, void (* pfun)(Item item));
void DeleteAll(Tree * ptree);
int GetCount(const Item *pi, const Tree *ptree);

#endif
```

```c
// tree.c
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "tree.h"

typedef struct pair {
    Trnode * parent;
    Trnode * child;
} Pair;

static Trnode * MakeNode(const Item * pi);
static bool ToLeft(const Item * i1, const Item * i2);
static bool ToRight(const Item * i1, const Item * i2);
static void AddNode (Trnode * new_node, Trnode * root);
static void InOrder(const Trnode * root, void (* pfun)(Item item));
static Pair SeekItem(const Item * pi, const Tree * ptree);
static void DeleteNode(Trnode **ptr);
static void DeleteAllNodes(Trnode * ptr);

void InitializeTree(Tree * ptree) {
    ptree->root = NULL;
    ptree->size = 0;
}

bool TreeIsEmpty(const Tree * ptree) {
    return ptree->root == NULL;
}

bool TreeIsFull(const Tree * ptree) {
    return ptree->size == MAXITEMS;
}

int TreeItemCount(const Tree * ptree) {
    return ptree->size;
}

bool AddItem(const Item * pi, Tree * ptree) {
    Trnode * new_node;
    Pair seek;
    
    if  (TreeIsFull(ptree)) {
        fprintf(stderr,"Tree is full\n");
        return false;
    }
    if ((seek = SeekItem(pi, ptree)).child != NULL) {
        seek.child->item.count++;
        return true;
    }
    new_node = MakeNode(pi);
    if (new_node == NULL) {
        fprintf(stderr, "Couldn't create node\n");
        return false;
    }
    ptree->size++;
    
    if (ptree->root == NULL)
        ptree->root = new_node;
    else
        AddNode(new_node,ptree->root);
    
    return true;
}

bool InTree(const Item * pi, const Tree * ptree) {
    return (SeekItem(pi, ptree).child == NULL) ? false : true;
}

bool DeleteItem(const Item * pi, Tree * ptree) {
    Pair look;
    
    look = SeekItem(pi, ptree);
    if (look.child == NULL)
        return false;
    if (look.child->item.count > 0) {
        look.child->item.count--;
    }
    else {
        if (look.parent == NULL)
            DeleteNode(&ptree->root);
        else if (look.parent->left == look.child)
            DeleteNode(&look.parent->left);
        else
            DeleteNode(&look.parent->right);
        ptree->size--;
    }
    
    return true;
}

void Traverse (const Tree * ptree, void (* pfun)(Item item)) {
    
    if (ptree != NULL)
        InOrder(ptree->root, pfun);
}

void DeleteAll(Tree * ptree) {
    if (ptree != NULL)
        DeleteAllNodes(ptree->root);
    ptree->root = NULL;
    ptree->size = 0;
}

int GetCount(const Item *pi, const Tree *ptree) {
    return ((SeekItem(pi, ptree).child)->item).count;
}

static void InOrder(const Trnode * root, void (* pfun)(Item item)) {
    if (root != NULL) {
        InOrder(root->left, pfun);
        (*pfun)(root->item);
        InOrder(root->right, pfun);
    }
}

static void DeleteAllNodes(Trnode * root) {
    Trnode * pright;
    
    if (root != NULL) {
        pright = root->right;
        DeleteAllNodes(root->left);
        free(root);
        DeleteAllNodes(pright);
    }
}

static void AddNode (Trnode * new_node, Trnode * root) {
    if (ToLeft(&new_node->item, &root->item)) {
        if (root->left == NULL)
            root->left = new_node;
        else
            AddNode(new_node, root->left);
    }
    else if (ToRight(&new_node->item, &root->item)) {
        if (root->right == NULL)
            root->right = new_node;
        else
            AddNode(new_node, root->right);
    }
    else {
        fprintf(stderr,"location error in AddNode()\n");
        exit(1);
    }
}

static bool ToLeft(const Item * i1, const Item * i2) {
    return strcmp(i1->word, i2->word) < 0 ? true : false;
}

static bool ToRight(const Item * i1, const Item * i2) {
    return strcmp(i1->word, i2->word) > 0 ? true : false;
}

static Trnode * MakeNode(const Item * pi) {
    Trnode * new_node;
    
    new_node = (Trnode *) malloc(sizeof(Trnode));
    if (new_node != NULL) {
        new_node->item = *pi;
        new_node->item.count = 1;
        new_node->left = NULL;
        new_node->right = NULL;
    }
    
    return new_node;
}

static Pair SeekItem(const Item * pi, const Tree * ptree) {
    Pair look;
    look.parent = NULL;
    look.child = ptree->root;
    
    if (look.child == NULL)
        return look;
    
    while (look.child != NULL) {
        if (ToLeft(pi, &(look.child->item))) {
            look.parent = look.child;
            look.child = look.child->left;
        }
        else if (ToRight(pi, &(look.child->item))) {
            look.parent = look.child;
            look.child = look.child->right;
        }
        else
            break;
    }
    
    return look;
}

static void DeleteNode(Trnode **ptr) {
    Trnode * temp;
    
    if ( (*ptr)->left == NULL) {
        temp = *ptr;
        *ptr = (*ptr)->right;
        free(temp);
    }
    else if ( (*ptr)->right == NULL) {
        temp = *ptr;
        *ptr = (*ptr)->left;
        free(temp);
    }
    else {
        for (temp = (*ptr)->left; temp->right != NULL;
             temp = temp->right)
            continue;
        temp->right = (*ptr)->right;
        temp = *ptr;
        *ptr =(*ptr)->left;
        free(temp); 
    }
}
```

```c
// 17-7.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "tree.h"

#define LEN 20

void eatline(void);
char menu(void);
void printitem (Item item);
void finditem (Tree * ptree);

int main(void) {
    
    Tree words;
    InitializeTree(&words);

    int ch;
    FILE * fp;
    char filename[LEN];

    printf("Input filename: ");
    scanf("%s", filename);
    eatline();
    if ((fp = fopen(filename, "r")) == NULL) {
        printf("Can't open %s\n", filename);
        exit(EXIT_FAILURE);
    }
    Item item;
    int count = 0;
    while ((ch = getc(fp)) != EOF) {
        if (!isalpha(ch) && count == 0) {
            continue;
        }
        if (!isalpha(ch) && count > 0) {
            item.word[count] = '\0';
            item.count = 1;
            count = 0;
            AddItem(&item, &words);
            memset(item.word, 0, sizeof(item.word));
            continue;
        }
        item.word[count++] = tolower(ch);
    }
    fclose(fp);

    char choice;
    while ((choice = menu()) != 'q') {
        switch (choice) {
            case 'a': {
                Traverse(&words, printitem);
                break;
            }
            case 'b': {
                finditem(&words);
                break;
            }
            default: {
                puts("Switching Error");
            }
        }
    }
    puts("Bye!");
    return 0;
}

void eatline(void) {
    while (getchar() != '\n') ;
}

char menu(void) {
    int ch;
    puts("*******************************************************************************");
    puts("选择数字以执行相应功能:");
    puts("a) 列出所有单词和出现的次数            b) 输出指定单词出现的次数");
    puts("q) 退出");
    puts("*******************************************************************************");
    while ((ch = getchar()) != EOF) {
        eatline();
        ch = tolower(ch);
        if (strchr("abq",ch) == NULL) {
            puts("Please enter an a, b, or q:");
        }
        else {
            break;
        }
    }
    if (ch == EOF) {
        ch = 'q';
    }
    return ch;
}

void printitem (Item item) {
    printf("%s %d\n", item.word, item.count);
}

void finditem (Tree * ptree) {
    char s[SLEN];
    printf("Input a word: ");
    scanf("%s", s);
    eatline();
    Item item;
    strcpy(item.word, s);
    if (InTree(&item, ptree)) {
        printf("%d\n", GetCount(&item, ptree));
    }
    else {
        puts("Word does not exist.");
    }
}
```

### Q8

```c
// tree.h
#ifndef _TREE_H_
#define _TREE_H_
#include <stdbool.h>

#define SLEN 20
typedef struct kind {
    char petkind[SLEN];
    struct kind *next;
} Kind;
typedef struct item {
    char petname[SLEN];
    Kind * petkinds;
    int petcount;
} Item;

#define MAXITEMS 10

typedef struct trnode {
    Item item;
    struct trnode * left;
    struct trnode * right;
} Trnode;
typedef struct tree {
    Trnode * root;
    int size;
} Tree;

void InitializeTree(Tree * ptree);
bool TreeIsEmpty(const Tree * ptree);
bool TreeIsFull(const Tree * ptree);
int TreeItemCount(const Tree * ptree);
bool AddItem(const Item * pi, Tree * ptree);
bool InTree(const Item * pi, const Tree * ptree);
bool DeleteItem(const Item * pi, Tree * ptree);
void Traverse (const Tree * ptree, void (* pfun)(Item item));
void DeleteAll(Tree * ptree);

#endif
```

```c
// tree.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tree.h"

typedef struct pair
{
    Trnode *parent;
    Trnode *child;
} Pair;

static Trnode *MakeNode(const Item *pi);
static bool ToLeft(const Item *i1, const Item *i2);
static bool ToRight(const Item *i1, const Item *i2);
static void AddNode(Trnode *new_node, Trnode *root);
static void InOrder(const Trnode *root, void (*pfun)(Item item));
static Pair SeekItem(const Item *pi, const Tree *ptree);
static void DeleteNode(Trnode **ptr);
static void DeleteAllNodes(Trnode *ptr);
static int all_pets_numbers(const Trnode *root);

void InitializeTree(Tree *ptree) {
    ptree->root = NULL;
    ptree->size = 0;
}

bool TreeIsEmpty(const Tree *ptree) {
    return ptree->root == NULL;
}

bool TreeIsFull(const Tree *ptree) {
    return ptree->size == MAXITEMS;
}

int TreeItemCount(const Tree *ptree) {
    return all_pets_numbers(ptree->root);
}

bool AddItem(const Item *pi, Tree *ptree) {
    Trnode *new_node;
    Trnode *find;

    if (TreeIsFull(ptree)) {
        fprintf(stderr, "Tree is full\n");
        return false;
    }
    if ((find = SeekItem(pi, ptree).child) != NULL) {
        Kind *temp;
        Kind *node;
        for (temp = find->item.petkinds; temp != NULL; temp = temp->next) {
            if (strcmp(pi->petkinds->petkind, temp->petkind) == 0) {
                break;
            }
        }
        if (temp != NULL) {
            fprintf(stderr, "Can't add duplicate item!\n");
            return false;
        }
        else {
            if ((node = (Kind *)malloc(sizeof(Kind))) == NULL) {
                fprintf(stderr, "Memory allocation failed!\n");
                return false;
            }
            else {
                strcpy(node->petkind, pi->petkinds->petkind);
                node->next = NULL;
                for (temp = find->item.petkinds; temp->next != NULL; temp = temp->next) {
                    continue;
                }
                temp->next = node;
                find->item.petcount++;
                return true;
            }
        }
    }
    new_node = MakeNode(pi);
    if (new_node == NULL) {
        fprintf(stderr, "Couldn't create node.\n");
        return false;
    }
    ptree->size++;
    if (ptree->root == NULL) {
        ptree->root = new_node;
    }
    else {
        AddNode(new_node, ptree->root);
    }
    return true;
}

bool InTree(const Item *pi, const Tree *ptree) {
    return (SeekItem(pi, ptree).child == NULL) ? false : true;
}

bool DeleteItem(const Item *pi, Tree *ptree) {
    Pair look;
    look = SeekItem(pi, ptree);

    if (look.child == NULL) {
        return false;
    }
    if (look.child->item.petcount != 1) {
        Kind *prior;
        Kind *current;
        for (prior = current = look.child->item.petkinds; current != NULL; current = current->next) {
            if (strcmp(current->petkind, pi->petkinds->petkind) == 0) {
                break;
            }
            prior = current;
        }
        if (current != NULL) {
            prior->next = current->next;
            free(current);
            look.child->item.petcount--;
            return true;
        }
        else {
            return false;
        }
    }
    else {
        if (strcmp(look.child->item.petkinds->petkind, pi->petkinds->petkind) == 0) {
            if (look.parent == NULL) {
                DeleteNode(&ptree->root);
            }
            else if (look.parent->left == look.child) {
                DeleteNode(&look.parent->left);
            }
            else {
                DeleteNode(&look.parent->right);
            }
            ptree->size--;
            return true;
        }
        else {
            return false;
        }
    }
}

void Traverse(const Tree *ptree, void (*pfun)(Item item)) {
    if (ptree != NULL) {
        InOrder(ptree->root, pfun);
    }
}

void DeleteAll(Tree *ptree) {
    if (ptree != NULL) {
        DeleteAllNodes(ptree->root);
    }
    ptree->root = NULL;
    ptree->size = 0;
}

static void InOrder(const Trnode *root, void (*pfun)(Item item)) {
    if (root != NULL) {
        InOrder(root->left, pfun);
        (*pfun)(root->item);
        InOrder(root->right, pfun);
    }
}

static void DeleteAllNodes(Trnode *root) {
    Trnode *pright;

    if (root != NULL) {
        pright = root->right;
        DeleteAllNodes(root->left);
        Kind *temp;
        while (root->item.petkinds != NULL) {
            temp = root->item.petkinds->next;
            free(root->item.petkinds);
            root->item.petkinds = temp;
        }
        free(root);
        DeleteAllNodes(pright);
    }
}

static void AddNode(Trnode *new_node, Trnode *root) {
    if (ToLeft(&new_node->item, &root->item)) {
        if (root->left == NULL) {
            root->left = new_node;
        }
        else {
            AddNode(new_node, root->left);
        }
    }
    else if (ToRight(&new_node->item, &root->item)) {
        if (root->right == NULL) {
            root->right = new_node;
        }
        else {
            AddNode(new_node, root->right);
        }
    }
    else {
        fprintf(stderr, "location error in AddNode()\n");
        exit(1);
    }
}

static bool ToLeft(const Item *i1, const Item *i2) {
    return strcmp(i1->petname, i2->petname) < 0 ? true : false;
}

static bool ToRight(const Item *i1, const Item *i2) {
    return strcmp(i1->petname, i2->petname) > 0 ? true : false;
}

static Trnode *MakeNode(const Item *pi) {
    Trnode *new_node;
    Kind *temp;

    if ((new_node = (Trnode *)malloc(sizeof(Trnode))) == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    if ((temp = (Kind *)malloc(sizeof(Kind))) == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    strcpy(temp->petkind, pi->petkinds->petkind);
    temp->next = NULL;
    strcpy(new_node->item.petname, pi->petname);
    new_node->item.petkinds = temp;
    new_node->item.petcount = 1;
    new_node->left = NULL;
    new_node->right = NULL;
    return new_node;
}

static Pair SeekItem(const Item *pi, const Tree *ptree) {
    Pair look;
    look.parent = NULL;
    look.child = ptree->root;

    if (look.child == NULL)
        return look;

    while (look.child != NULL) {
        if (ToLeft(pi, &(look.child->item))) {
            look.parent = look.child;
            look.child = look.child->left;
        }
        else if (ToRight(pi, &(look.child->item))) {
            look.parent = look.child;
            look.child = look.child->right;
        }
        else {
            break;
        }
    }
    return look;
}

static void DeleteNode(Trnode **ptr) {
    Kind *tp;
    Trnode *temp;

    if ((*ptr)->left == NULL) {
        temp = *ptr;
        *ptr = (*ptr)->right;
    }
    else if ((*ptr)->right == NULL) {
        temp = *ptr;
        *ptr = (*ptr)->left;
    }
    else {
        for (temp = (*ptr)->left; temp->right != NULL; temp = temp->right) {
            continue;
        }
        temp->right = (*ptr)->right;
        temp = *ptr;
        *ptr = (*ptr)->left;
    }
    while (temp->item.petkinds != NULL) {
        tp = temp->item.petkinds->next;
        free(temp->item.petkinds);
        temp->item.petkinds = tp;
    }
    free(temp);
}

static int all_pets_numbers(const Trnode *root) {
    static int count = 0;

    if (root != NULL) {
        all_pets_numbers(root->left);
        count += root->item.petcount;
        all_pets_numbers(root->right);
    }
    return count;
}
```

```c
// 17-8.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "tree.h"

char menu(void);
void addpet(Tree *pt);
void droppet(Tree *pt);
void showpets(const Tree *pt);
void findpet(const Tree *pt);
void printitem(Item item);
void uppercase(char *str);
char *s_gets(char *st, int n);
void printSamePets(Item item, const char *str);
void myTraverse(const Trnode *root, void (*pfun)(Item item, const char *temp), const char *str);
void petsCount(int *count, int *temp);

int main(void) {
    Tree pets;
    char choice;
    int count, temp;
    count = temp = 0;

    InitializeTree(&pets);
    while ((choice = menu()) != 'q') {
        switch (choice) {
            case 'a': {
                addpet(&pets);
                break;
            }
            case 'l': {
                showpets(&pets);
                break;
            }
            case 'f': {
                findpet(&pets);
                break;
            }
            case 'n': {
                count = TreeItemCount(&pets);
                petsCount(&count, &temp);
                printf("%d pets in club\n", count);
                break;
            }
            case 'd': {
                droppet(&pets);
                break;
            }
            default:{
                puts("Switching error");
            }
        }
    }
    DeleteAll(&pets);
    puts("Bye.");

    return 0;
}

char menu(void) {
    int ch;

    puts("Nerfville Pet Club Membership Program");
    puts("Enter the letter corresponding to your choice:");
    puts("a) add a pet          l) show list of pets");
    puts("n) number of pets     f) find pets");
    puts("d) delete a pet       q) quit");
    while ((ch = getchar()) != EOF) {
        while (getchar() != '\n')
            continue;
        ch = tolower(ch);
        if (strchr("alrfndq", ch) == NULL) {
            puts("Please enter an a, l, f, n, d, or q:");
        }
        else {
            break;
        }
    }
    if (ch == EOF) {
        ch = 'q';
    }
    return ch;
}

void addpet(Tree *pt) {
    Item temp;

    if (TreeIsFull(pt)) {
        puts("No room in the club!");
    }
    else {
        puts("Please enter name of pet:");
        s_gets(temp.petname, SLEN);
        temp.petkinds = (Kind *)malloc(sizeof(Kind));
        puts("Please enter pet kind:");
        s_gets(temp.petkinds->petkind, SLEN);
        uppercase(temp.petname);
        uppercase(temp.petkinds->petkind);
        AddItem(&temp, pt);
        free(temp.petkinds);
    }
}

void showpets(const Tree *pt) {
    if (TreeIsEmpty(pt)) {
        puts("No entries!");
    }
    else {
        Traverse(pt, printitem);
    }
}

void printitem(Item item) {
    if (1 == item.petcount) {
        printf("Pet: %-19s  Kind: %-19s\n", item.petname,
               item.petkinds->petkind);
    }
    else {
        Kind *temp = item.petkinds;
        while (temp != NULL) {
            printf("Pet: %-19s  Kind: %-19s\n", item.petname,
                temp->petkind);
            temp = temp->next;
        }
    }
}

void findpet(const Tree *pt) {
    Item temp;

    if (TreeIsEmpty(pt)) {
        puts("No entries!");
        return;
    }
    puts("Please enter name of pet you wish to find:");
    s_gets(temp.petname, SLEN);
    uppercase(temp.petname);
    if (InTree(&temp, pt)) {
        printf("All kinds of the %s pets:\n", temp.petname);
        myTraverse(pt->root, printSamePets, temp.petname);
    }
    else {
        printf("%s is not a member.\n", temp.petname);
    }
}

void droppet(Tree *pt) {
    Item temp;

    if (TreeIsEmpty(pt)) {
        puts("No entries!");
        return;
    }
    puts("Please enter name of pet you wish to delete:");
    s_gets(temp.petname, SLEN);
    temp.petkinds = (Kind *)malloc(sizeof(Kind));
    puts("Please enter pet kind:");
    s_gets(temp.petkinds->petkind, SLEN);
    uppercase(temp.petname);
    uppercase(temp.petkinds->petkind);
    printf("%s the %s ", temp.petname, temp.petkinds->petkind);
    if (DeleteItem(&temp, pt)) {
        printf("is dropped from the club.\n");
    }
    else {
        printf("is not a member.\n");
    }
    free(temp.petkinds);
}

void uppercase(char *str) {
    while (*str) {
        *str = toupper(*str);
        str++;
    }
}

char *s_gets(char *st, int n) {
    char *ret_val;
    char *find;

    ret_val = fgets(st, n, stdin);
    if (ret_val) {
        find = strchr(st, '\n');
        if (find) {
            *find = '\0';
        }
        else {
            while (getchar() != '\n')
                continue;
        }
    }
    return ret_val;
}

void printSamePets(Item item, const char *str) {
    Kind *temp = item.petkinds;

    while (temp != NULL) {
        if (0 == strcmp(item.petname, str)) {
            printf("Pet: %-19s  Kind: %-19s\n", item.petname,
                   temp->petkind);
        }
        temp = temp->next;
    }
}

void myTraverse(const Trnode *root, void (*pfun)(Item item, const char *temp), const char *str) {
    if (root != NULL) {
        myTraverse(root->left, pfun, str);
        (*pfun)(root->item, str);
        myTraverse(root->right, pfun, str);
    }
}

void petsCount(int *count, int *temp) {
    *count -= *temp;
    *temp += *count;
    return;
}
```