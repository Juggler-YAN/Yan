#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define LEN 100

int main(void) {
    char **pt;
    char temp[LEN];
    int n, len;

    printf("How many words do you wish to enter? ");
    scanf("%d", &n);
    pt = (char **) malloc(n * sizeof(char *));
    if (pt == NULL) {
        printf("Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    printf("Enter %d words now: \n", n);
    for (int i = 0; i < n; i++) {
        scanf("%99s", temp);
        len = strlen(temp) + 1;
        pt[i] = (char *)malloc(len * sizeof(char));
        if (pt[i] == NULL) {
            printf("Memory allocation failed!\n");
            exit(EXIT_FAILURE);
        }
        strcpy(pt[i], temp);
    }
    printf("Here are your words:\n");
    for (int i = 0; i < n; i++) {
        puts(pt[i]);
        free(pt[i]);
    }
    free(pt);
    return 0;
}