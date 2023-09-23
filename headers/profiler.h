#ifndef __PROFILER_H
#define __PROFILER_H

#if _POSIX_C_SOURCE < 199309L
#define _POSIX_C_SOURCE 199310L
#endif

#define PROFILER(x) __PROFILER_##x
#define __PROFILER_TABLE_MAX 1000

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>


typedef struct PROFILER(elem) {
    char* name;
    struct timespec T1, T2;
    double interval;
    struct PROFILER(elem)* next;
} PROFILER(elem);

PROFILER(elem) PROFILER(table)[__PROFILER_TABLE_MAX] = {0};

bool profiler_start_measure(const char* name);
void EndMeasure(const char* name);
void DumpMeasures(FILE *file);

#endif //__PROFILER_H


#ifdef __PROFILER_IMPLEMENTATION


static size_t hash(const char *name) {
    size_t r = 0;
    for (size_t i = 0; i < strlen(name); ++i) r += (size_t)name[i] + (size_t)name[i] * i;
    return r % __PROFILER_TABLE_MAX;
}

static PROFILER(elem) initelem(const char* name) {
    PROFILER(elem) ret = {0};
    size_t len = strlen(name);
    ret.name = (char*)calloc(len + 1, 1);
    memcpy(ret.name, name, len);
    ret.name[len] = '\0';
    ret.next = NULL;
    clock_gettime(CLOCK_REALTIME, &ret.T1);
    ret.T2 = (struct timespec){0};
    return ret;
}

static bool insert(const char* name) {
    size_t index = hash(name);
    if (!PROFILER(table)[index].name) {
        PROFILER(table)[index] = initelem(name);
        return true;
    }
  
    PROFILER(elem) *head = &PROFILER(table)[index];
    PROFILER(elem) *last = &PROFILER(table)[index];
    while (head) {
        if (strcmp(head->name, name) == 0) return false;
        last = head;
        head = head->next;
    }
  
    last->next = calloc(1, sizeof(PROFILER(elem)));  
    last = last->next;
    if (!last) return false;
    *last = initelem(name);
  
    return true;
}

bool profiler_start_measure(const char *name) {
    return insert(name);
}

void EndMeasure(const char* name) {
    size_t index = hash(name);
    PROFILER(elem) *head = &PROFILER(table)[index];
    while (head) {
      if (strcmp(head->name, name) == 0) {
          clock_gettime(CLOCK_REALTIME, &head->T2);
          head->interval = (double)(head->T2.tv_sec - head->T1.tv_sec) +
  			           (head->T2.tv_nsec - head->T1.tv_nsec) * 1.0e-9;
          break;
      }
          head = head->next;
    }
}

static void FreeList(PROFILER(elem) *head) {
    if (!head) return;
    
    FreeList(head->next);
    if (head->name) free(head->name);
    free(head);
}

void DumpMeasures(FILE *file) {
    for (size_t i = 0; i < __PROFILER_TABLE_MAX; ++i) {
        PROFILER(elem) *head = &PROFILER(table)[i];
        if (!head->name) continue;
    
        while (head) {
            fprintf(file, "[ %s ] -> %.9e sec\n", head->name, head->interval);
            head = head->next;
        }
    
        head = &PROFILER(table)[i];
        FreeList(head->next);
	
	free(head->name);
        memset(head, 0, sizeof(PROFILER(elem)));
    }
}

#endif //__PROFILER_IMPLEMENTATION
