#ifndef __PROFILER_H
#define __PROFILER_H

//#ifdef _WIN32
//#define profiler_start_measure(name)
//#define profiler_end_measure(name)
//#define profiler_print_measures(file)
//#endif

#include <stdint.h>
#define PROFILER(x) __PROFILER_##x
#define __PROFILER_TABLE_MAX 1000

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>

typedef struct PROFILER(elem) {
    char* name;
    double time_start, time_end;
    double interval;
    struct PROFILER(elem)* next;
    uint64_t count;
} PROFILER(elem);

PROFILER(elem) PROFILER(table)[__PROFILER_TABLE_MAX] = {0};

bool profiler_start_measure(const char* name);
void profiler_end_measure(const char* name);
void profiler_print_measures(FILE *file);

#endif //__PROFILER_H


#if defined(__PROFILER_IMPLEMENTATION)

static double get_time_sec() {
    struct timespec tmp = {0};
    clock_gettime(CLOCK_MONOTONIC, &tmp);
    return tmp.tv_sec + tmp.tv_nsec * 1.0e-9;
}

static uint64_t hash(const char *name) {
    uint64_t r = 0;
    for (uint64_t i = 0; i < strlen(name); ++i) r += (uint64_t)name[i] + (uint64_t)name[i] * i;
    return r % __PROFILER_TABLE_MAX;
}

static PROFILER(elem) initelem(const char* name) {
    PROFILER(elem) ret = {0};
    uint64_t len = strlen(name);
    ret.name = (char*)calloc(len + 1, 1);
    memcpy(ret.name, name, len);
    ret.name[len] = '\0';
    ret.next = NULL;
    ret.time_start = get_time_sec();
    ret.count++;
    return ret;
}

static bool insert(const char* name) {
    uint64_t index = hash(name);
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

void profiler_end_measure(const char* name) {
    uint64_t index = hash(name);
    PROFILER(elem) *head = &PROFILER(table)[index];
    while (head) {
      if (strcmp(head->name, name) == 0) {
          head->time_end = get_time_sec();
          head->interval += head->time_end - head->time_start;
          break;
      }
          head = head->next;
    }
}

static void profiler_free_list(PROFILER(elem) *head) {
    if (!head) return;
    
    profiler_free_list(head->next);
    if (head->name) free(head->name);
    free(head);
}

void profiler_print_measures(FILE *file) {
    for (uint64_t i = 0; i < __PROFILER_TABLE_MAX; ++i) {
        PROFILER(elem) *head = &PROFILER(table)[i];
        if (!head->name) continue;
    
        while (head) {
            fprintf(file, "[ %s ] -> %.9e sec\n", head->name, head->interval / head->count);
            head = head->next;
        }
    
        head = &PROFILER(table)[i];
        profiler_free_list(head->next);
	
	free(head->name);
        memset(head, 0, sizeof(PROFILER(elem)));
    }
}

#endif //__PROFILER_IMPLEMENTATION
