#ifndef __PROFILER_H
#define __PROFILER_H

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#define PROFILER(x) __PROFILER_##x
#define __PROFILER_TABLE_MAX 10

typedef struct PROFILER(elem) {
    char *name;
    double time_start, time_end;
    double interval;
    struct PROFILER(elem)* next;
    uint64_t count;
} PROFILER(elem);

static PROFILER(elem) PROFILER(table)[__PROFILER_TABLE_MAX];

bool profiler_start_measure(const char* name);
void profiler_end_measure(const char* name);
void profiler_print_measures(FILE *file);
double profiler_get_sec();


#endif //__PROFILER_H


#ifdef __PROFILER_IMPLEMENTATION
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <time.h>
LARGE_INTEGER getFILETIMEoffset() {
    SYSTEMTIME s;
    FILETIME f;
    LARGE_INTEGER t;

    s.wYear = 1970;
    s.wMonth = 1;
    s.wDay = 1;
    s.wHour = 0;
    s.wMinute = 0;
    s.wSecond = 0;
    s.wMilliseconds = 0;
    SystemTimeToFileTime(&s, &f);
    t.QuadPart = f.dwHighDateTime;
    t.QuadPart <<= 32;
    t.QuadPart |= f.dwLowDateTime;
    return (t);
}

//https://stackoverflow.com/questions/5404277/porting-clock-gettime-to-windows
int clock_gettime(int X, struct timespec *tv) {
    LARGE_INTEGER t;
    FILETIME f;
    double microseconds;
    static LARGE_INTEGER offset;
    static double frequencyToMicroseconds;
    static int initialized = 0;
    static BOOL usePerformanceCounter = 0;

    if (!initialized) {
        LARGE_INTEGER performanceFrequency;
        initialized = 1;
        usePerformanceCounter = QueryPerformanceFrequency(&performanceFrequency);
        if (usePerformanceCounter) {
            QueryPerformanceCounter(&offset);
            frequencyToMicroseconds = (double)performanceFrequency.QuadPart / 1000000.;
        } else {
            offset = getFILETIMEoffset();
            frequencyToMicroseconds = 10.;
        }
    }
    if (usePerformanceCounter)
        QueryPerformanceCounter(&t);
    else {
        GetSystemTimeAsFileTime(&f);
        t.QuadPart = f.dwHighDateTime;
        t.QuadPart <<= 32;
        t.QuadPart |= f.dwLowDateTime;
    }

    t.QuadPart -= offset.QuadPart;
    microseconds = (double)t.QuadPart / frequencyToMicroseconds;
    t.QuadPart = microseconds;
    tv->tv_sec = t.QuadPart / 1000000;
    tv->tv_nsec = (t.QuadPart % 1000000) * 1000;
    return 0;
}

#else
#include <time.h>
#endif //_WIN32

double profiler_get_sec() {
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
    ret.time_start = profiler_get_sec();
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
          head->time_end = profiler_get_sec();
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
