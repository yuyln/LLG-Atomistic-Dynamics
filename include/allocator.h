//Dumbest thing I ever did I think
#ifndef __ALLOCATOR_H
#define __ALLOCATOR_H

#define mmalloc(size) malloc_loc(size, __FILE__, __LINE__)
#define mrealloc(ptr, size) realloc_loc(ptr, size, __FILE__, __LINE__)
#define mfree(ptr) free(ptr) //this doesnt fail ever

void *malloc_loc(unsigned int size, const char *file, int line);
void *realloc_loc(void *ptr, unsigned int size, const char *file, int line);
#endif

#ifdef __ALLOCATOR_C
#include "logging.h"
#include <stdlib.h>

void *malloc_loc(unsigned int size, const char *file, int line) {
    void *ret = calloc(size, 1);
    if (!ret)
        logging_log(LOG_FATAL, "%s:%d What the actual fuck. Buy more RAM I guess, lol", file, line);
    return ret;
}

void *realloc_loc(void *ptr, unsigned int size, const char *file, int line) {
    void *ret = realloc(ptr, size);
    if (!ret)
        logging_log(LOG_FATAL, "%s:%d What the actual fuck. Buy more RAM I guess, lol", file, line);
    return ret;
}
#endif
