#ifndef __LOGGING_H
#define __LOGGING_H
#include <stdio.h>
#define mfopen(file, mode) mfopen_loc(file, mode, __FILE__, __LINE__)
#define mfclose(file) mfclose_loc(file, __FILE__, __LINE__)

typedef enum {
    LOG_FATAL,
    LOG_ERROR,
    LOG_WARNING,
    LOG_INFO,
} logging_level;

void logging_log(logging_level level, const char *fmt, ...);
FILE *mfopen_loc(const char *path, const char *mode, const char *file, int line);
void mfclose_loc(FILE *f, const char *file, int line);

#endif
