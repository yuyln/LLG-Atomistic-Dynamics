#ifndef __LOGGING_H
#define __LOGGING_H
#include <stdio.h>

typedef enum {
    LOG_FATAL,
    LOG_ERROR,
    LOG_WARNING,
    LOG_INFO,
} logging_level;

void logging_log(logging_level level, const char *fmt, ...);

#endif
