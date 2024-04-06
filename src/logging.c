#include "logging.h"
#include <stdbool.h>
#include <stdarg.h>
#include <stdlib.h>

void logging_log(logging_level level, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    bool terminate = false;
    FILE *f = stdout;
    switch (level) {
        case LOG_FATAL:
            f = stderr;
            fprintf(f, "[ FATAL ] ");
            terminate = true;
            break;
        case LOG_ERROR:
            f = stderr;
            fprintf(f, "[ ERROR ] ");
            break;
        case LOG_WARNING:
            f = stderr;
            fprintf(f, "[ WARNING ] ");
            break;
        case LOG_INFO:
            fprintf(f, "[ INFO ] ");
            break;
        default: {} break;
    }
    if (fmt)
        vfprintf(f, fmt, args);

    fprintf(f, "\n");
    va_end(args);
    if (terminate)
        exit(1);
}
