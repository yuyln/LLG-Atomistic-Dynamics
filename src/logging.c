#include "logging.h"
#include <stdbool.h>
#include <stdarg.h>
#include <stdlib.h>

void logging_log(FILE *f, logging_level level, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    bool terminate = false;
    switch (level) {
        case LOG_FATAL:
            fprintf(f, "[ FATAL ] ");
            terminate = true;
            break;
        case LOG_ERROR:
            fprintf(f, "[ ERROR ] ");
            break;
        case LOG_WARNING:
            fprintf(f, "[ WARNING ] ");
            break;
        case LOG_INFO:
            fprintf(f, "[ INFO ] ");
            break;
        default: {} break;
    }
    vfprintf(f, fmt, args);
    fprintf(f, "\n");
    va_end(args);
    if (terminate)
        exit(1);
}
