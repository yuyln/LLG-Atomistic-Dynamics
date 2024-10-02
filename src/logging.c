#include "logging.h"
#include <stdbool.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

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

FILE *mfopen_loc(const char *path, const char *mode, const char *file, int line) {
    FILE *ret = fopen(path, mode);
    if (!ret) {
        logging_log(LOG_ERROR, "Could not open file \"%s\": %s", path, strerror(errno));
        return NULL;
    }
    logging_log(LOG_INFO, "%s:%d Opened file \"%s\" with mode \"%s\". File ptr: 0x%016X", file, line, path, mode, (void*)ret);
    return ret;
}

void mfclose_loc(FILE *f, const char *file, int line) {
    logging_log(LOG_INFO, "%s:%d Closed file with ptr 0x%016X", file, line, (void*)f);
    fclose(f);
}
