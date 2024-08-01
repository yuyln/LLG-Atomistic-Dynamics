#ifndef __UTILS_H
#define __UTILS_H
#include "string_view.h"
#include "grid_funcs.h"
#include "allocator.h"

#define da_append(da, item) do { \
    if ((da)->len >= (da)->cap) { \
        if ((da)->cap <= 1) \
            (da)->cap = (da)->len + 2;\
        else\
            (da)->cap *= 1.5; \
        (da)->items = realloc((da)->items, sizeof(*(da)->items) * (da)->cap); \
        if (!(da)->items) \
            logging_log(LOG_FATAL, "%s:%d Could not append item to dynamic array. Allocation failed. Buy more RAM I guess, lol", __FILE__, __LINE__); \
        memset(&(da)->items[(da)->len], 0, sizeof(*(da)->items) * ((da)->cap - (da)->len));\
    } \
    (da)->items[(da)->len] = (item); \
    (da)->len += 1; \
} while(0)

#define da_remove(da, idx) do { \
    if ((idx) >= (da)->len || (idx) < 0) { \
        logging_log(LOG_ERROR, "%s:%d Trying to remove out of range idx %"PRIi64" from dynamic array", __FILE__, __LINE__, (s64)idx); \
        break; \
    } \
    memmove(&(da)->items[(idx)], &(da)->items[(idx) + 1], sizeof(*(da)->items) * ((da)->len - (idx) - 1)); \
    if ((da)->len <= (da)->cap / 2) { \
        (da)->cap /= 1.5; \
        (da)->items = realloc((da)->items, sizeof(*(da)->items) * (da)->cap); \
        if (!(da)->items) \
            logging_log(LOG_FATAL, "%s:%d Could not append item to dynamic array. Allocation failed. Buy more RAM I guess, lol", __FILE__, __LINE__); \
        memset(&(da)->items[(da)->len], 0, sizeof(*(da)->items) * ((da)->cap - (da)->len));\
    } \
    (da)->len -= 1;\
} while(0)

#define rb_at(rb, idx) ((rb)->items[((idx) + (rb)->start) % (rb)->cap])

#define rb_append(rb, item) do {\
    rb_at((rb), (rb)->len) = (item); \
    (rb)->len += 1; \
} while (0)

bool organize_clusters_inplace(const char *clusters_file_path, double sample_dx, double sample_dy, double distance_square_invalid, bool has_size);
bool organize_clusters(const char *clusters_file_path, const char *clusters_output_path, double sample_dx, double sample_dy, double distance_square_invalid, bool has_size);

#endif
