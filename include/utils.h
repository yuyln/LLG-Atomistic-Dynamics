#ifndef __UTILS_H
#define __UTILS_H
#include "grid_funcs.h"
#include "allocator.h"

#ifndef MAX_STRS
#define MAX_STRS 256
#endif

#ifndef MAX_STR_LEN
#define MAX_STR_LEN 2048
#endif

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

bool organize_clusters_inplace(const char *clusters_file_path, double sample_dx, double sample_dy, double sample_dz, double distance_square_invalid);
bool organize_clusters(const char *clusters_file_path, const char *clusters_output_path, double sample_dx, double sample_dy, double sample_dz, double distance_square_invalid);

const char *str_fmt_tmp(const char *fmt, ...);

double min_double(double a, double b);
double max_double(double a, double b);
bool barycentric_4pts(v3d p0, v3d p1, v3d p2, v3d p3, double x, double y, double z, double *b0, double *b1, double *b2, double *b3);
bool barycentric_8pts(v3d p0, v3d p1, v3d p2, v3d p3, v3d p4, v3d p5, v3d p6, v3d p7, double x, double y, double z, double *b0, double *b1, double *b2, double *b3, double *b4, double *b5, double *b6, double *b7);
#endif
