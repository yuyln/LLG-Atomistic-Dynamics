#ifndef __COLORS_H
#define __COLORS_H
#include "v3d.h"
#include "constants.h"
#include <stdint.h>
#include <math.h>

#ifdef OPENCL_COMPILATION
#define uint32_t uchar4
#endif

typedef union {
    struct { unsigned char b, g, r, a; };
    struct { unsigned char x, y, z, w; };
    //cl_char4 cl_rgba;
    uint32_t bgra;
} RGBA32;

RGBA32 linear_mapping(double t, v3d start, v3d middle, v3d end);
RGBA32 hsl_to_rgb(double h, double s, double l);

#endif
