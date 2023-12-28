#ifndef __RENDER_H
#define __RENDER_H
#include <stdint.h>
#include "openclwrapper.h"
#include "integrate.h"

typedef union {
    struct { uint8_t r, g, b, a; };
    cl_char4 cl_rgba;
    uint32_t rgba;
} RGBA32;

typedef struct simulation_window simulation_window;
simulation_window render_init(unsigned int width, unsigned int height);

#endif
