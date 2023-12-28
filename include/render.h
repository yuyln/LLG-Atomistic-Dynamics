#ifndef __RENDER_H
#define __RENDER_H
#include <stdint.h>
#include <stdbool.h>
#include "openclwrapper.h"
#include "integrate.h"

typedef union {
    struct { uint8_t r, g, b, a; };
    cl_char4 cl_rgba;
    uint32_t rgba;
} RGBA32;

typedef struct simulation_window simulation_window;

typedef struct {
    bool key_pressed[256];
} window_input;

simulation_window *window_init(unsigned int width, unsigned int height);
bool window_should_close(simulation_window *window);
void window_poll(simulation_window *window);
void window_close(simulation_window *w);
bool window_key_pressed(simulation_window *w, char k);

#endif
