#ifndef __RENDER_H
#define __RENDER_H
#include <stdint.h>
#include <stdbool.h>

typedef union {
    struct { uint8_t b, g, r, a; };
    //cl_char4 cl_rgba;
    uint32_t bgra;
} RGBA32;

typedef struct render_window render_window;

typedef struct {
    bool key_pressed[256];
} window_input;

render_window *window_init(unsigned int width, unsigned int height);
bool window_should_close(render_window *window);
void window_poll(render_window *window);
void window_close(render_window *window);
bool window_key_pressed(render_window *window, char c);
void window_render(render_window *window);
void window_draw_from_bytes(render_window *window, RGBA32 *bytes, int x, int y, int width, int height);
int window_width(render_window *window);
int window_height(render_window *window);

#endif
