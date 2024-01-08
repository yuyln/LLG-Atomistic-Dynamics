#ifndef __RENDER_H
#define __RENDER_H
#include <stdint.h>
#include <stdbool.h>
#include "colors.h"


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
void window_resize(render_window *window); //@TODO

#endif
