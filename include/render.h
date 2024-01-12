#ifndef __RENDER_H
#define __RENDER_H
#include <stdint.h>
#include <stdbool.h>
#include "colors.h"


typedef struct render_window render_window;

typedef struct {
    bool key_pressed[256];
} window_input;

void window_init(const char *name, unsigned int width, unsigned int height);
bool window_should_close(void);
void window_poll(void);
//void window_close(void);
bool window_key_pressed(char c);
void window_render(void);
void window_draw_from_bytes(RGBA32 *bytes, int x, int y, int width, int height);
int window_width(void);
int window_height(void);

#endif
