#ifndef __GRID_RENDER_H
#define __GRID_RENDER_H
#include "render.h"
#include "grid_funcs.h"
#include "gpu.h"
#include "openclwrapper.h"

typedef struct {
    grid *g;
    gpu_cl gpu;

    cl_mem rgba_gpu;
    RGBA32 *rgba_cpu;

    render_window *window;
} grid_renderer;

grid_renderer grid_renderer_init(grid *g, render_window *window);
void grid_renderer_close(grid_renderer *gr);

#endif
