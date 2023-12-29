#include "grid_render.h"

grid_renderer grid_renderer_init(grid *g, render_window *window) {
    grid_renderer ret = {0};
    int width = window_width(window);
    int height = window_height(window);
    ret.window = window;
    ret.g = g;
    ret.gpu = gpu_cl_init(0, 0);
    ret.rgba_cpu = calloc(width * height, sizeof(*ret.rgba_cpu));
    cl_int err;
    ret.rgba_gpu = clCreateBuffer(ret.gpu.ctx, CL_MEM_READ_WRITE, sizeof(*ret.rgba_cpu) * width * height, NULL, &err);
    clw_print_cl_error(stderr, err, "[ FATAL ] Could not create RGBA buffer on gpu");
    return ret;
}

void grid_renderer_close(grid_renderer *gr) {
    window_close(gr->window);
    free(gr->rgba_cpu);
    clw_print_cl_error(stderr, clReleaseMemObject(gr->rgba_gpu), "[ FATAL ] Could not release RGBA buffer from gpu");
    gpu_cl_close(&gr->gpu);
}
