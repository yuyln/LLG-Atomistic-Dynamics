#include <stdint.h>

#include "gpu.h"
#include "constants.h"
#include "grid_funcs.h"
#include "string_view.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//@TODO: Change openclwrapper to print file and location correctly
//@TODO: Check uint64_t->int changes
//@TODO: Optimize for 3D->2D
int main(void) {

    gpu_cl gpu = gpu_cl_init(0, 0);
    char *kernel;
    clw_read_file("./kernel_complete.cl", &kernel);
    const char cmp[] = "-DOPENCL_COMPILATION -DNBULK";
    string_view kernel_view = (string_view){.str = kernel, .len = strlen(kernel)};
    string_view compile_opt = (string_view){.str = cmp, .len = strlen(cmp) + 1};
    gpu_cl_compile_source(&gpu, kernel_view, compile_opt);
    free(kernel);
    uint64_t k1 = gpu_append_kernel(&gpu, "gpu_step");
    uint64_t k2 = gpu_append_kernel(&gpu, "exchange_grid");
    uint64_t k3 = gpu_append_kernel(&gpu, "v3d_to_rgb");


    grid g = grid_init(32, 32);
    v3d_fill_with_random(g.m, g.gi.rows, g.gi.cols);
    grid_set_anisotropy(&g, (anisotropy){.ani = 0.02 * QE * 1.0e-3, .dir=v3d_c(0, 0, 1)});
    grid_to_gpu(&g, gpu);
    g.gi.pbc.dirs = (1 << 1);

    cl_mem new_buffer = clw_create_buffer(g.gi.rows * g.gi.cols * sizeof(*g.m), gpu.ctx, CL_MEM_READ_WRITE);
    cl_mem rgb_buffer = clw_create_buffer(g.gi.rows * g.gi.cols * sizeof(cl_char4), gpu.ctx, CL_MEM_READ_WRITE);


    double dt = 1.0e-15;
    double time = 0;

    clw_set_kernel_arg(gpu.kernels[k1], 0, sizeof(cl_mem), &g.gp_buffer);
    clw_set_kernel_arg(gpu.kernels[k1], 1, sizeof(cl_mem), &g.m_buffer);
    clw_set_kernel_arg(gpu.kernels[k1], 2, sizeof(cl_mem), &new_buffer);
    clw_set_kernel_arg(gpu.kernels[k1], 3, sizeof(double), &dt);
    clw_set_kernel_arg(gpu.kernels[k1], 4, sizeof(double), &time);
    clw_set_kernel_arg(gpu.kernels[k1], 5, sizeof(grid_info), &g.gi);

    clw_set_kernel_arg(gpu.kernels[k2], 0, sizeof(cl_mem), &g.m_buffer);
    clw_set_kernel_arg(gpu.kernels[k2], 1, sizeof(cl_mem), &new_buffer);

    clw_set_kernel_arg(gpu.kernels[k3], 0, sizeof(cl_mem), &g.m_buffer);
    clw_set_kernel_arg(gpu.kernels[k3], 1, sizeof(cl_mem), &rgb_buffer);

    size_t global = g.gi.rows * g.gi.cols;
    size_t local = clw_gcd(global, 32);

    uint32_t *rgba = calloc(sizeof(uint32_t) * g.gi.cols * g.gi.rows, 1);
    clw_enqueue_nd(gpu.queue, gpu.kernels[k3], 1, NULL, &global, &local);
    clw_read_buffer(rgb_buffer, rgba, sizeof(uint32_t) * g.gi.cols * g.gi.rows, 0, gpu.queue);
    stbi_write_png("before.png", g.gi.cols, g.gi.rows, 4, rgba, g.gi.cols * sizeof(uint32_t));

    printf("Start\n");
    while (time <= 1 * NS) {
        clw_enqueue_nd(gpu.queue, gpu.kernels[k1], 1, NULL, &global, &local);
        clw_enqueue_nd(gpu.queue, gpu.kernels[k2], 1, NULL, &global, &local);
        time += dt;
        //printf("%e\n", time);
    }
    printf("End\n");


    clw_enqueue_nd(gpu.queue, gpu.kernels[k3], 1, NULL, &global, &local);
    clw_read_buffer(rgb_buffer, rgba, sizeof(uint32_t) * g.gi.cols * g.gi.rows, 0, gpu.queue);
    stbi_write_png("after.png", g.gi.cols, g.gi.rows, 4, rgba, g.gi.cols * sizeof(uint32_t));
    


    clw_print_cl_error(stderr, clReleaseMemObject(new_buffer), "[ FATAL ] Could not release new buffer from GPU");
    clw_print_cl_error(stderr, clReleaseMemObject(rgb_buffer), "[ FATAL ] Could not release rgb buffer from GPU");

    grid_free(&g);
    grid_release_from_gpu(&g);
    gpu_cl_close(&gpu);
    free(rgba);
    return 0;
}
