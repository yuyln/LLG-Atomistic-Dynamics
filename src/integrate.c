#include "integrate.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void integrate_vars(grid *g, integration_params param) {
    integrate_base(g, param.dt, param.duration, param.interval_for_information, param.interval_for_writing_grid, param.current_generation_function, param.field_generation_function, param.output_path);
}

void integrate_base(grid *g, double dt, double duration, unsigned int interval_info, unsigned int interval_grid, string_view func_current, string_view func_field, string_view dir_out) {
    UNUSED(interval_info);
    UNUSED(func_field);
    UNUSED(func_current);
    UNUSED(interval_grid);
    UNUSED(dir_out);

    gpu_cl gpu = gpu_cl_init(0, 0);

    char *kernel;
    clw_read_file("./kernel_complete.cl", &kernel);

    const char cmp[] = "-DOPENCL_COMPILATION";
    string_view kernel_view = sv_from_cstr(kernel);
    string_view compile_opt = sv_from_cstr(cmp);

    gpu_cl_compile_source(&gpu, kernel_view, compile_opt);

    free(kernel);

    uint64_t step_id = gpu_append_kernel(&gpu, "gpu_step");
    uint64_t exchange_id = gpu_append_kernel(&gpu, "exchange_grid");

    grid_to_gpu(g, gpu);

    cl_mem swap_buffer = clw_create_buffer(g->gi.rows * g->gi.cols * sizeof(*g->m), gpu.ctx, CL_MEM_READ_WRITE);

    double time = 0.0;

    clw_set_kernel_arg(gpu.kernels[step_id], 0, sizeof(cl_mem), &g->gp_buffer);
    clw_set_kernel_arg(gpu.kernels[step_id], 1, sizeof(cl_mem), &g->m_buffer);
    clw_set_kernel_arg(gpu.kernels[step_id], 2, sizeof(cl_mem), &swap_buffer);
    clw_set_kernel_arg(gpu.kernels[step_id], 3, sizeof(double), &dt);
    clw_set_kernel_arg(gpu.kernels[step_id], 4, sizeof(double), &time);
    clw_set_kernel_arg(gpu.kernels[step_id], 5, sizeof(grid_info), &g->gi);

    clw_set_kernel_arg(gpu.kernels[exchange_id], 0, sizeof(cl_mem), &g->m_buffer);
    clw_set_kernel_arg(gpu.kernels[exchange_id], 1, sizeof(cl_mem), &swap_buffer);

    size_t global = g->gi.rows * g->gi.cols;
    size_t local = clw_gcd(global, 32);

    uint64_t step = 0;
    while (time <= duration) {
        integrate_step(time, &gpu, step_id, exchange_id, global, local);
        time += dt;
        step++;
    }

    clw_print_cl_error(stderr, clReleaseMemObject(swap_buffer), "[ FATAL ] Could not release swap buffer from GPU");

    grid_release_from_gpu(g);
    gpu_cl_close(&gpu);
}

void integrate_step(double time, gpu_cl *gpu, uint64_t step_id, uint64_t exchange_id, uint64_t global, uint64_t local) {
    clw_set_kernel_arg(gpu->kernels[step_id], 4, sizeof(double), &time);
    clw_enqueue_nd(gpu->queue, gpu->kernels[step_id], 1, NULL, &global, &local);
    clw_enqueue_nd(gpu->queue, gpu->kernels[exchange_id], 1, NULL, &global, &local);
}
