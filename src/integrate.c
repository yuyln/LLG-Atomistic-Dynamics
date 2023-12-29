#include "integrate.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void integrate_vars(grid *g, integration_params param) {
    integrate_base(g, param.dt, param.duration, param.interval_for_information, param.interval_for_writing_grid, param.current_generation_function, param.field_generation_function, param.output_path, param.kernel_augment, param.compile_augment);
}

void integrate_base(grid *g, double dt, double duration, unsigned int interval_info, unsigned int interval_grid, string_view func_current, string_view func_field, string_view dir_out, string_view kernel_augment, string_view compile_augment) {
    UNUSED(interval_info);
    UNUSED(func_field);
    UNUSED(func_current);
    UNUSED(interval_grid);
    UNUSED(dir_out);
    UNUSED(kernel_augment);
    UNUSED(compile_augment);

    gpu_cl gpu = gpu_cl_init(0, 0);

    const char cmp[] = "-DOPENCL_COMPILATION";
    string_view compile_opt = sv_from_cstr(cmp);

    gpu_cl_compile_source(&gpu, sv_from_cstr(complete_kernel), compile_opt);


    uint64_t step_id = gpu_append_kernel(&gpu, "gpu_step");
    uint64_t exchange_id = gpu_append_kernel(&gpu, "exchange_grid");
    uint64_t info_id = gpu_append_kernel(&gpu, "extract_info");

    grid_to_gpu(g, gpu);

    cl_mem swap_buffer = clw_create_buffer(g->gi.rows * g->gi.cols * sizeof(*g->m), gpu.ctx, CL_MEM_READ_WRITE);

    information_packed *info = calloc(g->gi.rows * g->gi.cols, sizeof(information_packed));
    cl_mem info_buffer = clw_create_buffer(g->gi.rows * g->gi.cols * sizeof(*info), gpu.ctx, CL_MEM_READ_WRITE);

    double time = 0.0;

    gpu_fill_kernel_args(&gpu, step_id, 0, 6, &g->gp_buffer, sizeof(cl_mem),
                                              &g->m_buffer, sizeof(cl_mem),
                                              &swap_buffer, sizeof(cl_mem),
                                              &dt, sizeof(double),
                                              &time, sizeof(double),
                                              &g->gi, sizeof(grid_info));

    gpu_fill_kernel_args(&gpu, info_id, 0, 7, &g->gp_buffer, sizeof(cl_mem),
                                              &g->m_buffer, sizeof(cl_mem),
                                              &swap_buffer, sizeof(cl_mem),
                                              &info_buffer, sizeof(cl_mem),
                                              &dt, sizeof(double),
                                              &time, sizeof(double),
                                              &g->gi, sizeof(grid_info));

    gpu_fill_kernel_args(&gpu, exchange_id, 0, 2, &g->m_buffer, sizeof(cl_mem),
                                                  &swap_buffer, sizeof(cl_mem));

    size_t global = g->gi.rows * g->gi.cols;
    size_t local = clw_gcd(global, 32);

    uint64_t step = 0;

    FILE *output_info = fopen(dir_out.str, "w");
    fprintf(output_info, "time(s),energy(eV),exchange_energy(eV),dm_energy(eV),field_energy(eV),anisotropy_energy(eV),cubic_anisotropy_energy(eV),");
    fprintf(output_info, "charge_finite,charge_lattice,");
    fprintf(output_info, "avg_mx,avg_my,avg_mz,");
    fprintf(output_info, "eletric_x,eletric_y,eletric_z,");
    fprintf(output_info, "magnetic_lattice_x,magnetic_lattice_y,magnetic_lattice_z,");
    fprintf(output_info, "magnetic_derivative_x,magnetic_derivative_y,magnetic_derivative_z\n");


    while (time <= duration) {
        integrate_step(time, &gpu, step_id, global, local);

        if (step % interval_info == 0) {
            integrate_get_info(time, &gpu, info_id, global, local);
            clw_print_cl_error(stderr, clEnqueueReadBuffer(gpu.queue, info_buffer, CL_TRUE, 0, sizeof(*info) * g->gi.rows * g->gi.cols, info, 0, NULL, NULL), "[ FATAL ] Could not read info_buiffer");
            information_packed info_local = {0};
            for (uint64_t i = 0; i < g->gi.rows * g->gi.cols; ++i) {
                info_local.energy += info[i].energy;
                info_local.cubic_energy += info[i].cubic_energy;
                info_local.anisotropy_energy += info[i].anisotropy_energy;
                info_local.field_energy += info[i].field_energy;
                info_local.dm_energy += info[i].dm_energy;
                info_local.exchange_energy += info[i].exchange_energy;
                info_local.charge_finite += info[i].charge_finite;
                info_local.charge_lattice += info[i].charge_lattice;
                info_local.avg_m = v3d_sum(info_local.avg_m, v3d_scalar(info[i].avg_m, 1.0 / (g->gi.rows * g->gi.cols)));
                info_local.eletric_field = v3d_sum(info_local.eletric_field , info[i].eletric_field);
                info_local.magnetic_field_lattice = v3d_sum(info_local.magnetic_field_lattice, info[i].magnetic_field_lattice);
                info_local.magnetic_field_derivative = v3d_sum(info_local.magnetic_field_derivative, info[i].magnetic_field_derivative);
            }
            fprintf(output_info, "%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,", time, info_local.energy, info_local.exchange_energy, info_local.dm_energy, info_local.field_energy, info_local.anisotropy_energy, info_local.cubic_energy);
            fprintf(output_info, "%.15e,%.15e,", info_local.charge_finite, info_local.charge_lattice);
            fprintf(output_info, "%.15e,%.15e,%.15e,", info_local.avg_m.x, info_local.avg_m.y, info_local.avg_m.z);
            fprintf(output_info, "%.15e,%.15e,%.15e,", info_local.eletric_field.x, info_local.eletric_field.y, info_local.eletric_field.z);
            fprintf(output_info, "%.15e,%.15e,%.15e,", info_local.magnetic_field_lattice.x, info_local.magnetic_field_lattice.y, info_local.magnetic_field_lattice.z);
            fprintf(output_info, "%.15e,%.15e,%.15e\n", info_local.magnetic_field_derivative.x, info_local.magnetic_field_derivative.y, info_local.magnetic_field_derivative.z);
        }

        integrate_exchange_grids(&gpu, exchange_id, global, local);
        time += dt;
        step++;
    }
    fclose(output_info);

    clw_print_cl_error(stderr, clReleaseMemObject(swap_buffer), "[ FATAL ] Could not release swap buffer from GPU");
    clw_print_cl_error(stderr, clReleaseMemObject(info_buffer), "[ FATAL ] Could not release info buffer from GPU");

    grid_release_from_gpu(g);
    gpu_cl_close(&gpu);
    free(info);
}

void integrate_step(double time, gpu_cl *gpu, uint64_t step_id, uint64_t global, uint64_t local) {
    clw_set_kernel_arg(gpu->kernels[step_id], 4, sizeof(double), &time);
    clw_enqueue_nd(gpu->queue, gpu->kernels[step_id], 1, NULL, &global, &local);
}

void integrate_get_info(double time, gpu_cl *gpu, uint64_t info_id, uint64_t global, uint64_t local) {
    clw_set_kernel_arg(gpu->kernels[info_id], 5, sizeof(double), &time);
    clw_enqueue_nd(gpu->queue, gpu->kernels[info_id], 1, NULL, &global, &local);
}

void integrate_exchange_grids(gpu_cl *gpu, uint64_t exchange_id, uint64_t global, uint64_t local) {
    clw_enqueue_nd(gpu->queue, gpu->kernels[exchange_id], 1, NULL, &global, &local);
}
