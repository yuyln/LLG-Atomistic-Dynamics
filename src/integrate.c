#include "integrate.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "kernel_funcs.h"

integrate_context integrate_context_init(grid *grid, gpu_cl *gpu, double dt) {
    integrate_context ctx = {0};
    ctx.g = grid;
    ctx.gpu = gpu;
    grid_to_gpu(grid, *gpu);
    ctx.dt = dt;
    ctx.time = 0.0;
    ctx.swap_buffer = clw_create_buffer(sizeof(*grid->m) * grid->gi.rows * grid->gi.cols, gpu->ctx, CL_MEM_READ_WRITE);
    ctx.step_id = gpu_append_kernel(gpu, "gpu_step");
    ctx.exchange_id = gpu_append_kernel(gpu, "exchange_grid");

    gpu_fill_kernel_args(gpu, ctx.step_id, 0, 6, &grid->gp_buffer, sizeof(cl_mem),
                                                 &grid->m_buffer, sizeof(cl_mem),
                                                 &ctx.swap_buffer, sizeof(cl_mem),
                                                 &dt, sizeof(double),
                                                 &ctx.time, sizeof(double),
                                                 &grid->gi, sizeof(grid_info));

    gpu_fill_kernel_args(gpu, ctx.exchange_id, 0, 2, &grid->m_buffer, sizeof(cl_mem), &ctx.swap_buffer, sizeof(cl_mem));
    ctx.global = grid->gi.cols * grid->gi.rows;
    ctx.local = clw_gcd(ctx.global, 32);

    return ctx;
}

void integrate_context_close(integrate_context *ctx) {
    clw_print_cl_error(stderr, clReleaseMemObject(ctx->swap_buffer), "[ FATAL ] Could not release ctx swap buffer from GPU");
}

void integrate_vars(grid *g, integration_params param) {
    integrate_base(g, param.dt, param.duration, param.interval_for_information, param.interval_for_writing_grid, param.current_generation_function, param.field_generation_function, param.temperature_generation_function, param.output_path,  param.compile_augment);
}

void integrate_base(grid *g, double dt, double duration, unsigned int interval_info, unsigned int interval_grid, string_view func_current, string_view func_field, string_view func_temperature, string_view dir_out, string_view compile_augment) {

    gpu_cl gpu = gpu_cl_init(func_current, func_field, func_temperature, sv_from_cstr(""), compile_augment);
    integrate_context ctx = integrate_context_init(g, &gpu, dt);

    uint64_t info_id = gpu_append_kernel(&gpu, "extract_info");

    information_packed *info = calloc(g->gi.rows * g->gi.cols, sizeof(information_packed));
    cl_mem info_buffer = clw_create_buffer(g->gi.rows * g->gi.cols * sizeof(*info), gpu.ctx, CL_MEM_READ_WRITE);

    gpu_fill_kernel_args(&gpu, info_id, 0, 7, &g->gp_buffer, sizeof(cl_mem),
                                              &g->m_buffer, sizeof(cl_mem),
                                              &ctx.swap_buffer, sizeof(cl_mem),
                                              &info_buffer, sizeof(cl_mem),
                                              &dt, sizeof(double),
                                              &ctx.time, sizeof(double),
                                              &g->gi, sizeof(grid_info));

    uint64_t step = 0;

    string output_info_path = (string){0};
    string_add_sv(&output_info_path, dir_out);
    string_add_cstr(&output_info_path, "/integration_info.dat");

    FILE *output_info = fopen(string_as_cstr(&output_info_path), "w");
    if (!output_info) {
        fprintf(stderr, "[ FATAL ] Could not open file %.*s: %s", (int)output_info_path.len, output_info_path.str, strerror(errno));
        exit(1);
    }
    string_free(&output_info_path);

    fprintf(output_info, "time(s),energy(eV),exchange_energy(eV),dm_energy(eV),field_energy(eV),anisotropy_energy(eV),cubic_anisotropy_energy(eV),");
    fprintf(output_info, "charge_finite,charge_lattice,");
    fprintf(output_info, "avg_mx,avg_my,avg_mz,");
    fprintf(output_info, "eletric_x,eletric_y,eletric_z,");
    fprintf(output_info, "magnetic_lattice_x,magnetic_lattice_y,magnetic_lattice_z,");
    fprintf(output_info, "magnetic_derivative_x,magnetic_derivative_y,magnetic_derivative_z\n");

    string output_grid_path = (string){0};
    string_add_sv(&output_grid_path, dir_out);
    string_add_cstr(&output_grid_path, "/integration_evolution.dat");

    FILE *grid_evolution = fopen(string_as_cstr(&output_grid_path), "w");
    if (!output_info) {
        fprintf(stderr, "[ FATAL ] Could not open file %.*s: %s", (int)output_grid_path.len, output_grid_path.str, strerror(errno));
        exit(1);
    }
    string_free(&output_grid_path);
    grid_dump(grid_evolution, g);


    while (ctx.time <= duration) {
        integrate_step(&ctx);

        if (step % interval_info == 0) {
            integrate_get_info(&ctx, info_id);
            clw_print_cl_error(stderr, clEnqueueReadBuffer(gpu.queue, info_buffer, CL_TRUE, 0, sizeof(*info) * g->gi.rows * g->gi.cols, info, 0, NULL, NULL), "[ FATAL ] Could not read info_buiffer");
            information_packed info_local = {0};
            for (uint64_t i = 0; i < g->gi.rows * g->gi.cols; ++i) {
                info_local.energy += info[i].energy;
                info_local.dipolar_energy += info[i].dipolar_energy;
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
            fprintf(output_info, "%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,", ctx.time, info_local.energy, info_local.exchange_energy, info_local.dm_energy, info_local.field_energy, info_local.anisotropy_energy, info_local.cubic_energy);
            fprintf(output_info, "%.15e,%.15e,", info_local.charge_finite, info_local.charge_lattice);
            fprintf(output_info, "%.15e,%.15e,%.15e,", info_local.avg_m.x, info_local.avg_m.y, info_local.avg_m.z);
            fprintf(output_info, "%.15e,%.15e,%.15e,", info_local.eletric_field.x, info_local.eletric_field.y, info_local.eletric_field.z);
            fprintf(output_info, "%.15e,%.15e,%.15e,", info_local.magnetic_field_lattice.x, info_local.magnetic_field_lattice.y, info_local.magnetic_field_lattice.z);
            fprintf(output_info, "%.15e,%.15e,%.15e\n", info_local.magnetic_field_derivative.x, info_local.magnetic_field_derivative.y, info_local.magnetic_field_derivative.z);
        }

        if (step % interval_grid == 0) {
            v3d_from_gpu(g->m, g->m_buffer, g->gi.rows, g->gi.cols, gpu);
            v3d_dump(grid_evolution, g->m, g->gi.rows, g->gi.cols);
        }

        integrate_exchange_grids(&ctx);
        ctx.time += dt;
        step++;
    }
    printf("Steps: %d\n", (int)step);
    fclose(output_info);

    v3d_from_gpu(g->m, g->m_buffer, g->gi.rows, g->gi.cols, gpu);
    v3d_dump(grid_evolution, g->m, g->gi.rows, g->gi.cols);

    integrate_context_close(&ctx);
    clw_print_cl_error(stderr, clReleaseMemObject(info_buffer), "[ FATAL ] Could not release info buffer from GPU");

    grid_release_from_gpu(g);
    gpu_cl_close(&gpu);
    free(info);
}

void integrate_step(integrate_context *ctx) {
    clw_set_kernel_arg(ctx->gpu->kernels[ctx->step_id], 4, sizeof(double), &ctx->time);
    clw_enqueue_nd(ctx->gpu->queue, ctx->gpu->kernels[ctx->step_id], 1, NULL, &ctx->global, &ctx->local);
}

void integrate_get_info(integrate_context *ctx, uint64_t info_id) {
    clw_set_kernel_arg(ctx->gpu->kernels[info_id], 5, sizeof(double), &ctx->time);
    clw_enqueue_nd(ctx->gpu->queue, ctx->gpu->kernels[info_id], 1, NULL, &ctx->global, &ctx->local);
}

void integrate_exchange_grids(integrate_context *ctx) {
    clw_enqueue_nd(ctx->gpu->queue, ctx->gpu->kernels[ctx->exchange_id], 1, NULL, &ctx->global, &ctx->local);
}

void integrate_context_read_grid(integrate_context *ctx) {
    v3d_from_gpu(ctx->g->m, ctx->g->m_buffer, ctx->g->gi.rows, ctx->g->gi.cols, *ctx->gpu);
}
