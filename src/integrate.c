#include "integrate.h"
#include "kernel_funcs.h"
#include "logging.h"

#include <inttypes.h>

integrate_context integrate_context_init(grid *grid, gpu_cl *gpu, double dt) {
    integrate_context ctx = {0};
    ctx.g = grid;
    ctx.gpu = gpu;
    grid_to_gpu(grid, *gpu);
    ctx.dt = dt;
    ctx.time = 0.0;
    ctx.swap_buffer = gpu_cl_create_buffer(gpu, sizeof(*grid->m) * grid->gi.rows * grid->gi.cols, CL_MEM_READ_WRITE);
    ctx.step_id = gpu_cl_append_kernel(gpu, "gpu_step");
    ctx.exchange_id = gpu_cl_append_kernel(gpu, "exchange_grid");

    gpu_cl_fill_kernel_args(gpu, ctx.step_id, 0, 6, &grid->gp_buffer, sizeof(cl_mem),
                                                 &grid->m_buffer, sizeof(cl_mem),
                                                 &ctx.swap_buffer, sizeof(cl_mem),
                                                 &dt, sizeof(double),
                                                 &ctx.time, sizeof(double),
                                                 &grid->gi, sizeof(grid_info));

    gpu_cl_fill_kernel_args(gpu, ctx.exchange_id, 0, 2, &grid->m_buffer, sizeof(cl_mem), &ctx.swap_buffer, sizeof(cl_mem));
    ctx.global = grid->gi.cols * grid->gi.rows;
    ctx.local = gpu_cl_gcd(ctx.global, 32);

    return ctx;
}

void integrate_context_close(integrate_context *ctx) {
    grid_from_gpu(ctx->g, *ctx->gpu);
    gpu_cl_release_memory(ctx->swap_buffer);
}

void integrate_base(grid *g, double dt, double duration, unsigned int interval_info, unsigned int interval_grid, string func_current, string func_field, string func_temperature, string dir_out, string compile_augment) {
    gpu_cl gpu = gpu_cl_init(func_current, func_field, func_temperature, (string){.str="\0", .len=0}, compile_augment);
    integrate_context ctx = integrate_context_init(g, &gpu, dt);

    uint64_t info_id = gpu_cl_append_kernel(&gpu, "extract_info");

    information_packed *info = calloc(g->gi.rows * g->gi.cols, sizeof(information_packed));
    if (!info)
        logging_log(LOG_FATAL, "Could not allocate for information[%"PRIu64" bytes]", g->gi.rows * g->gi.cols * sizeof(*info));

    cl_mem info_buffer = gpu_cl_create_buffer(&gpu, g->gi.rows * g->gi.cols * sizeof(*info), CL_MEM_READ_WRITE);

    gpu_cl_fill_kernel_args(&gpu, info_id, 0, 7, &g->gp_buffer, sizeof(cl_mem),
                                              &g->m_buffer, sizeof(cl_mem),
                                              &ctx.swap_buffer, sizeof(cl_mem),
                                              &info_buffer, sizeof(cl_mem),
                                              &dt, sizeof(double),
                                              &ctx.time, sizeof(double),
                                              &g->gi, sizeof(grid_info));

    uint64_t step = 0;

    string output_info_path = (string){0};

    str_cat_str(&output_info_path, dir_out);
    str_cat_cstr(&output_info_path, "/integration_info.dat");

    FILE *output_info = fopen(str_as_cstr(&output_info_path), "w");
    if (!output_info)
        logging_log(LOG_FATAL, "Could not open file %.*s: %s", (int)output_info_path.len, output_info_path.str, strerror(errno));

    str_free(&output_info_path);

    fprintf(output_info, "time(s),energy(J),exchange_energy(J),dm_energy(J),field_energy(J),anisotropy_energy(J),cubic_anisotropy_energy(J),");
    fprintf(output_info, "charge_finite,charge_lattice,");
    fprintf(output_info, "avg_mx,avg_my,avg_mz,");
    fprintf(output_info, "eletric_x(V/m),eletric_y(V/m),eletric_z(V/m),");
    fprintf(output_info, "magnetic_lattice_x(T),magnetic_lattice_y(T),magnetic_lattice_z(T),");
    fprintf(output_info, "magnetic_derivative_x(T),magnetic_derivative_y(T),magnetic_derivative_z(T),");
    fprintf(output_info, "charge_center_x(m),charge_center_y(m)\n");

    string output_grid_path = (string){0};
    str_cat_str(&output_grid_path, dir_out);
    str_cat_cstr(&output_grid_path, "/integration_evolution.dat");

    FILE *grid_evolution = fopen(str_as_cstr(&output_grid_path), "w");
    if (!output_info)
        logging_log(LOG_FATAL, "Could not open file %.*s: %s", (int)output_grid_path.len, output_grid_path.str, strerror(errno));

    str_free(&output_grid_path);
    grid_dump(grid_evolution, g);

    uint64_t expected_steps = duration / dt + 1;
    logging_log(LOG_INFO, "Expected integration steps: %"PRIu64, expected_steps);

    while (ctx.time <= duration) {
        integrate_step(&ctx);

        if (step % interval_info == 0) {
            integrate_get_info(&ctx, info_id);
            gpu_cl_read_buffer(&gpu, sizeof(*info) * g->gi.rows * g->gi.cols, 0, info, info_buffer);
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
                info_local.charge_center_x += info[i].charge_center_x;
                info_local.charge_center_y += info[i].charge_center_y;
            }
            fprintf(output_info, "%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,", ctx.time, info_local.energy, info_local.exchange_energy, info_local.dm_energy, info_local.field_energy, info_local.anisotropy_energy, info_local.cubic_energy);
            fprintf(output_info, "%.15e,%.15e,", info_local.charge_finite, info_local.charge_lattice);
            fprintf(output_info, "%.15e,%.15e,%.15e,", info_local.avg_m.x, info_local.avg_m.y, info_local.avg_m.z);
            fprintf(output_info, "%.15e,%.15e,%.15e,", info_local.eletric_field.x, info_local.eletric_field.y, info_local.eletric_field.z);
            fprintf(output_info, "%.15e,%.15e,%.15e,", info_local.magnetic_field_lattice.x, info_local.magnetic_field_lattice.y, info_local.magnetic_field_lattice.z);
            fprintf(output_info, "%.15e,%.15e,%.15e,", info_local.magnetic_field_derivative.x, info_local.magnetic_field_derivative.y, info_local.magnetic_field_derivative.z);
            fprintf(output_info, "%.15e,%.15e\n", info_local.charge_center_x / info_local.charge_finite, info_local.charge_center_y / info_local.charge_finite);
        }

        if (step % interval_grid == 0) {
            v3d_from_gpu(g->m, g->m_buffer, g->gi.rows, g->gi.cols, gpu);
            v3d_dump(grid_evolution, g->m, g->gi.rows, g->gi.cols);
        }

        if (step % (expected_steps / 100) == 0)
            logging_log(LOG_INFO, "%.3es - %.2f%%", ctx.time, ctx.time / duration * 100.0);

        integrate_exchange_grids(&ctx);
        ctx.time += dt;
        step++;
    }
    logging_log(LOG_INFO, "Steps: %d", step);
    fclose(output_info);

    v3d_from_gpu(g->m, g->m_buffer, g->gi.rows, g->gi.cols, gpu);
    v3d_dump(grid_evolution, g->m, g->gi.rows, g->gi.cols);
    integrate_context_close(&ctx);
    gpu_cl_release_memory(info_buffer);
    grid_release_from_gpu(g);
    gpu_cl_close(&gpu);
    free(info);
}

void integrate(grid *g, integration_params param) {
    integrate_base(g, param.dt, param.duration, param.interval_for_information, param.interval_for_writing_grid, param.current_generation_function, param.field_generation_function, param.temperature_generation_function, param.output_path,  param.compile_augment);
}

void integrate_step(integrate_context *ctx) {
    gpu_cl_set_kernel_arg(ctx->gpu, ctx->step_id, 4, sizeof(double), &ctx->time);
    gpu_cl_enqueue_nd(ctx->gpu, ctx->step_id, 1, &ctx->local, &ctx->global, NULL);
}

void integrate_get_info(integrate_context *ctx, uint64_t info_id) {
    gpu_cl_set_kernel_arg(ctx->gpu, info_id, 5, sizeof(double), &ctx->time);
    gpu_cl_enqueue_nd(ctx->gpu, info_id, 1, &ctx->local, &ctx->global, NULL);
}

void integrate_exchange_grids(integrate_context *ctx) {
    gpu_cl_enqueue_nd(ctx->gpu, ctx->exchange_id, 1, &ctx->local, &ctx->global, NULL);
}

void integrate_context_read_grid(integrate_context *ctx) {
    v3d_from_gpu(ctx->g->m, ctx->g->m_buffer, ctx->g->gi.rows, ctx->g->gi.cols, *ctx->gpu);
}
