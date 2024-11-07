#include "integrate.h"
#include "kernel_funcs.h"
#include "logging.h"
#include "allocator.h"
#include "utils.h"
#include "string_builder.h"

#include "stb_image_write.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>


integrate_context integrate_context_init(grid *grid, gpu_cl *gpu, integrate_params params) {
    integrate_context ctx = {0};
    ctx.g = grid;
    ctx.gpu = gpu;
    grid_to_gpu(grid, *gpu);
    ctx.params = params;
    ctx.time = 0.0;
    ctx.swap_gpu = gpu_cl_create_gpu(gpu, sizeof(*grid->m) * grid->dimensions, CL_MEM_READ_WRITE);
    ctx.step_id = gpu_cl_append_kernel(gpu, "gpu_step");
    ctx.exchange_id = gpu_cl_append_kernel(gpu, "exchange_grid");

    gpu_cl_fill_kernel_args(gpu, ctx.step_id, 0, 6, &grid->gp_gpu, sizeof(cl_mem),
                                                 &grid->m_gpu, sizeof(cl_mem),
                                                 &ctx.swap_gpu, sizeof(cl_mem),
                                                 &params.dt, sizeof(double),
                                                 &ctx.time, sizeof(double),
                                                 &grid->gi, sizeof(grid_info));

    gpu_cl_fill_kernel_args(gpu, ctx.exchange_id, 0, 5, &grid->m_gpu, sizeof(cl_mem), &ctx.swap_gpu, sizeof(cl_mem), &ctx.g->gi.rows, sizeof(ctx.g->gi.rows), &ctx.g->gi.cols, sizeof(ctx.g->gi.cols), &ctx.g->gi.depth, sizeof(ctx.g->gi.depth));
    ctx.global = grid->dimensions;
    ctx.global = ctx.global + (gpu_optimal_wg - ctx.global % gpu_optimal_wg);
    ctx.local = gpu_optimal_wg;

    string_builder output_info_path = {0};
    sb_cat_cstr(&output_info_path, params.output_path);
    sb_cat_cstr(&output_info_path, "/integrate_info.dat");

    ctx.integrate_info = mfopen(sb_as_cstr(&output_info_path), "w");

    sb_free(&output_info_path);

    fprintf(ctx.integrate_info, "time(s),energy(J),exchange_energy(J),dm_energy(J),field_energy(J),anisotropy_energy(J),cubic_anisotropy_energy(J),dipolar_energy(J),");
    fprintf(ctx.integrate_info, "charge_finite,charge_lattice,");
    fprintf(ctx.integrate_info, "avg_mx,avg_my,avg_mz,");
    fprintf(ctx.integrate_info, "electric_x(V/m),electric_y(V/m),electric_z(V/m),");
    fprintf(ctx.integrate_info, "magnetic_lattice_x(T),magnetic_lattice_y(T),magnetic_lattice_z(T),");
    fprintf(ctx.integrate_info, "magnetic_finite_x(T),magnetic_finite_y(T),magnetic_finite_z(T),");
    fprintf(ctx.integrate_info, "charge_center_x(m),charge_center_y(m),charge_center_z(m),");
    fprintf(ctx.integrate_info, "abs_charge_center_x(m),abs_charge_center_y(m),abs_charge_center_z(m),");
    fprintf(ctx.integrate_info, "D_xx,D_yy,D_zz,D_xy,D_xz,D_yz\n");

    if (params.do_cluster) {
        string_builder output_cluster_path = {0};
        sb_cat_cstr(&output_cluster_path, params.output_path);
        sb_cat_cstr(&output_cluster_path, "/clusters.dat");

        ctx.clusters = mfopen(sb_as_cstr(&output_cluster_path), "w");

        sb_free(&output_cluster_path);
    }

    string_builder output_grid_path = {0};
    sb_cat_cstr(&output_grid_path, params.output_path);
    sb_cat_cstr(&output_grid_path, "/integrate_evolution.dat");

    ctx.integrate_evolution = mfopen(sb_as_cstr(&output_grid_path), "wb");
    sb_free(&output_grid_path);

    ctx.info_id = gpu_cl_append_kernel(gpu, "extract_info");
    ctx.info = mmalloc(grid->dimensions * sizeof(*ctx.info));
    ctx.info_gpu = gpu_cl_create_gpu(gpu, grid->dimensions * sizeof(*ctx.info), CL_MEM_READ_WRITE);

    gpu_cl_fill_kernel_args(gpu, ctx.info_id, 0, 7, &grid->gp_gpu, sizeof(cl_mem),
                                                     &grid->m_gpu, sizeof(cl_mem),
                                                     &ctx.swap_gpu, sizeof(cl_mem),
                                                     &ctx.info_gpu, sizeof(cl_mem),
                                                     &ctx.params.dt, sizeof(double),
                                                     &ctx.time, sizeof(double),
                                                     &grid->gi, sizeof(grid_info));

    uint64_t expected_steps = params.duration / params.dt + 1;
    if (ctx.params.interval_for_information == 0)
        ctx.params.interval_for_information = expected_steps + 1;

    if (ctx.params.interval_for_raw_grid == 0)
        ctx.params.interval_for_raw_grid = expected_steps + 1;

    uint64_t number_raw = 3 + expected_steps / ctx.params.interval_for_raw_grid;
    logging_log(LOG_INFO, "Expected raw frames written %"PRIu64, number_raw);

    fwrite(&number_raw, sizeof(number_raw), 1, ctx.integrate_evolution);
    grid_dump(ctx.integrate_evolution, grid);

    uint64_t dump_size = grid->dimensions * (sizeof(*grid->gp) + number_raw * sizeof(*grid->m)) + sizeof(number_raw);
    logging_log(LOG_INFO, "Expected raw grid dump %.2f MB", dump_size / 1.0e6);
    return ctx;
}

void integrate_context_close(integrate_context *ctx) {
    grid_from_gpu(ctx->g, *ctx->gpu);
    gpu_cl_release_memory(ctx->swap_gpu);
    mfclose(ctx->integrate_info);
    mfclose(ctx->integrate_evolution);

    if (ctx->params.do_cluster)
        mfclose(ctx->clusters);

    gpu_cl_release_memory(ctx->info_gpu);
    mfree(ctx->info);

    grid_release_from_gpu(ctx->g);
    gpu_cl_close(ctx->gpu);
}

integrate_params integrate_params_init(void) {
    integrate_params ret = {0};
    ret.dt = 1.0e-15;
    ret.duration = 200.0 * NS;
    ret.interval_for_information = 1000;
    ret.interval_for_cluster = ret.interval_for_information;
    ret.interval_for_raw_grid = 10000;
    ret.interval_for_rgb_grid = 50000;

    ret.cluster_eps = 0.1;
    ret.cluster_min_pts = 5;
    ret.do_cluster = true;

    ret.current_func = "return (current){.type = CUR_NONE};";
    ret.field_func = "return v3d_s(0);";
    ret.temperature_func = "return 0;";
    ret.compile_augment = "-cl-fast-relaxed-math";
    ret.output_path = "./";
    return ret;
}

void integrate(grid *g, integrate_params params) {
    gpu_cl gpu = gpu_cl_init(params.current_func, params.field_func, params.temperature_func, NULL, params.compile_augment);
    integrate_context ctx = integrate_context_init(g, &gpu, params);

    uint64_t expected_steps = params.duration / params.dt + 1;
    logging_log(LOG_INFO, "Expected integrate steps: %"PRIu64, expected_steps);

    for (unsigned int t = 0; t < expected_steps; ++t) {
        integrate_step(&ctx);

        if (ctx.integrate_step % (expected_steps / 100) == 0)
            logging_log(LOG_INFO, "%.3es - %.2f%%", ctx.time, ctx.time / params.duration * 100.0);

        integrate_exchange_grids(&ctx);
    }
    logging_log(LOG_INFO, "Steps: %d", ctx.integrate_step);

    v3d_from_gpu(g->m, g->m_gpu, g->gi.rows, g->gi.cols, g->gi.depth, gpu);
    v3d_dump(ctx.integrate_evolution, g->m, g->gi.rows, g->gi.cols, g->gi.depth);

    integrate_context_close(&ctx);
}

void integrate_step(integrate_context *ctx) {
    gpu_cl_set_kernel_arg(ctx->gpu, ctx->step_id, 4, sizeof(double), &ctx->time);
    gpu_cl_enqueue_nd(ctx->gpu, ctx->step_id, 1, &ctx->local, &ctx->global, NULL);
    bool read_grid_from_gpu = false;

    if (ctx->integrate_step % ctx->params.interval_for_information == 0) {
        information_packed info = integrate_get_info(ctx);
        fprintf(ctx->integrate_info, "%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,", ctx->time, info.energy, info.exchange_energy, info.dm_energy, info.field_energy, info.anisotropy_energy, info.cubic_energy, info.dipolar_energy);
        fprintf(ctx->integrate_info, "%.15e,%.15e,", info.charge_finite, info.charge_lattice);
        fprintf(ctx->integrate_info, "%.15e,%.15e,%.15e,", info.avg_m.x, info.avg_m.y, info.avg_m.z);
        fprintf(ctx->integrate_info, "%.15e,%.15e,%.15e,", info.electric_field.x, info.electric_field.y, info.electric_field.z);
        fprintf(ctx->integrate_info, "%.15e,%.15e,%.15e,", info.magnetic_field_lattice.x, info.magnetic_field_lattice.y, info.magnetic_field_lattice.z);
        fprintf(ctx->integrate_info, "%.15e,%.15e,%.15e,", info.magnetic_field_finite.x, info.magnetic_field_finite.y, info.magnetic_field_finite.z);
        fprintf(ctx->integrate_info, "%.15e,%.15e,%.15e,", info.charge_center_x / info.charge_finite, info.charge_center_y / info.charge_finite, info.charge_center_z / info.charge_finite);
        fprintf(ctx->integrate_info, "%.15e,%.15e,%.15e,", info.abs_charge_center_x / info.abs_charge_finite, info.abs_charge_center_y / info.abs_charge_finite, info.abs_charge_center_z / info.abs_charge_finite);
        fprintf(ctx->integrate_info, "%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n", info.D_xx, info.D_yy, info.D_zz, info.D_xy, info.D_xz, info.D_yz);
    }

    if (ctx->integrate_step % ctx->params.interval_for_raw_grid == 0) {
        v3d_from_gpu(ctx->g->m, ctx->g->m_gpu, ctx->g->gi.rows, ctx->g->gi.cols, ctx->g->gi.depth, *ctx->gpu);
        v3d_dump(ctx->integrate_evolution, ctx->g->m, ctx->g->gi.rows, ctx->g->gi.cols, ctx->g->gi.depth);
        read_grid_from_gpu = true;
    }

    if (ctx->params.do_cluster && ctx->integrate_step % ctx->params.interval_for_cluster == 0) {
        if (!read_grid_from_gpu)
            v3d_from_gpu(ctx->g->m, ctx->g->m_gpu, ctx->g->gi.rows, ctx->g->gi.cols, ctx->g->gi.depth, *ctx->gpu);

        grid_cluster(ctx->g, ctx->params.cluster_eps, ctx->params.cluster_min_pts, NULL, NULL, NULL, NULL);

        fprintf(ctx->clusters, "%.15e,", ctx->time);
	for (uint64_t i = 0; i < ctx->g->clusters.len; ++i) {
            fprintf(ctx->clusters, "%.15e,%.15e,%.15e,%.15e", ctx->g->clusters.items[i].x, ctx->g->clusters.items[i].y, ctx->g->clusters.items[i].z, ctx->g->clusters.items[i].count / ((double)ctx->g->dimensions));
	    if ((i + 1) < ctx->g->clusters.len)
		fprintf(ctx->clusters, ",");
	    else
		fprintf(ctx->clusters, "\n");
	}
    }

    ctx->integrate_step += 1;
    ctx->time += ctx->params.dt;
}

information_packed integrate_get_info(integrate_context *ctx) {
    gpu_cl_set_kernel_arg(ctx->gpu, ctx->info_id, 5, sizeof(double), &ctx->time);
    gpu_cl_enqueue_nd(ctx->gpu, ctx->info_id, 1, &ctx->local, &ctx->global, NULL);

    gpu_cl_read_gpu(ctx->gpu, sizeof(*ctx->info) * ctx->g->dimensions, 0, ctx->info, ctx->info_gpu);

    information_packed info_local = {0};
    for (uint64_t i = 0; i < ctx->g->dimensions; ++i) {
        info_local.energy += ctx->info[i].energy;
        info_local.dipolar_energy += ctx->info[i].dipolar_energy;
        info_local.cubic_energy += ctx->info[i].cubic_energy;
        info_local.anisotropy_energy += ctx->info[i].anisotropy_energy;
        info_local.field_energy += ctx->info[i].field_energy;
        info_local.dm_energy += ctx->info[i].dm_energy;
        info_local.exchange_energy += ctx->info[i].exchange_energy;
        info_local.charge_finite += ctx->info[i].charge_finite;
        info_local.charge_lattice += ctx->info[i].charge_lattice;
        info_local.abs_charge_finite += ctx->info[i].abs_charge_finite;
        info_local.abs_charge_lattice += ctx->info[i].abs_charge_lattice;
        info_local.avg_m = v3d_sum(info_local.avg_m, v3d_scalar(ctx->info[i].avg_m, 1.0 / (ctx->g->dimensions)));
        info_local.electric_field = v3d_sum(info_local.electric_field , ctx->info[i].electric_field);
        info_local.magnetic_field_lattice = v3d_sum(info_local.magnetic_field_lattice, ctx->info[i].magnetic_field_lattice);
        info_local.magnetic_field_finite = v3d_sum(info_local.magnetic_field_finite, ctx->info[i].magnetic_field_finite);
        info_local.charge_center_x += ctx->info[i].charge_center_x;
        info_local.charge_center_y += ctx->info[i].charge_center_y;
        info_local.charge_center_z += ctx->info[i].charge_center_z;
        info_local.abs_charge_center_x += ctx->info[i].abs_charge_center_x;
        info_local.abs_charge_center_y += ctx->info[i].abs_charge_center_y;
        info_local.abs_charge_center_z += ctx->info[i].abs_charge_center_z;
        info_local.D_xx += ctx->info[i].D_xx;
        info_local.D_yy += ctx->info[i].D_yy;
        info_local.D_zz += ctx->info[i].D_zz;
        info_local.D_xy += ctx->info[i].D_xy;
        info_local.D_xz += ctx->info[i].D_xz;
        info_local.D_yz += ctx->info[i].D_yz;
    }
    return info_local;
}

void integrate_exchange_grids(integrate_context *ctx) {
    gpu_cl_enqueue_nd(ctx->gpu, ctx->exchange_id, 1, &ctx->local, &ctx->global, NULL);
}

void integrate_context_read_grid(integrate_context *ctx) {
    v3d_from_gpu(ctx->g->m, ctx->g->m_gpu, ctx->g->gi.rows, ctx->g->gi.cols, ctx->g->gi.depth, *ctx->gpu);
}
