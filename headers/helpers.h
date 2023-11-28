#ifndef __HELP
#define __HELP

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <omp.h>
#include <time.h>
#include <stdint.h>
#include <assert.h>

#include "vec.h"
#include "constants.h"
#include "grid.h"
#include "funcs.h"
#include "opencl_kernel.h"

#define _PARSER_IMPLEMENTATION
#include "parserL.h"

#define OPENCLWRAPPER_IMPLEMENTATION
#include "opencl_wrapper.h"

#define __PROFILER_IMPLEMENTATION
#include "profiler.h"

typedef struct {
    double qA, qT, qV, T0;
    uint64_t inner_loop, outer_loop, print_param;
} gsa_param_t;

typedef struct gpu_t {
    cl_platform_id *plats; uint64_t n_plats; int i_plat;
    cl_device_id *devs; uint64_t n_devs; int i_dev;
    cl_context ctx;
    cl_command_queue queue;
    cl_program program;
    kernel_t *kernels; uint64_t n_kernels;
} gpu_t;

typedef struct simulator_t {
    uint64_t n_steps, relax_steps, gradient_steps;
    double dt, dt_gradient, alpha_gradient, beta_gradient, temp_gradient, factor_gradient, mass_gradient;
    uint64_t n_cpu;
    uint64_t write_cut;
    uint64_t write_vel_charge_cut;
    bool write_to_file, use_gpu, do_gsa, do_relax, doing_relax, do_integrate, write_human, write_on_fly, calculate_energy, do_gradient;
    gsa_param_t gsap;
    gpu_t gpu;
    grid_t g_old;
    grid_t g_new;
    grid_param_t real_param;
    cl_mem g_old_buffer, g_new_buffer;
    v3d *grid_out_file;
    info_pack_t *simulation_info;
} simulator_t;

FILE* file_open(const char* name, const char* mode, int exit_) {
    FILE *f = fopen(name, mode);
    if (!f) {
        fprintf(stderr, "Could not open file %s: %s\n", name, strerror(errno));
        if (exit_)
            exit(exit_);
    }
    return f;
}

double rand_double() {
    return (double)rand() / (double)RAND_MAX;
}

double random_range(double min, double max) {
    return rand_double() * (max - min) + min;
}

void grid_free(grid_t *g) {
    if (g->grid) {
        free(g->grid);
        g->grid = NULL;
    }

    if (g->ani) {
        free(g->ani);
        g->ani = NULL;
    }
    
    if (g->pinning) {
        free(g->pinning);
        g->pinning = NULL;
    }
    
    if (g->regions) {
        free(g->regions);
        g->regions = NULL;
    }
}

void grid_copy(grid_t *to, grid_t *from) {
    grid_free(to);
    memcpy(&to->param, &from->param, sizeof(grid_param_t));

    to->grid = (v3d*)calloc(from->param.total, sizeof(v3d));
    to->ani = (anisotropy_t*)calloc(from->param.total, sizeof(anisotropy_t));
    to->pinning = (pinning_t*)calloc(from->param.total, sizeof(pinning_t));
    to->regions = (region_param_t*)calloc(from->param.total, sizeof(region_param_t));

    memcpy(to->grid, from->grid, from->param.total * sizeof(v3d));
    memcpy(to->ani, from->ani, from->param.total * sizeof(anisotropy_t));
    memcpy(to->pinning, from->pinning, from->param.total * sizeof(pinning_t));
    memcpy(to->regions, from->regions, from->param.total * sizeof(region_param_t));
}

void copy_grid_to_allocated_grid(grid_t *to, grid_t *from) {
    memcpy(to->grid, from->grid, from->param.total * sizeof(v3d));
    memcpy(to->ani, from->ani, from->param.total * sizeof(anisotropy_t));
    memcpy(to->pinning, from->pinning, from->param.total * sizeof(pinning_t));
    memcpy(to->regions, from->regions, from->param.total * sizeof(region_param_t));
}

void copy_spins_to_allocated_grid(grid_t *to, grid_t *from) {
    memcpy(to->grid, from->grid, from->param.total * sizeof(v3d));
}

grid_t grid_init_null() {
    grid_t ret;
    memset(&ret, 0, sizeof(grid_t));
    return ret;
}

int find_rows_in_file(const char *path) {
    FILE* f = file_open(path, "rb", 1);

    fseek(f, 0, SEEK_SET);
    fseek(f, 0, SEEK_END);

    uint64_t file_size = ftell(f);
    char* file_data = (char*)malloc(file_size + 1);

    fseek(f, 0, SEEK_SET);
    fread(file_data, 1, file_size, f);

    file_data[file_size] = '\0';

    char* ptr = file_data;
    int rows = 0;

    while(*ptr)
        if (*ptr++ == '\n')
            ++rows;
    
    if (*--ptr != '\n')
        rows++;

    free(file_data);
    fclose(f);
    return rows;
}

v3d* init_v3d_grid_from_file_bin(const char* path, int *rows, int *cols) {
    printf("Reading Grid from Binary format\n");
    FILE *f = file_open(path, "rb", 1);
    char binary[6] = {0};
    fread(binary, sizeof(binary), 1, f);
    fread(rows, sizeof(int), 1, f);
    fread(cols, sizeof(int), 1, f);
    uint64_t start = ftell(f);
    fseek(f, 0, SEEK_END);
    uint64_t end = ftell(f);
    fseek(f, start, SEEK_SET);
    uint64_t size = (*rows) * (*cols) * sizeof(v3d);
    v3d *ret = (v3d*)calloc(size, 1);
    assert(size == (end - start));
    fread(ret, size, 1, f);
    fclose(f);
    for (uint64_t I = 0; I < (uint64_t)((*rows) * (*cols)); ++I)
        ret[I] = v3d_normalize(ret[I]);
    return ret;
}

v3d* init_v3d_grid_from_file_human(const char* path, int *rows, int *cols) {
    printf("Reading Grid from Human format\n");
    int rows_ = find_rows_in_file(path);
    *rows = rows_;
    parser_context ctx = parser_init_context(global_parser_context.seps);
    parser_start(path, &ctx);

    int cols_ = ctx.n / (3 * rows_);
    *cols = cols_;
    v3d* ret = (v3d*)calloc(rows_ * cols_, sizeof(v3d));
    for (uint64_t I = 0; I < ctx.n; I += 3) {
        int j = (I / 3) % cols_;
        int i = rows_ - 1 - (I / 3 - j) / cols_;
        ret[i * cols_ + j] = v3d_normalize(v3d_c(strtod(ctx.state[I], NULL),
                                                 strtod(ctx.state[I + 1], NULL),
                                                 strtod(ctx.state[I + 2], NULL)));
    }
    parser_end(&ctx);
    return ret;
}

v3d* init_v3d_grid_from_file(const char* path, int *rows, int *cols) {
    FILE *f = file_open(path, "rb", 1);
    char binary[6] = "BINARY";
    char buffer[6] = {0};
    fread(buffer, sizeof(binary), 1, f);
    fclose(f);
    if (memcmp(binary, buffer, sizeof(binary)) == 0)
        return init_v3d_grid_from_file_bin(path, rows, cols);
    return init_v3d_grid_from_file_human(path, rows, cols);
}

v3d* init_v3d_grid_random(uint64_t rows, uint64_t cols) {
    v3d* ret = (v3d*)calloc(rows * cols, sizeof(v3d));
    for (uint64_t I = 0; I < rows * cols; ++I)
        ret[I] = v3d_normalize(v3d_c(random_range(-1.0, 1.0), 
                                      random_range(-1.0, 1.0), 
                                      random_range(-1.0, 1.0)));
    return ret;
}

grid_t init_grid_from_file(const char* path) {
    grid_t out = grid_init_null();
    out.grid = init_v3d_grid_from_file(path, &out.param.rows, &out.param.cols);
    out.param.total = out.param.rows * out.param.cols;

    out.ani = (anisotropy_t*)calloc(out.param.total, sizeof(anisotropy_t));
    out.pinning = (pinning_t*)calloc(out.param.total, sizeof(pinning_t));
    out.regions = (region_param_t*)calloc(out.param.total, sizeof(region_param_t));
    return out;
}

grid_t init_grid_random(int rows, int cols) {
    srand(time(NULL));
    grid_t ret = grid_init_null();
    ret.param.rows = rows;
    ret.param.cols = cols;
    ret.param.total = rows * cols;
    ret.grid = init_v3d_grid_random(rows, cols);
    ret.ani = (anisotropy_t*)calloc(ret.param.total, sizeof(anisotropy_t));
    ret.pinning = (pinning_t*)calloc(ret.param.total, sizeof(pinning_t));
    ret.regions = (region_param_t*)calloc(ret.param.total, sizeof(region_param_t));
    return ret;
}

uint64_t find_grid_size_bytes(const grid_t* g) {
    uint64_t param = sizeof(grid_param_t);
    uint64_t grid_v3d = g->param.total * sizeof(v3d);
    uint64_t grid_pinning = g->param.total * sizeof(pinning_t);
    uint64_t grid_ani = g->param.total * sizeof(anisotropy_t);
    uint64_t grid_regions = g->param.total * sizeof(region_param_t);
    return param + grid_v3d + grid_pinning + grid_ani + grid_regions;
}

void print_v3d_grid(FILE* f, v3d* v, int rows, int cols) {
    for (int row = rows - 1; row >= 1; --row) {
        for (int col = 0; col < cols - 1; ++col) {
            fprintf(f, "%.15f\t%.15f\t%.15f\t", v[row * cols + col].x, v[row * cols + col].y, v[row * cols + col].z);
        }
        int col = cols - 1;
        fprintf(f, "%.15f\t%.15f\t%.15f\n", v[row * cols + col].x, v[row * cols + col].y, v[row * cols + col].z);
    }

    int row = 0;

    for (int col = 0; col < cols - 1; ++col) {
        fprintf(f, "%.15f\t%.15f\t%.15f\t", v[row * cols + col].x, v[row * cols + col].y, v[row * cols + col].z);
    }
    int col = cols - 1;
    fprintf(f, "%.15f\t%.15f\t%.15f", v[row * cols + col].x, v[row * cols + col].y, v[row * cols + col].z);
}

void print_v3d_grid_path(const char* path, v3d* v, int rows, int cols) {
    FILE *f = file_open(path, "wb", 1);

    print_v3d_grid(f, v, rows, cols);

    fclose(f);
}

void find_grid_param_path(const char* path, grid_param_t* g) {
    parser_context ctx = parser_init_context(global_parser_context.seps);
    parser_start(path, &ctx);

    g->exchange = parser_get_double("EXCHANGE", QE * 1.0e-3, &ctx);
    g->dm = parser_get_double("DMI", -0.18, &ctx) * fabs(g->exchange);
    g->dm_ani = parser_get_double("DMI_ANISOTROPIC", 0, &ctx) * fabs(g->exchange);
    g->lattice = parser_get_double("LATTICE", 5.0e-10, &ctx);
    g->cubic_ani = parser_get_double("CUBIC", 0, &ctx) * fabs(g->exchange);
    g->lande = parser_get_double("LANDE", 2.002318, &ctx);
    g->avg_spin = parser_get_double("SPIN", 1, &ctx);
    g->alpha = parser_get_double("ALPHA", 0.3, &ctx);
    g->gamma = parser_get_double("GAMMA", 1.760859644e11, &ctx);
    g->mu_s = g->gamma * HBAR;

    g->dm_type = parser_get_int("DM_TYPE", 10, Z_CROSS_R_ij, &ctx);
    if (g->dm_type > 2 || g->dm_type < 0) {
        fprintf(stderr, "Invalid DM, falling back to Z_CROSS_R_ij\n");
        g->dm_type = Z_CROSS_R_ij;
    }


    g->pbc.pbc_type = parser_get_int("PBC_TYPE", 10, 0, &ctx);
    if (g->pbc.pbc_type > 3 || g->pbc.pbc_type < 0) {
        fprintf(stderr, "Invalid PBC, falling back to XY\n");
        g->pbc.pbc_type = PBC_XY;
    }

    g->pbc.dir.x = parser_get_double("PBC_X", 0, &ctx);
    g->pbc.dir.y = parser_get_double("PBC_Y", 0, &ctx);
    g->pbc.dir.z = parser_get_double("PBC_Z", 0, &ctx);


    parser_end(&ctx);
}

void full_grid_write_buffer(cl_command_queue q, cl_mem buffer, grid_t *g) {
    uint64_t off = 0;
    clw_write_buffer(buffer, &g->param, sizeof(grid_param_t), off, q);
    off += sizeof(grid_param_t);
    clw_write_buffer(buffer, g->grid, sizeof(v3d) * g->param.total, off, q);
    off += sizeof(v3d) * g->param.total;
    clw_write_buffer(buffer, g->ani, sizeof(anisotropy_t) * g->param.total, off, q);
    off += sizeof(anisotropy_t) * g->param.total;
    clw_write_buffer(buffer, g->pinning, sizeof(pinning_t) * g->param.total, off, q);
    off += sizeof(pinning_t) * g->param.total;
    clw_write_buffer(buffer, g->regions, sizeof(region_param_t) * g->param.total, off, q);
    off += sizeof(region_param_t) * g->param.total;
}

void write_v3d_grid_buffer(cl_command_queue q, cl_mem buffer, grid_t *g) {
    clw_write_buffer(buffer, g->grid, g->param.total * sizeof(v3d), sizeof(grid_param_t), q);
}

void read_full_grid_buffer(cl_command_queue q, cl_mem buffer, grid_t *g) {
    uint64_t off = 0;
    clw_read_buffer(buffer, &g->param, sizeof(grid_param_t), off, q);
    off += sizeof(grid_param_t);
    clw_read_buffer(buffer, g->grid, sizeof(v3d) * g->param.total, off, q);
    off += sizeof(v3d) * g->param.total;
    clw_read_buffer(buffer, g->ani, sizeof(anisotropy_t) * g->param.total, off, q);
    off += sizeof(anisotropy_t) * g->param.total;
    clw_read_buffer(buffer, g->pinning, sizeof(pinning_t) * g->param.total, off, q);
    off += sizeof(pinning_t) * g->param.total;
    clw_read_buffer(buffer, g->regions, sizeof(region_param_t) * g->param.total, off, q);
    off += sizeof(region_param_t) * g->param.total;
}

void read_v3d_grid_buffer(cl_command_queue q, cl_mem buffer, grid_t *g) {
    clw_read_buffer(buffer, g->grid, g->param.total * sizeof(v3d), sizeof(grid_param_t), q);
}

void integrate_simulator(simulator_t *s, const char* file_name) {
    FILE *fly = file_open(file_name, "wb", 1);
    if (s->write_on_fly && (!s->doing_relax)) {
        int steps = (int)(s->n_steps / s->write_cut);
        int fcut = s->write_cut;
        double dt_real = s->dt * HBAR / fabs(s->real_param.exchange);
        fwrite(&s->g_old.param.rows, sizeof(int), 1, fly);
        fwrite(&s->g_old.param.cols, sizeof(int), 1, fly);
        fwrite(&steps, sizeof(int), 1, fly);
        fwrite(&fcut, sizeof(int), 1, fly);
        fwrite(&dt_real, sizeof(double), 1, fly);
        fwrite(&s->real_param.lattice, sizeof(double), 1, fly);
    }

    memset(s->simulation_info, 0, sizeof(info_pack_t) * s->n_steps / s->write_vel_charge_cut);

    clw_set_kernel_arg(s->gpu.kernels[2], 0, sizeof(cl_mem), &s->g_old_buffer);
    clw_set_kernel_arg(s->gpu.kernels[2], 1, sizeof(cl_mem), &s->g_new_buffer);

    clw_set_kernel_arg(s->gpu.kernels[3], 0, sizeof(cl_mem), &s->g_old_buffer);
    clw_set_kernel_arg(s->gpu.kernels[3], 1, sizeof(cl_mem), &s->g_new_buffer);
    clw_set_kernel_arg(s->gpu.kernels[3], 2, sizeof(double), &s->dt);


    clw_set_kernel_arg(s->gpu.kernels[6], 0, sizeof(cl_mem), &s->g_old_buffer);
    clw_set_kernel_arg(s->gpu.kernels[6], 1, sizeof(cl_mem), &s->g_new_buffer);
    clw_set_kernel_arg(s->gpu.kernels[6], 2, sizeof(double), &s->dt);
    clw_set_kernel_arg(s->gpu.kernels[6], 5, sizeof(int), &s->calculate_energy);

    uint64_t global = s->g_old.param.total;
    uint64_t local = gcd(global, 32);

    info_pack_t* gpu_sim_info = (info_pack_t*)calloc(s->g_old.param.total, sizeof(info_pack_t));

    cl_mem gpu_sim_info_buffer = clw_create_buffer(sizeof(info_pack_t) * s->g_old.param.total, s->gpu.ctx, CL_MEM_READ_WRITE);
    clw_write_buffer(gpu_sim_info_buffer, gpu_sim_info, sizeof(info_pack_t) * s->g_old.param.total, 0, s->gpu.queue);

    clw_set_kernel_arg(s->gpu.kernels[6], 4, sizeof(cl_mem), &gpu_sim_info_buffer);
    profiler_start_measure("GPU INTEGRATION");

    for (uint64_t i = 0; i < s->n_steps; ++i) {
        if (i % (s->n_steps / 100) == 0) {
            printf("%.3f%%\n", 100.0 * (double)i / (double)s->n_steps);
            fflush(stdout);
        }

        double norm_time = (double)i * s->dt * (!s->doing_relax);
        clw_set_kernel_arg(s->gpu.kernels[3], 3, sizeof(double), &norm_time);

        clw_enqueue_nd(s->gpu.queue, s->gpu.kernels[3], 1, NULL, &global, &local);

        if (i % s->write_vel_charge_cut == 0) {
            uint64_t t = i / s->write_vel_charge_cut;

            clw_set_kernel_arg(s->gpu.kernels[6], 3, sizeof(double), &norm_time);
            clw_enqueue_nd(s->gpu.queue, s->gpu.kernels[6], 1, NULL, &global, &local);
            clw_read_buffer(gpu_sim_info_buffer, gpu_sim_info, sizeof(info_pack_t) * s->g_old.param.total, 0, s->gpu.queue);

            for (uint64_t k = 0; k < s->g_old.param.total; ++k) {
                s->simulation_info[t].vx += gpu_sim_info[k].vx;
                s->simulation_info[t].vy += gpu_sim_info[k].vy;
                s->simulation_info[t].avg_mag = v3d_add(s->simulation_info[t].avg_mag, gpu_sim_info[k].avg_mag);
                s->simulation_info[t].charge_lattice += gpu_sim_info[k].charge_lattice;
                s->simulation_info[t].charge_finite += gpu_sim_info[k].charge_finite;
                s->simulation_info[t].charge_cx += gpu_sim_info[k].charge_cx;
                s->simulation_info[t].charge_cy += gpu_sim_info[k].charge_cy;
                s->simulation_info[t].energy += gpu_sim_info[k].energy;
                s->simulation_info[t].energy_exchange += gpu_sim_info[k].energy_exchange;
                s->simulation_info[t].energy_dm += gpu_sim_info[k].energy_dm;
                s->simulation_info[t].energy_zeeman += gpu_sim_info[k].energy_zeeman;
                s->simulation_info[t].energy_anisotropy += gpu_sim_info[k].energy_anisotropy;
                s->simulation_info[t].energy_cubic_anisotropy += gpu_sim_info[k].energy_cubic_anisotropy;
            }

            s->simulation_info[t].avg_mag = v3d_scalar(s->simulation_info[t].avg_mag, 1.0 / (double)s->g_old.param.total);
        }

        clw_enqueue_nd(s->gpu.queue, s->gpu.kernels[2], 1, NULL, &global, &local);

        if (i % s->write_cut == 0) {
            uint64_t t = i / s->write_cut;
            if (s->write_to_file) {
                read_v3d_grid_buffer(s->gpu.queue, s->g_old_buffer, &s->g_old);
                memcpy(&s->grid_out_file[t * s->g_old.param.total], s->g_old.grid, sizeof(v3d) * s->g_old.param.total);
            }
            if (s->write_on_fly && (!s->doing_relax)) {
                read_v3d_grid_buffer(s->gpu.queue, s->g_old_buffer, &s->g_old);
                fwrite(s->g_old.grid, sizeof(v3d) * s->g_old.param.total, 1, fly);
            }
        }
    }
    profiler_end_measure("GPU INTEGRATION");
    read_v3d_grid_buffer(s->gpu.queue, s->g_old_buffer, &s->g_old);
    clw_print_cl_error(stderr, clReleaseMemObject(gpu_sim_info_buffer), "Could not release gpu_sim_info_buffer obj");
    free(gpu_sim_info);
    fclose(fly);
}

void write_simulation_data(const char* root_path, simulator_t* s) {
    char *out_charge;
    uint64_t out_charge_size = snprintf(NULL, 0, "%s_charge.out", root_path) + 1;
    out_charge = (char*)calloc(out_charge_size, 1);
    snprintf(out_charge, out_charge_size, "%s_charge.out", root_path);
    out_charge[out_charge_size - 1] = '\0';

    char *out_velocity;
    uint64_t out_velocity_size = snprintf(NULL, 0, "%s_velocity.out", root_path) + 1;
    out_velocity = (char*)calloc(out_velocity_size, 1);
    snprintf(out_velocity, out_velocity_size, "%s_velocity.out", root_path);
    out_velocity[out_velocity_size - 1] = '\0';

    char *out_charge_center;
    uint64_t out_charge_center_size = snprintf(NULL, 0, "%s_charge_center.out", root_path) + 1;
    out_charge_center = (char*)calloc(out_charge_center_size, 1);
    snprintf(out_charge_center, out_charge_center_size, "%s_charge_center.out", root_path);
    out_charge_center[out_charge_center_size - 1] = '\0';

    char *out_avg_mag;
    uint64_t out_avg_mag_size = snprintf(NULL, 0, "%s_avg_mag.out", root_path) + 1;
    out_avg_mag = (char*)calloc(out_avg_mag_size, 1);
    snprintf(out_avg_mag, out_avg_mag_size, "%s_avg_mag.out", root_path);
    out_avg_mag[out_avg_mag_size - 1] = '\0';

    char *out_energy;
    uint64_t out_energy_size = snprintf(NULL, 0, "%s_energy.out", root_path) + 1;
    out_energy = (char*)calloc(out_energy_size, 1);
    snprintf(out_energy, out_energy_size, "%s_energy.out", root_path);
    out_energy[out_energy_size - 1] = '\0';

    FILE *charge_total = file_open(out_charge, "w", 0);
    FILE *velocity_total = file_open(out_velocity, "w", 0);
    FILE *charge_center = file_open(out_charge_center, "w", 0);
    FILE *avg_mag_f = file_open(out_avg_mag, "w", 0);
    FILE *energy_f = file_open(out_energy, "w", 0);

    free(out_charge);
    free(out_velocity);
    free(out_charge_center);
    free(out_avg_mag);
    free(out_energy);

    double J_abs = fabs(s->real_param.exchange);
    printf("Writing charges related output\n");
    fprintf(velocity_total, "t(tau)\tt(ns)\tvx_lat_charge(a/tau)\tvy_lat_charge(a/tau)\tvx_diff_charge(a/tau)\tvy_diff_charge(a/tau)\t"
                            "vx_lat_charge(m/s)\tvy_lat_charge(m/s)\tvx_diff_charge(m/s)\tvy_diff_charge(m/s)\n");
    fprintf(charge_total, "t(tau)\tt(ns)\tQ_lattice\tQ_finite\n");
    fprintf(charge_center, "t(tau)\tt(ns)\tcx(a)\tcy(a)\tcx(nm)\tcy(nm)\n");
    fprintf(avg_mag_f, "t(tau)\tt(ns)\t<mx>\t<my>\t<mz>\n");
    fprintf(energy_f, "t(tau)\tt(ns)\tE_total(normalized)\tExchange(normalized)\tDM(normalized)\tZeeman(normalized)\tAnisotropy(normalized)\tCubic_Anisotropy(normalized)\tE_total(eV)\tExchange(eV)\tDM(eV)\tZeeman(eV)\tAnisotropy(eV)\tCubic_Anisotropy(eV)\n");
    for (uint64_t i = 0; i < s->n_steps; ++i) {
        if (i % (s->n_steps / 10) == 0)
            printf("%.3f%%\n", (double)i / (double)s->n_steps * 100.0);
        if (i % s->write_vel_charge_cut)
            continue;

        uint64_t t = i / s->write_vel_charge_cut;
        double vx = s->simulation_info[t].vx;
        double vy = s->simulation_info[t].vy;
        double chpr = s->simulation_info[t].charge_lattice;
        double chim = s->simulation_info[t].charge_finite;
        double energy = s->simulation_info[t].energy;
        double energy_exchange = s->simulation_info[t].energy_exchange;
        double energy_dm = s->simulation_info[t].energy_dm;
        double energy_zeeman = s->simulation_info[t].energy_zeeman;
        double energy_anisotropy = s->simulation_info[t].energy_anisotropy;
        double energy_cubic_anisotropy = s->simulation_info[t].energy_cubic_anisotropy;
        double cx = s->simulation_info[t].charge_cx;
        double cy = s->simulation_info[t].charge_cy;
        v3d avg_mag = s->simulation_info[t].avg_mag;

        double v_factor = s->real_param.lattice * J_abs / HBAR;
        double t_factor = HBAR / J_abs * 1.0 / 1e-9;
        double tn = i * s->dt;

        fprintf(velocity_total, "%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t", tn, tn * t_factor, vx / chpr, vy / chpr, vx / chim, vy / chim);
        fprintf(velocity_total, "%.15e\t%.15e\t%.15e\t%.15e\n", vx / chpr * v_factor, vy * v_factor / chpr, vx * v_factor / chim, vy * v_factor / chim);

        fprintf(charge_total, "%.15e\t%.15e\t%.15e\t%.15e\n", tn, tn * t_factor, chpr, chim);

        fprintf(charge_center, "%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\n", tn, tn * t_factor, cx / chpr, cy / chpr,
                                                               cx / chpr * s->real_param.lattice / 1.0e-9, cy / chpr * s->real_param.lattice / 1.0e-9);

        fprintf(avg_mag_f, "%.15e\t%.15e\t%.15e\t%.15e\t%.15e\n", tn, tn * t_factor, avg_mag.x, avg_mag.y, avg_mag.z);


	    fprintf(energy_f, "%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t", tn, tn * t_factor,
                                energy, energy_exchange, energy_dm, energy_zeeman, energy_anisotropy, energy_cubic_anisotropy);

	    fprintf(energy_f, "%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\n",  energy * J_abs / QE, energy_exchange * J_abs / QE, energy_dm * J_abs / QE, energy_zeeman * J_abs / QE, energy_anisotropy * J_abs / QE, energy_cubic_anisotropy * J_abs / QE);
    } 
    printf("Done writing charges related output\n");
    fclose(charge_total);
    fclose(velocity_total);
    fclose(charge_center);
    fclose(avg_mag_f);
    fclose(energy_f);

    if (!(s->write_to_file && s->write_human))
        return;
}

void dump_v3d_grid(const char* file_path, v3d* g, int rows, int cols, double lattice) {
    FILE *f = file_open(file_path, "wb", 1);
    fwrite(&rows, sizeof(int), 1, f);
    fwrite(&cols, sizeof(int), 1, f);
    fwrite(&lattice, sizeof(double), 1, f);
    fwrite(g, sizeof(v3d) * rows * cols, 1, f);
    fclose(f);
}

#endif
