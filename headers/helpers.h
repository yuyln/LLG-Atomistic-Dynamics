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

#include "./headers/vec.h"
#include "./headers/constants.h"
#include "./headers/grid.h"
#include "./headers/funcs.h"
#include "./headers/opencl_kernel.h"

#define _PARSER_IMPLEMENTATION
#include "./headers/parserL.h"

#define OPENCLWRAPPER_IMPLEMENTATION
#include "./headers/opencl_wrapper.h"

#define PROFILER_IMPLEMENTATION
#include "./headers/profiler.h"

#if __STDC_VERSION__ > 201603L
#define mynodiscard [[nodiscard]]
#else
#define mynodiscard 
#endif

typedef struct {
    double qA, qT, qV, T0;
    size_t inner_loop, outer_loop, print_param;
} gsa_param_t;

typedef struct gpu_t {
    cl_platform_id *plats; size_t n_plats; int i_plat;
    cl_device_id *devs; size_t n_devs; int i_dev;
    cl_context ctx;
    cl_command_queue queue;
    cl_program program;
    kernel_t *kernels; size_t n_kernels;
} gpu_t;

typedef struct simulator_t {
    size_t n_steps, relax_steps, gradient_steps;
    double dt, dt_gradient, alpha_gradient, beta_gradient, temp_gradient, factor_gradient, mass_gradient;
    size_t n_cpu;
    size_t write_cut;
    size_t write_vel_charge_cut;
    bool write_to_file, use_gpu, do_gsa, do_relax, doing_relax, do_integrate, write_human, write_on_fly, calculate_energy, do_gradient;
    gsa_param_t gsap;
    gpu_t gpu;
    grid_t g_old;
    grid_t g_new;
    cl_mem g_old_buffer, g_new_buffer;
    v3d *grid_out_file, *velxy_Ez, *pos_xy, *avg_mag, *chpr_chim;
} simulator_t;

FILE* file_open(const char* name, const char* mode, int exit_) {
    FILE *f = fopen(name, mode);
    if (!f) {
        fprintf(stderr, "Could not open file %s: %s\n", name, strerror(errno));
        if (exit_) exit(exit_);
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

    size_t file_size = ftell(f);
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

v3d* init_v3d_grid_from_file(const char* path, int *rows, int *cols) {
    int rows_ = find_rows_in_file(path);
    *rows = rows_;
    parser_start(path, NULL);

    int cols_ = global_parser_context.n / (3 * rows_);
    *cols = cols_;
    v3d* ret = (v3d*)calloc(rows_ * cols_, sizeof(v3d));
    for (size_t I = 0; I < global_parser_context.n; I += 3) {
        int j = (I / 3) % cols_;
        int i = rows_ - 1 - (I / 3 - j) / cols_;
        ret[i * cols_ + j] = v3d_normalize(v3d_c(strtod(global_parser_context.state[I], NULL),
                                                  strtod(global_parser_context.state[I + 1], NULL),
                                                  strtod(global_parser_context.state[I + 2], NULL)));
    }
    parser_end(NULL);
    return ret;
}

v3d* init_v3d_grid_random(size_t rows, size_t cols) {
    v3d* ret = (v3d*)calloc(rows * cols, sizeof(v3d));
    for (size_t I = 0; I < rows * cols; ++I)
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

size_t find_grid_size_bytes(const grid_t* g) {
    size_t param = sizeof(grid_param_t);
    size_t grid_v3d = g->param.total * sizeof(v3d);
    size_t grid_pinning = g->param.total * sizeof(pinning_t);
    size_t grid_ani = g->param.total * sizeof(anisotropy_t);
    size_t grid_regions = g->param.total * sizeof(region_param_t);
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
    parser_start(path, NULL);

    g->exchange = parser_get_double("EXCHANGE", 0, NULL);
    g->dm = parser_get_double("DMI", 0, NULL) * g->exchange;
    g->lattice = parser_get_double("LATTICE", 0, NULL);
    g->cubic_ani = parser_get_double("CUBIC", 0, NULL);
    g->lande = parser_get_double("LANDE", 0, NULL);
    g->avg_spin = parser_get_double("SPIN", 0, NULL);
    g->mu_s = g->lande * MU_B * g->avg_spin;
    g->alpha = parser_get_double("ALPHA", 0, NULL);
    g->gamma = parser_get_double("GAMMA", 0, NULL);

    if (parser_get_int("DM_TYPE", 10, 0, NULL) > 1 || parser_get_int("DM_TYPE", 10, 0, NULL) < 0) {
        fprintf(stderr, "Invalid DM\n");
        exit(1);
    }
    g->dm_type = parser_get_int("DM_TYPE", 10, 0, NULL);


    if (parser_get_int("PBC_TYPE", 10, 0, NULL) > 3 || parser_get_int("PBC_TYPE", 10, 0, NULL) < 0) {
        fprintf(stderr, "Invalid PBC\n");
        exit(1);
    }
    g->pbc.pbc_type = parser_get_int("PBC_TYPE", 10, 0, NULL);
    g->pbc.dir.x = parser_get_double("PBC_X", 0, NULL);
    g->pbc.dir.y = parser_get_double("PBC_Y", 0, NULL);
    g->pbc.dir.z = parser_get_double("PBC_Z", 0, NULL);


    parser_end(NULL);
}

v3d field_joule_to_tesla(v3d field, double mu_s) {
    return v3d_scalar(field, 1.0 / mu_s);
}

v3d field_tesla_to_joule(v3d field, double mu_s) {
    return v3d_scalar(field, mu_s);
}

void full_grid_write_buffer(cl_command_queue q, cl_mem buffer, grid_t *g) {
    size_t off = 0;
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
    size_t off = 0;
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


//@TODO: Different cut for velocity and charge files
void integrate_simulator_single(simulator_t* s, v3d field, current_t cur, const char* file_name) {
    FILE *fly = file_open(file_name, "wb", 1);
    if (s->write_on_fly && (!s->doing_relax)) {
	int steps = (int)(s->n_steps / s->write_cut);
	int fcut = s->write_cut;
	double dt_real = s->dt * HBAR / s->g_old.param.exchange;
        fwrite(&s->g_old.param.rows, sizeof(int), 1, fly);
        fwrite(&s->g_old.param.cols, sizeof(int), 1, fly);
        fwrite(&steps, sizeof(int), 1, fly);
	fwrite(&fcut, sizeof(int), 1, fly);
	fwrite(&dt_real, sizeof(double), 1, fly);
	fwrite(&s->g_old.param.lattice, sizeof(double), 1, fly);
    }
    memset(s->velxy_Ez, 0, sizeof(v3d) * s->n_steps / s->write_vel_charge_cut);
    memset(s->pos_xy, 0, sizeof(v3d) * s->n_steps / s->write_vel_charge_cut);
    memset(s->avg_mag, 0, sizeof(v3d) * s->n_steps / s->write_vel_charge_cut);
    memset(s->chpr_chim, 0, sizeof(v3d) * s->n_steps / s->write_vel_charge_cut);
    
    for (size_t i = 0; i < s->n_steps; ++i) {
        if (i % (s->n_steps / 10) == 0) {
            printf("%.3f%%\n", 100.0 * (double)i / (double)s->n_steps);
            fflush(stdout);
        }
        double norm_time = (double)i * s->dt * (!s->doing_relax);
        size_t t = i / s->write_vel_charge_cut;
        for (size_t I = 0; I < s->g_old.param.total; ++I) {
            s->g_new.grid[I] = v3d_add(s->g_old.grid[I], step(I, &s->g_old, field, cur, s->dt, norm_time));
            grid_normalize(I, s->g_new.grid, s->g_new.pinning);

            if (i % s->write_vel_charge_cut == 0) {
                size_t x = I % s->g_old.param.cols;
                size_t y = (I - x) / s->g_old.param.cols;
                double charge_i = charge(I, s->g_new.grid, s->g_old.param.rows, s->g_old.param.cols, s->g_old.param.pbc);
                double charge_i_old = charge_old(I, s->g_new.grid, s->g_old.param.rows, s->g_old.param.cols, s->g_old.param.lattice, s->g_old.param.lattice, s->g_old.param.pbc);
                s->pos_xy[t].x += (double)x * s->g_old.param.lattice * charge_i;
                s->pos_xy[t].y += (double)y * s->g_old.param.lattice * charge_i;

                v3d vt = velocity_weighted(I, s->g_new.grid, s->g_old.grid, s->g_new.grid, s->g_old.param.rows, s->g_old.param.cols, 
                            s->g_old.param.lattice, s->g_old.param.lattice, 0.5 * s->dt * HBAR / fabs(s->g_old.param.exchange), s->g_old.param.pbc);

                s->velxy_Ez[t].x += vt.x;
                s->velxy_Ez[t].y += vt.y;

                s->avg_mag[t] = v3d_add(s->avg_mag[t], s->g_new.grid[I]);

                s->chpr_chim[t].x = charge_i;
                s->chpr_chim[t].y = charge_i_old;
            }
        }
        
        memcpy(s->g_old.grid, s->g_new.grid, sizeof(v3d) * s->g_old.param.total);
        if (i % s->write_vel_charge_cut == 0) {
            // Moved to export phase
            /*s->velxy_Ez_chargez[t].x /= s->velxy_Ez_chargez[t].z;
            s->velxy_Ez_chargez[t].y /= s->velxy_Ez_chargez[t].z;
            s->pos_xy[t].x /= s->velxy_Ez_chargez[t].z;
            s->pos_xy[t].y /= s->velxy_Ez_chargez[t].z;*/
            s->avg_mag[t] = v3d_scalar(s->avg_mag[t], 1.0 / (double)s->g_old.param.total);
        }

        if (i % s->write_cut == 0) {
            t = i / s->write_cut;
            if (s->write_to_file)
                memcpy(&s->grid_out_file[t * s->g_old.param.total], s->g_old.grid, sizeof(v3d) * s->g_old.param.total);
            if (s->write_on_fly && (!s->doing_relax))
                fwrite(s->g_old.grid, sizeof(v3d) * s->g_old.param.total, 1, fly);
        }
    }

    fclose(fly);
}

void integrate_simulator_multiple(simulator_t* s, v3d field, current_t cur, const char* file_name) {
    FILE *fly = file_open(file_name, "wb", 1);
    if (s->write_on_fly && (!s->doing_relax)) {
	int steps = (int)(s->n_steps / s->write_cut);
	int fcut = s->write_cut;
	double dt_real = s->dt * HBAR / s->g_old.param.exchange;
        fwrite(&s->g_old.param.rows, sizeof(int), 1, fly);
        fwrite(&s->g_old.param.cols, sizeof(int), 1, fly);
        fwrite(&steps, sizeof(int), 1, fly);
	fwrite(&fcut, sizeof(int), 1, fly);
	fwrite(&dt_real, sizeof(double), 1, fly);
	fwrite(&s->g_old.param.lattice, sizeof(double), 1, fly);
    }

    memset(s->velxy_Ez, 0, sizeof(v3d) * s->n_steps / s->write_vel_charge_cut);
    memset(s->pos_xy, 0, sizeof(v3d) * s->n_steps / s->write_vel_charge_cut);
    memset(s->avg_mag, 0, sizeof(v3d) * s->n_steps / s->write_vel_charge_cut);
    memset(s->chpr_chim, 0, sizeof(v3d) * s->n_steps / s->write_vel_charge_cut);

    v3d *velxy_thread = (v3d*)(calloc(s->n_cpu, sizeof(v3d)));
    v3d *pos_xy_thread = (v3d*)(calloc(s->n_cpu, sizeof(v3d)));
    v3d *avg_mag_thread = (v3d*)(calloc(s->n_cpu, sizeof(v3d)));
    v3d *chpr_chim_thread = (v3d*)(calloc(s->n_cpu, sizeof(v3d)));

    for (size_t i = 0; i < s->n_steps; ++i) {
        double norm_time = (double)i * s->dt * (!s->doing_relax);
        size_t t = i / s->write_vel_charge_cut;

        if (i % s->write_vel_charge_cut == 0) {
            memset(velxy_thread, 0, sizeof(v3d) * s->n_cpu);
            memset(pos_xy_thread, 0, sizeof(v3d) * s->n_cpu);
            memset(avg_mag_thread, 0, sizeof(v3d) * s->n_cpu);
            memset(chpr_chim_thread, 0, sizeof(v3d) * s->n_cpu);
        }

        if (i % (s->n_steps / 10) == 0) {
            printf("%.3f%%\n", 100.0 * (double)i / (double)s->n_steps);
            fflush(stdout);
        }

        int I = 0; //For Visual Studio not complain
        #pragma omp parallel for num_threads(s->n_cpu)
        for (I = 0; I < s->g_old.param.total; ++I) {
            s->g_new.grid[I] = v3d_add(s->g_old.grid[I], step(I, &s->g_old, field, cur, s->dt, norm_time));
            grid_normalize(I, s->g_new.grid, s->g_new.pinning);

            if (i % s->write_vel_charge_cut == 0) {
                int nt = omp_get_thread_num();
                size_t x = I % s->g_old.param.cols;
                size_t y = (I - x) / s->g_old.param.cols;
                double charge_i = charge(I, s->g_new.grid, s->g_old.param.rows, s->g_old.param.cols, s->g_old.param.pbc);
                double charge_i_old = charge_old(I, s->g_new.grid, s->g_old.param.rows, s->g_old.param.cols, s->g_old.param.lattice, s->g_old.param.lattice, s->g_old.param.pbc);
                pos_xy_thread[nt].x += (double)x * s->g_old.param.lattice * charge_i;
                pos_xy_thread[nt].y += (double)y * s->g_old.param.lattice * charge_i;

                v3d vt = velocity_weighted(I, s->g_new.grid, s->g_old.grid, s->g_new.grid, s->g_old.param.rows, s->g_old.param.cols, 
                            s->g_old.param.lattice, s->g_old.param.lattice, 0.5 * s->dt * HBAR / fabs(s->g_old.param.exchange), s->g_old.param.pbc);
                velxy_thread[nt].x += vt.x;
                velxy_thread[nt].y += vt.y;

                avg_mag_thread[nt] = v3d_add(avg_mag_thread[nt], s->g_new.grid[I]);

                chpr_chim_thread[nt] = v3d_add(chpr_chim_thread[nt], v3d_c(charge_i, charge_i_old, 0.0));
            }
        }
        
        memcpy(s->g_old.grid, s->g_new.grid, sizeof(v3d) * s->g_old.param.total);
        if (i % s->write_vel_charge_cut == 0)  {
            for (size_t k = 0; k < s->n_cpu; ++k) {
                s->velxy_Ez[t] = v3d_add(s->velxy_Ez[t], velxy_thread[k]);
                s->pos_xy[t] = v3d_add(s->pos_xy[t], pos_xy_thread[k]);
                s->avg_mag[t] = v3d_add(s->avg_mag[t], avg_mag_thread[k]);
                s->chpr_chim[t] = v3d_add(s->chpr_chim[t], chpr_chim_thread[k]);
            }

            // Moved to export phase
            /*s->velxy_Ez_chargez[t].x /= s->velxy_Ez_chargez[t].z;
            s->velxy_Ez_chargez[t].y /= s->velxy_Ez_chargez[t].z;
            s->pos_xy[t].x /= s->velxy_Ez_chargez[t].z;
            s->pos_xy[t].y /= s->velxy_Ez_chargez[t].z;*/
            s->avg_mag[t] = v3d_scalar(s->avg_mag[t], 1.0 / (double)s->g_old.param.total);
        }
        
        if (i % s->write_cut == 0) {
            t = i / s->write_cut;
            if (s->write_to_file)
                memcpy(&s->grid_out_file[t * s->g_old.param.total], s->g_old.grid, sizeof(v3d) * s->g_old.param.total);
            if (s->write_on_fly && (!s->doing_relax))
                fwrite(s->g_old.grid, sizeof(v3d) * s->g_old.param.total, 1, fly);
        }
    }
    fclose(fly);
    free(velxy_thread);
    free(pos_xy_thread);
    free(avg_mag_thread);
    free(chpr_chim_thread);
}

void integrate_simulator_gpu(simulator_t *s, v3d field, current_t cur, const char* file_name) {
    FILE *fly = file_open(file_name, "wb", 1);
    if (s->write_on_fly && (!s->doing_relax)) {
	    int steps = (int)(s->n_steps / s->write_cut);
    	int fcut = s->write_cut;
	    double dt_real = s->dt * HBAR / s->g_old.param.exchange;
        fwrite(&s->g_old.param.rows, sizeof(int), 1, fly);
        fwrite(&s->g_old.param.cols, sizeof(int), 1, fly);
        fwrite(&steps, sizeof(int), 1, fly);
    	fwrite(&fcut, sizeof(int), 1, fly);
	    fwrite(&dt_real, sizeof(double), 1, fly);
    	fwrite(&s->g_old.param.lattice, sizeof(double), 1, fly);
    }

    memset(s->velxy_Ez, 0, sizeof(v3d) * s->n_steps / s->write_vel_charge_cut);
    memset(s->pos_xy, 0, sizeof(v3d) * s->n_steps / s->write_vel_charge_cut);
    memset(s->avg_mag, 0, sizeof(v3d) * s->n_steps / s->write_vel_charge_cut);
    memset(s->chpr_chim, 0, sizeof(v3d) * s->n_steps / s->write_vel_charge_cut);

    clw_set_kernel_arg(s->gpu.kernels[2], 0, sizeof(cl_mem), &s->g_old_buffer);
    clw_set_kernel_arg(s->gpu.kernels[2], 1, sizeof(cl_mem), &s->g_new_buffer);

    clw_set_kernel_arg(s->gpu.kernels[3], 0, sizeof(cl_mem), &s->g_old_buffer);
    clw_set_kernel_arg(s->gpu.kernels[3], 1, sizeof(cl_mem), &s->g_new_buffer);
    clw_set_kernel_arg(s->gpu.kernels[3], 2, sizeof(v3d), &field);
    clw_set_kernel_arg(s->gpu.kernels[3], 3, sizeof(double), &s->dt);
    clw_set_kernel_arg(s->gpu.kernels[3], 4, sizeof(current_t), &cur);

    // int cut = s->write_vel_charge_cut;
    clw_set_kernel_arg(s->gpu.kernels[3], 7, sizeof(int), &s->write_vel_charge_cut);
    clw_set_kernel_arg(s->gpu.kernels[3], 9, sizeof(int), &s->calculate_energy);

    size_t global = s->g_old.param.total;
    size_t local = gcd(global, 32);

    v3d* vxvy_Ez_avg_mag_chpr_chim = (v3d*)calloc(3 * s->g_old.param.total, sizeof(v3d));
    memset(vxvy_Ez_avg_mag_chpr_chim, 1, sizeof(v3d) * s->g_old.param.total * 3);
    memset(vxvy_Ez_avg_mag_chpr_chim, 0, sizeof(v3d) * s->g_old.param.total * 3);

    cl_mem vxvy_Ez_avg_mag_chpr_chim_buffer = clw_create_buffer(3 * sizeof(v3d) * s->g_old.param.total, s->gpu.ctx, CL_MEM_READ_WRITE);
    clw_write_buffer(vxvy_Ez_avg_mag_chpr_chim_buffer, vxvy_Ez_avg_mag_chpr_chim, 3 * sizeof(v3d) * s->g_old.param.total, 0, s->gpu.queue);

    clw_set_kernel_arg(s->gpu.kernels[3], 8, sizeof(cl_mem), &vxvy_Ez_avg_mag_chpr_chim_buffer);

    for (size_t i = 0; i < s->n_steps; ++i) {
        if (i % (s->n_steps / 100) == 0) {
            printf("%.3f%%\n", 100.0 * (double)i / (double)s->n_steps);
            fflush(stdout);
        }

        double norm_time = (double)i * s->dt * (!s->doing_relax);
        clw_set_kernel_arg(s->gpu.kernels[3], 5, sizeof(double), &norm_time);
        int t = i;
        clw_set_kernel_arg(s->gpu.kernels[3], 6, sizeof(int), &t);
        
        clw_enqueue_nd(s->gpu.queue, s->gpu.kernels[3], 1, NULL, &global, &local);
        clw_enqueue_nd(s->gpu.queue, s->gpu.kernels[2], 1, NULL, &global, &local);

        if (i % s->write_vel_charge_cut == 0) {
            size_t t = i / s->write_vel_charge_cut;

            /*
                So, reading the gpu_t this frequent is BAAAD
                If Visual Studio is right, this read call takes about 60%-80% of the function call (cumulative)
                The optimal way would be to create a large buffer on the gpu_t, write to that buffer via kernels, and
                after all the integration is finished, read that buffer. However, this is almost impossible without
                using too much memory.
                A typical lattice of 272x272 with 3000000 integration steps would generate a huge buffer
                3 * sizeof(double) * 272 * 272 * (3000000 / s->write_vel_charge_cut) * 2(velocity and avg_mag)
               =3 * 8 * 272 * 272 * (3000000 / 100) * 2 (values used in real simulations made by me)
               =106.5GB
                So, no, there is no way of doing a large buffer for all the integration steps
                What could work is doing a smaller buffer, and reading it in batches during the integration.
                This way instead of 3000000 steps, as used above, we use the amount os elements per batch in the buffer.
                3 * sizeof(double) * 272 * 272 * elementes per batch * 2
               =3 * 8 * 272 * 272 * 500 * 2
               =1.7GB
                So, this would be more reasonable to use. However, usually we run many programs at the same time (~40)
                and 1GB(+other memory needed, such as grid buffer and so on) per program would not fit on any RTX gpu_t,
                so, yeah, no, we cannot use that either.
                We are left with reading the buffers more frequent than what i would like,
                but it is what it is and it isn't what it isn't
            */
            clw_read_buffer(vxvy_Ez_avg_mag_chpr_chim_buffer, vxvy_Ez_avg_mag_chpr_chim, 3 * sizeof(v3d) * s->g_old.param.total, 0, s->gpu.queue);
            for (size_t k = 0; k < s->g_old.param.total; ++k) {
                size_t x = k % s->g_old.param.cols;
                size_t y = (k - x) / s->g_old.param.cols;
                s->velxy_Ez[t].x += vxvy_Ez_avg_mag_chpr_chim[k].x;
                s->velxy_Ez[t].y += vxvy_Ez_avg_mag_chpr_chim[k].y;
		s->velxy_Ez[t].z += vxvy_Ez_avg_mag_chpr_chim[k].z;

                s->pos_xy[t].x += (double)x * s->g_old.param.lattice * vxvy_Ez_avg_mag_chpr_chim[2 * s->g_old.param.total + k].x;
                s->pos_xy[t].y += (double)y * s->g_old.param.lattice * vxvy_Ez_avg_mag_chpr_chim[2 * s->g_old.param.total + k].x;

                s->avg_mag[t] = v3d_add(s->avg_mag[t], vxvy_Ez_avg_mag_chpr_chim[s->g_old.param.total + k]);

                s->chpr_chim[t] = v3d_add(s->chpr_chim[t], vxvy_Ez_avg_mag_chpr_chim[2 * s->g_old.param.total + k]);
            }

            // Moved to export phase
            /*s->velxy_Ez_chargez[t].x /= s->velxy_Ez_chargez[t].z;
            s->velxy_Ez_chargez[t].y /= s->velxy_Ez_chargez[t].z;
            s->pos_xy[t].x /= s->velxy_Ez_chargez[t].z;
            s->pos_xy[t].y /= s->velxy_Ez_chargez[t].z;*/
            s->avg_mag[t] = v3d_scalar(s->avg_mag[t], 1.0 / (double)s->g_old.param.total);
        }
        
        if (i % s->write_cut == 0) {
            size_t t = i / s->write_cut;
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
    read_v3d_grid_buffer(s->gpu.queue, s->g_old_buffer, &s->g_old);
    clw_print_cl_error(stderr, clReleaseMemObject(vxvy_Ez_avg_mag_chpr_chim_buffer), "Could not release vxvy_Ez_avg_mag_chpr_chim_buffer obj");
    if (vxvy_Ez_avg_mag_chpr_chim)
        free(vxvy_Ez_avg_mag_chpr_chim);

    fclose(fly);
}

void integrate_simulator(simulator_t *s, v3d field, current_t cur, const char* file_name) {
    if (s->use_gpu)
        integrate_simulator_gpu(s, field, cur, file_name);
    else if (s->n_cpu > 1)
        integrate_simulator_multiple(s, field, cur, file_name);
    else
        integrate_simulator_single(s, field, cur, file_name);
}

double current_normalized_to_real(double density, grid_param_t params) {
    return 2.0 * QE * params.avg_spin * fabs(params.exchange) * density / (params.lattice * params.lattice * HBAR);
}

double current_real_to_normalized(double density, grid_param_t params) {
    return params.lattice * params.lattice * HBAR * density / (2.0 * QE * params.avg_spin * fabs(params.exchange));
}

void create_skyrmion_bloch(v3d *g, int rows, int cols, int stride, int cx, int cy, int R, double P, double Q) {
    double R2 = R * R;
    for (int i = cy - 2 * R; i < cy + 2 * R; ++i) {
        double dy = (double)i - cy;
        int il = i % rows;
        if (il < 0) il += rows;
        for (int j = cx - 2 * R; j < cx + 2 * R; ++j) {
            int jl = j % cols;
            if (jl < 0) jl += cols;

            double dx = (double)j - cx;
            double r2 = dx * dx + dy * dy;
            double r = sqrt(r2);
            if (r > (2.0 * R))
                continue;
            
            g[il * stride + jl].z = 2.0 * Q * (exp(-r2 / R2) - 0.5);
        
            if (r != 0) {
                g[il * stride + jl].x = -dy * P / r * (1.0 - fabs(g[il * stride + jl].z));
                g[il * stride + jl].y = dx * P / r * (1.0 - fabs(g[il * stride + jl].z));
            }
            else {
                g[il * stride + jl].x = 0.0;
                g[il * stride + jl].y = 0.0;
            }

            g[il * stride + jl] = v3d_normalize(g[il * stride + jl]);
        }
    }
}

void create_skyrmion_neel(v3d *g, int rows, int cols, int stride, int cx, int cy, int R, double P, double Q) {
    double R2 = R * R;
    for (int i = cy - 2 * R; i < cy + 2 * R; ++i) {
        double dy = (double)i - cy;
        int il = i % rows;
        if (il < 0) il += rows;
        for (int j = cx - 2 * R; j < cx + 2 * R; ++j) {
            int jl = j % cols;
            if (jl < 0) jl += cols;

            double dx = (double)j - cx;
            double r2 = dx * dx + dy * dy;
            double r = sqrt(r2);
            if (r > (2.0 * R))
                continue;
            
            g[il * stride + jl].z = 2.0 * Q * (exp(-r2 / R2) - 0.5);
        
            if (r != 0) {
                g[il * stride + jl].x = dx * P / r * (1.0 - fabs(g[il * stride + jl].z));
                g[il * stride + jl].y = dy * P / r * (1.0 - fabs(g[il * stride + jl].z));
            }
            else {
                g[il * stride + jl].x = 0.0;
                g[il * stride + jl].y = 0.0;
            }
            g[il * stride + jl] = v3d_normalize(g[il * stride + jl]);
        }
    }
}

void create_bimeron(v3d *g, int rows, int cols, int stride, int cx, int cy, int r, double Q, double P) {
    double R2 = r * r;
    for (int i = 0; i < rows; ++i) {
        double dy = (double)i - cy;
        for (int j = 0; j < cols; ++j) {
            double dx = (double)j - cx;
            double r2 = dx * dx + dy * dy;
            double r = sqrt(r2);
            
            g[i * stride + j].x = 2.0 * P * (exp(-r2 / R2) - 0.5);
        
            if (r != 0) {
                g[i * stride + j].z = dx * Q / r * (1.0 - fabs(g[i * stride + j].x));
                g[i * stride + j].y = dy * Q / r * (1.0 - fabs(g[i * stride + j].x));
            }
            else {
                g[i * stride + j].z = 0.0;
                g[i * stride + j].y = 0.0;
            }
            g[i * stride + j] = v3d_normalize(g[i * stride + j]);
        }
    }
}

v3d total_velocity(v3d *current, v3d *before, v3d *after, int rows, int cols, double dx, double dy, double dt, pbc_t pbc) {
    v3d ret = v3d_s(0.0);
    for (size_t I = 0; I < (size_t)(rows * cols); ++I)
        ret = v3d_add(ret, velocity(I, current, before, after, rows, cols, dx, dy, dt, pbc));
    return v3d_scalar(ret, dx * dy);
}

v3d total_velocityW(v3d *current, v3d *before, v3d *after, int rows, int cols, double dx, double dy, double dt, pbc_t pbc) {
    v3d ret = v3d_s(0.0);
    for (size_t I = 0; I < (size_t)(rows * cols); ++I)
        ret = v3d_add(ret, velocity_weighted(I, current, before, after, rows, cols, dx, dy, dt, pbc));
    return ret;
}

void write_simulation_data(const char* root_path, simulator_t* s) {
    char *out_charge;
    size_t out_charge_size = snprintf(NULL, 0, "%s_charge.out", root_path) + 1;
    out_charge = (char*)calloc(out_charge_size, 1);
    snprintf(out_charge, out_charge_size, "%s_charge.out", root_path);
    out_charge[out_charge_size - 1] = '\0';

    char *out_velocity;
    size_t out_velocity_size = snprintf(NULL, 0, "%s_velocity.out", root_path) + 1;
    out_velocity = (char*)calloc(out_velocity_size, 1);
    snprintf(out_velocity, out_velocity_size, "%s_velocity.out", root_path);
    out_velocity[out_velocity_size - 1] = '\0';

    char *out_pos_xy;
    size_t out_pos_xy_size = snprintf(NULL, 0, "%s_pos_xy.out", root_path) + 1;
    out_pos_xy = (char*)calloc(out_pos_xy_size, 1);
    snprintf(out_pos_xy, out_pos_xy_size, "%s_pos_xy.out", root_path);
    out_pos_xy[out_pos_xy_size - 1] = '\0';

    char *out_avg_mag;
    size_t out_avg_mag_size = snprintf(NULL, 0, "%s_avg_mag.out", root_path) + 1;
    out_avg_mag = (char*)calloc(out_avg_mag_size, 1);
    snprintf(out_avg_mag, out_avg_mag_size, "%s_avg_mag.out", root_path);
    out_avg_mag[out_avg_mag_size - 1] = '\0';

    char *out_energy;
    size_t out_energy_size = snprintf(NULL, 0, "%s_energy.out", root_path) + 1;
    out_energy = (char*)calloc(out_energy_size, 1);
    snprintf(out_energy, out_energy_size, "%s_energy.out", root_path);
    out_energy[out_energy_size - 1] = '\0';

    FILE *charge_total = file_open(out_charge, "w", 0);
    FILE *velocity_total = file_open(out_velocity, "w", 0);
    FILE *pos_xy = file_open(out_pos_xy, "w", 0);
    FILE *avg_mag = file_open(out_avg_mag, "w", 0);
    FILE *energy = file_open(out_energy, "w", 0);

    free(out_charge);
    free(out_velocity);
    free(out_pos_xy);
    free(out_avg_mag);
    free(out_energy);

    double J_abs = fabs(s->g_old.param.exchange);
    printf("Writing charges related output\n");
    for (size_t i = 0; i < s->n_steps; ++i) {
        if (i % (s->n_steps / 10) == 0)
            printf("%.3f%%\n", (double)i / (double)s->n_steps * 100.0);
        if (i % s->write_vel_charge_cut)
            continue;

        size_t t = i / s->write_vel_charge_cut;
        double vx, vy, chpr, chim;
        vx = s->velxy_Ez[t].x;
        vy = s->velxy_Ez[t].y;
        chpr = s->chpr_chim[t].x;
        chim = s->chpr_chim[t].y;
        fprintf(velocity_total, "%e\t%e\t%e\t%e\t%e\n", (double)i * s->dt * HBAR / J_abs, vx / chpr, vy / chpr, vx / chim, vy / chim);
        fprintf(charge_total, "%e\t%e\t%e\n", (double)i * s->dt * HBAR / J_abs, chpr, chim);
        fprintf(pos_xy, "%e\t%e\t%e\n", (double)i * s->dt * HBAR / J_abs, s->pos_xy[t].x / chpr, s->pos_xy[t].y / chpr);
        fprintf(avg_mag, "%e\t%e\t%e\t%e\n", (double)i * s->dt * HBAR / J_abs, s->avg_mag[t].x, s->avg_mag[t].y, s->avg_mag[t].z);
	    fprintf(energy, "%e\t%e\n", (double)i * s->dt * HBAR / J_abs, s->velxy_Ez[t].z);
    } 
    printf("Done writing charges related output\n");
    fclose(charge_total);
    fclose(velocity_total);
    fclose(pos_xy);
    fclose(avg_mag);
    fclose(energy);

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

void create_triangular_neel_skyrmion_lattice(v3d *g, int rows, int cols, int stride, int R, int nx, double P, double Q) {
    double Sl = ((double)cols - 2.0 * R * (double)nx) / (double)nx;
    double S = Sl + 2.0 * R;
    
    double yc = sqrt(3.0) / 4.0 * S;
    int j = 0;
    while (yc < rows) {
        for (int i = 0; i < nx; ++i) {
            double xc = S / 2.0 + i * S;
            if (j % 2) xc -= S / 2.0;
            create_skyrmion_neel(g, rows, cols, stride, xc, yc, R, P, Q);
        }
        yc += sqrt(3.0) / 2.0 * S;
        ++j;
    }
}

void create_triangular_bloch_skyrmion_lattice(v3d *g, int rows, int cols, int stride, int R, int nx, double P, double Q) {
    double Sl = ((double)cols - 2.0 * R * (double)nx) / (double)nx;
    double S = Sl + 2.0 * R;
    
    double yc = sqrt(3.0) / 4.0 * S;
    int j = 0;
    while (yc < rows) {
        for (int i = 0; i < nx; ++i) {
            double xc = S / 2.0 + i * S;
            if (j % 2) xc -= S / 2.0;
            create_skyrmion_bloch(g, rows, cols, stride, xc, yc, R, P, Q);
        }
        yc += sqrt(3.0) / 2.0 * S;
        ++j;
    }
}
#endif
