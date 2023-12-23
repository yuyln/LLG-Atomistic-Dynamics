#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "grid_funcs.h"
#include "constants.h"

#define CHECK_BOUNDS(rows, cols, row, col) do { if (row >= (int)rows || col >= (int)cols || \
    row < 0 || col < 0) { \
    fprintf(stderr, "[ WARNING ] Location (%d %d) out of bounds (%u %u)\n", row, col, rows, cols);\
    return; \
}} while(0)

double shit_random(double from, double to) {
    double r = (double)rand() / (double)RAND_MAX;
    return from + r * (to - from);
}

grid grid_init(unsigned int rows, unsigned int cols) {
    grid ret = {0};
    ret.gi.rows = rows;
    ret.gi.cols = cols;
    ret.gi.pbc = (pbc_rules){.dirs = (1 << 0) | (1 << 1) | (1 << 2), .m = {0}};
    ret.gp = calloc(sizeof(*ret.gp) * rows * cols, 1);
    ret.m = calloc(sizeof(*ret.m) * rows * cols, 1);

    if (!ret.gp || !ret.m) {
        fprintf(stderr, "[ FATAL ] Could not allocate grid. Buy more ram lol");
        exit(1);
    }

    grid_site_param default_grid = (grid_site_param){
        .exchange = 1.0e-3 * QE,
            .dm = 0.18 * 1.0e-3 * QE,
            .dm_ani = 0.0,
            .lattice = 5.0e-10,
            .cubic_ani = 0.0,
            .mu = 1.856952954255053e-23,
            .alpha = 0.3,
            .gamma = 1.760859644000000e+11,
            .dm_sym = R_ij_CROSS_Z,
            .ani = {{0}},
            .pin = {{0}},
    };

    for (unsigned int row = 0; row < rows; ++row) { 
        for (unsigned int col = 0; col < cols; ++col) {
            ret.gp[row * cols + col] = default_grid;
            ret.gp[row * cols + col].row = row;
            ret.gp[row * cols + col].col = col;
            ret.m[row * cols + col] = v3d_s(0);
        }
    }

    return ret;
}

void grid_set_exchange_loc(grid *g, int row, int col, double exchange) {
    CHECK_BOUNDS(g->gi.rows, g->gi.cols, row, col);
    g->gp[row * g->gi.cols + col].exchange = exchange;
}

void grid_set_dm_loc(grid *g, int row, int col, double dm, double dm_ani, dm_symmetry dm_sym) {
    CHECK_BOUNDS(g->gi.rows, g->gi.cols, row, col);
    g->gp[row * g->gi.cols + col].dm = dm;
    g->gp[row * g->gi.cols + col].dm_ani = dm_ani;
    g->gp[row * g->gi.cols + col].dm_sym = dm_sym;
}

void grid_set_lattice_loc(grid *g, int row, int col, double lattice) {
    CHECK_BOUNDS(g->gi.rows, g->gi.cols, row, col);
    g->gp[row * g->gi.cols + col].lattice = lattice;
}

void grid_set_cubic_anisotropy_loc(grid *g, int row, int col, double cubic_ani) {
    CHECK_BOUNDS(g->gi.rows, g->gi.cols, row, col);
    g->gp[row * g->gi.cols + col].cubic_ani = cubic_ani;
}

void grid_set_mu_loc(grid *g, int row, int col, double mu) {
    CHECK_BOUNDS(g->gi.rows, g->gi.cols, row, col);
    g->gp[row * g->gi.cols + col].mu = mu;
}

void grid_set_alpha_loc(grid *g, int row, int col, double alpha) {
    CHECK_BOUNDS(g->gi.rows, g->gi.cols, row, col);
    g->gp[row * g->gi.cols + col].alpha = alpha;
}

void grid_set_gamma_loc(grid *g, int row, int col, double gamma) {
    CHECK_BOUNDS(g->gi.rows, g->gi.cols, row, col);
    g->gp[row * g->gi.cols + col].gamma = gamma;
}

void grid_set_anisotropy_loc(grid *g, int row, int col, anisotropy ani) {
    CHECK_BOUNDS(g->gi.rows, g->gi.cols, row, col);
    g->gp[row * g->gi.cols + col].ani = ani;
}

void grid_set_pinning_loc(grid *g, int row, int col, pinning pin) {
    CHECK_BOUNDS(g->gi.rows, g->gi.cols, row, col);
    g->gp[row * g->gi.cols + col].pin = pin;
}

void v3d_set_at_loc(v3d *g, unsigned int rows, unsigned int cols, int row, int col, v3d m) {
    CHECK_BOUNDS(rows, cols, row, col);
    g[row * cols + col] = m;
}

void grid_set_exchange(grid *g, double exchange) {
        for (unsigned int r = 0; r < g->gi.rows; ++r)
            for (unsigned int c = 0; c < g->gi.cols; ++c)
                grid_set_exchange_loc(g, r, c, exchange);
}

void grid_set_dm(grid *g, double dm, double dm_ani, dm_symmetry dm_sym) {
    for (unsigned int r = 0; r < g->gi.rows; ++r)
        for (unsigned int c = 0; c < g->gi.cols; ++c)
            grid_set_dm_loc(g, r, c, dm, dm_ani, dm_sym);
}

void grid_set_lattice(grid *g, double lattice) {
    for (unsigned int r = 0; r < g->gi.rows; ++r)
        for (unsigned int c = 0; c < g->gi.cols; ++c)
            grid_set_lattice_loc(g, r, c, lattice);

}

void grid_set_cubic_anisotropy(grid *g, double cubic_ani) {
        for (unsigned int r = 0; r < g->gi.rows; ++r)
            for (unsigned int c = 0; c < g->gi.cols; ++c)
                grid_set_cubic_anisotropy_loc(g, r, c, cubic_ani);
}

void grid_set_mu(grid *g, double mu) {
        for (unsigned int r = 0; r < g->gi.rows; ++r)
            for (unsigned int c = 0; c < g->gi.cols; ++c)
                grid_set_mu_loc(g, r, c, mu);
}

void grid_set_alpha(grid *g, double alpha) {
        for (unsigned int r = 0; r < g->gi.rows; ++r)
            for (unsigned int c = 0; c < g->gi.cols; ++c)
                grid_set_alpha_loc(g, r, c, alpha);

}

void grid_set_gamma(grid *g, double gamma) {
        for (unsigned int r = 0; r < g->gi.rows; ++r)
            for (unsigned int c = 0; c < g->gi.cols; ++c)
                grid_set_gamma_loc(g, r, c, gamma);
}

void grid_set_anisotropy(grid *g, anisotropy ani) {
        for (unsigned int r = 0; r < g->gi.rows; ++r)
            for (unsigned int c = 0; c < g->gi.cols; ++c)
                grid_set_anisotropy_loc(g, r, c, ani);
}

void grid_free(grid *g) {
    free(g->gp);
    free(g->m);
    g->gp = NULL;
    g->m = NULL;
    memset(g, 0, sizeof(g->gi));
}

void grid_release_from_gpu(grid *g) {
    grid_free(g);
    clw_print_cl_error(stderr, clReleaseMemObject(g->gp_buffer), "[ FATAL ] Could not release gp buffer from GPU");
    clw_print_cl_error(stderr, clReleaseMemObject(g->m_buffer), "[ FATAL ] Could not release m buffer from GPU");
}

void grid_to_gpu(grid *g, gpu_cl gpu) {
    int gp_size_bytes = g->gi.rows * g->gi.cols * sizeof(*g->gp);
    int m_size_bytes = g->gi.rows * g->gi.cols * sizeof(*g->m);

    cl_int err;
    g->gp_buffer = clCreateBuffer(gpu.ctx, CL_MEM_READ_WRITE, gp_size_bytes, NULL, &err);
    clw_print_cl_error(stderr, err, "[ FATAL ] Creating grid site param buffer with size %d B to GPU", gp_size_bytes);

    g->m_buffer = clCreateBuffer(gpu.ctx, CL_MEM_READ_WRITE, m_size_bytes, NULL, &err);
    clw_print_cl_error(stderr, err, "[ FATAL ] Creating vectors grid buffer with size %d B to GPU", m_size_bytes);

    err = clEnqueueWriteBuffer(gpu.queue, g->gp_buffer, CL_TRUE, 0, gp_size_bytes, g->gp, 0, NULL, NULL);
    clw_print_cl_error(stderr, err, "[ FATAL ] Writing grid site params with size %d B to GPU", gp_size_bytes);

    err = clEnqueueWriteBuffer(gpu.queue, g->m_buffer, CL_TRUE, 0, m_size_bytes, g->m, 0, NULL, NULL);
    clw_print_cl_error(stderr, err, "[ FATAL ] Writing grid vectors with size %d B to GPU", m_size_bytes);
}

void grid_from_gpu(grid *g, gpu_cl gpu) {
    int gp_size_bytes = g->gi.rows * g->gi.cols * sizeof(*g->gp);
    int m_size_bytes = g->gi.rows * g->gi.cols * sizeof(*g->m);

    cl_int err = clEnqueueReadBuffer(gpu.queue, g->gp_buffer, CL_TRUE, 0, gp_size_bytes, g->gp, 0, NULL, NULL);
    clw_print_cl_error(stderr, err, "[ FATAL ] Reading grid site params with size %d B from GPU", gp_size_bytes);

    err = clEnqueueReadBuffer(gpu.queue, g->m_buffer, CL_TRUE, 0, m_size_bytes, g->m, 0, NULL, NULL);
    clw_print_cl_error(stderr, err, "[ FATAL ] Reading grid vectors with size %d B from GPU", m_size_bytes);
}

void v3d_from_gpu(v3d *g, cl_mem buffer, unsigned int rows, unsigned int cols, gpu_cl gpu) {
    int m_size_bytes = rows * cols * sizeof(*g);

    cl_int err = clEnqueueReadBuffer(gpu.queue, buffer, CL_TRUE, 0, m_size_bytes, g, 0, NULL, NULL);
    clw_print_cl_error(stderr, err, "[ FATAL ] Reading grid vectors with size %d B from GPU", m_size_bytes);
}

void v3d_dump(FILE *f, v3d *v, unsigned int rows, unsigned int cols) {
    fwrite(v, rows * cols * sizeof(*v), 1, f);
}

void grid_full_dump(FILE *f, grid *g) {
    fwrite(&g->gi, sizeof(g->gi), 1, f);
    fwrite(g->gp, sizeof(*g->gp) * g->gi.rows * g->gi.cols, 1, f);
    fwrite(g->m, sizeof(*g->m) * g->gi.rows * g->gi.cols, 1, f);
}
