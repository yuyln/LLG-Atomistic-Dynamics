#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "grid_funcs.h"
#include "constants.h"

#define CHECK_BOUNDS(size, loc) do { if (loc.row >= (int64_t)size.rows || loc.col >= (int64_t)size.cols || loc.depth >= (int64_t)size.depths || \
    loc.row < 0 || loc.col < 0 || loc.depth < 0) { \
    fprintf(stderr, "[ WARNING ] Location (%d %d %d) out of bounds (%d %d %d)\n", loc.row, loc.col, loc.depth, \
                                                                                     size.rows, size.cols, size.depths); \
    return; \
}} while(0)

double shit_random(double from, double to) {
    double r = (double)rand() / (double)RAND_MAX;
    return from + r * (to - from);
}

grid grid_init(matrix_size size) {
    grid ret = {0};
    ret.g_info.size = size;
    ret.g_info.pbc = (pbc_rules){.dirs = (1 << 0) | (1 << 1) | (1 << 2), .m = {0}};
    //ret.g_info.total_time = 100 * NS;
    ret.gsp = calloc(sizeof(*ret.gsp) * size.rows * size.cols * size.depths, 1);
    ret.m = calloc(sizeof(*ret.m) * size.rows * size.cols * size.depths, 1);

    if (!ret.gsp || !ret.m) {
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
        .total_time = 100 * NS
    };

    for (int depth = 0; depth < size.depths; ++depth) {
        for (int row = 0; row < size.rows; ++row) { 
            for (int col = 0; col < size.cols; ++col) {
                ret.gsp[LOC(row, col, depth, size.rows, size.cols)] = default_grid;
                ret.gsp[LOC(row, col, depth, size.rows, size.cols)].loc = (matrix_loc){.row = row, .col = col, .depth = depth};
                ret.m[LOC(row, col, depth, size.rows, size.cols)] = v3d_s(0);
            }
        }
    }

    return ret;
}

void grid_set_exchange_loc(grid *g, matrix_loc loc, double exchange) {
    CHECK_BOUNDS(g->g_info.size, loc);
    g->gsp[LOC(loc.row, loc.col, loc.depth, g->g_info.size.rows, g->g_info.size.cols)].exchange = exchange;
}

void grid_set_dm_loc(grid *g, matrix_loc loc, double dm, double dm_ani, dm_symmetry dm_sym) {
    CHECK_BOUNDS(g->g_info.size, loc);
    g->gsp[LOC(loc.row, loc.col, loc.depth, g->g_info.size.rows, g->g_info.size.cols)].dm = dm;
    g->gsp[LOC(loc.row, loc.col, loc.depth, g->g_info.size.rows, g->g_info.size.cols)].dm_ani = dm_ani;
    g->gsp[LOC(loc.row, loc.col, loc.depth, g->g_info.size.rows, g->g_info.size.cols)].dm_sym = dm_sym;
}

void grid_set_lattice_loc(grid *g, matrix_loc loc, double lattice) {
    CHECK_BOUNDS(g->g_info.size, loc);
    g->gsp[LOC(loc.row, loc.col, loc.depth, g->g_info.size.rows, g->g_info.size.cols)].lattice = lattice;
}

void grid_set_cubic_anisotropy_loc(grid *g, matrix_loc loc, double cubic_ani) {
    CHECK_BOUNDS(g->g_info.size, loc);
    g->gsp[LOC(loc.row, loc.col, loc.depth, g->g_info.size.rows, g->g_info.size.cols)].cubic_ani = cubic_ani;
}

void grid_set_mu_loc(grid *g, matrix_loc loc, double mu) {
    CHECK_BOUNDS(g->g_info.size, loc);
    g->gsp[LOC(loc.row, loc.col, loc.depth, g->g_info.size.rows, g->g_info.size.cols)].mu = mu;
}

void grid_set_alpha_loc(grid *g, matrix_loc loc, double alpha) {
    CHECK_BOUNDS(g->g_info.size, loc);
    g->gsp[LOC(loc.row, loc.col, loc.depth, g->g_info.size.rows, g->g_info.size.cols)].alpha = alpha;
}

void grid_set_gamma_loc(grid *g, matrix_loc loc, double gamma) {
    CHECK_BOUNDS(g->g_info.size, loc);
    g->gsp[LOC(loc.row, loc.col, loc.depth, g->g_info.size.rows, g->g_info.size.cols)].gamma = gamma;
}

void grid_set_anisotropy_loc(grid *g, matrix_loc loc, anisotropy ani) {
    CHECK_BOUNDS(g->g_info.size, loc);
    g->gsp[LOC(loc.row, loc.col, loc.depth, g->g_info.size.rows, g->g_info.size.cols)].ani = ani;
}

void grid_set_pinning_loc(grid *g, matrix_loc loc, pinning pin) {
    CHECK_BOUNDS(g->g_info.size, loc);
    g->gsp[LOC(loc.row, loc.col, loc.depth, g->g_info.size.rows, g->g_info.size.cols)].pin = pin;
}

void v3d_set_at_loc(v3d *g, matrix_size size, matrix_loc loc, v3d m) {
    CHECK_BOUNDS(size, loc);
    g[LOC(loc.row, loc.col, loc.depth, size.rows, size.cols)] = m;
}

void grid_set_exchange(grid *g, double exchange) {
    for (int d = 0; d < g->g_info.size.dim[2]; ++d)
        for (int r = 0; r < g->g_info.size.dim[0]; ++r)
            for (int c = 0; c < g->g_info.size.dim[1]; ++c)
                grid_set_exchange_loc(g, (matrix_loc){.dim={r, c, d}}, exchange);
}

void grid_set_dm(grid *g, double dm, double dm_ani, dm_symmetry dm_sym) {
    for (int d = 0; d < g->g_info.size.dim[2]; ++d)
        for (int r = 0; r < g->g_info.size.dim[0]; ++r)
            for (int c = 0; c < g->g_info.size.dim[1]; ++c)
                grid_set_dm_loc(g, (matrix_loc){.dim={r, c, d}}, dm, dm_ani, dm_sym);
}

void grid_set_lattice(grid *g, double lattice) {
    for (int d = 0; d < g->g_info.size.dim[2]; ++d)
        for (int r = 0; r < g->g_info.size.dim[0]; ++r)
            for (int c = 0; c < g->g_info.size.dim[1]; ++c)
                grid_set_lattice_loc(g, (matrix_loc){.dim={r, c, d}}, lattice);

}

void grid_set_cubic_anisotropy(grid *g, double cubic_ani) {
    for (int d = 0; d < g->g_info.size.dim[2]; ++d)
        for (int r = 0; r < g->g_info.size.dim[0]; ++r)
            for (int c = 0; c < g->g_info.size.dim[1]; ++c)
                grid_set_cubic_anisotropy_loc(g, (matrix_loc){.dim={r, c, d}}, cubic_ani);
}

void grid_set_mu(grid *g, double mu) {
    for (int d = 0; d < g->g_info.size.dim[2]; ++d)
        for (int r = 0; r < g->g_info.size.dim[0]; ++r)
            for (int c = 0; c < g->g_info.size.dim[1]; ++c)
                grid_set_mu_loc(g, (matrix_loc){.dim={r, c, d}}, mu);
}

void grid_set_alpha(grid *g, double alpha) {
    for (int d = 0; d < g->g_info.size.dim[2]; ++d)
        for (int r = 0; r < g->g_info.size.dim[0]; ++r)
            for (int c = 0; c < g->g_info.size.dim[1]; ++c)
                grid_set_alpha_loc(g, (matrix_loc){.dim={r, c, d}}, alpha);

}

void grid_set_gamma(grid *g, double gamma) {
    for (int d = 0; d < g->g_info.size.dim[2]; ++d)
        for (int r = 0; r < g->g_info.size.dim[0]; ++r)
            for (int c = 0; c < g->g_info.size.dim[1]; ++c)
                grid_set_gamma_loc(g, (matrix_loc){.dim={r, c, d}}, gamma);
}

void grid_set_anisotropy(grid *g, anisotropy ani) {
    for (int d = 0; d < g->g_info.size.dim[2]; ++d)
        for (int r = 0; r < g->g_info.size.dim[0]; ++r)
            for (int c = 0; c < g->g_info.size.dim[1]; ++c)
                grid_set_anisotropy_loc(g, (matrix_loc){.dim={r, c, d}}, ani);
}

void grid_free(grid *g) {
    free(g->gsp);
    free(g->m);
    memset(g, 0, sizeof(grid));
}

cl_mem grid_to_gpu(grid *g, gpu_cl gpu) {
    int gsp_size_bytes = g->g_info.size.rows * g->g_info.size.cols * g->g_info.size.depths * sizeof(*g->gsp);
    int v3d_size_bytes = g->g_info.size.rows * g->g_info.size.cols * g->g_info.size.depths * sizeof(*g->m);
    int grid_size_bytes = gsp_size_bytes + v3d_size_bytes;

    cl_int err;
    cl_mem g_buffer = clCreateBuffer(gpu.ctx, CL_MEM_READ_WRITE, grid_size_bytes, NULL, &err);
    clw_print_cl_error(stderr, err, "[ FATAL ] Creating grid with size %d B to GPU", grid_size_bytes);

    err = clEnqueueWriteBuffer(gpu.queue, g_buffer, CL_TRUE, 0, gsp_size_bytes, g->gsp, 0, NULL, NULL);
    clw_print_cl_error(stderr, err, "[ FATAL ] Writing grid site params with size %d B to GPU", gsp_size_bytes);

    err = clEnqueueWriteBuffer(gpu.queue, g_buffer, CL_TRUE, gsp_size_bytes, v3d_size_bytes, g->m, 0, NULL, NULL);
    clw_print_cl_error(stderr, err, "[ FATAL ] Writing grid vectors with size %d B to GPU", gsp_size_bytes);

    return g_buffer;
}

void grid_from_gpu(grid *g, cl_mem g_buffer, gpu_cl gpu) {
    int gsp_size_bytes = g->g_info.size.rows * g->g_info.size.cols * g->g_info.size.depths * sizeof(*g->gsp);
    int v3d_size_bytes = g->g_info.size.rows * g->g_info.size.cols * g->g_info.size.depths * sizeof(*g->m);

    cl_int err = clEnqueueReadBuffer(gpu.queue, g_buffer, CL_TRUE, 0, gsp_size_bytes, g->gsp, 0, NULL, NULL);
    clw_print_cl_error(stderr, err, "[ FATAL ] Reading grid site params with size %d B from GPU", gsp_size_bytes);

    err = clEnqueueReadBuffer(gpu.queue, g_buffer, CL_TRUE, gsp_size_bytes, v3d_size_bytes, g->m, 0, NULL, NULL);
    clw_print_cl_error(stderr, err, "[ FATAL ] Reading grid vectors with size %d B from GPU", v3d_size_bytes);
}

INCEPTION("[ DANGER ] using sizeof on type, not on variable, check on changes")
void v3d_from_gpu(v3d *g, matrix_size sz, cl_mem grid_buffer, gpu_cl gpu) {
    int gsp_size_bytes = sz.rows * sz.cols * sz.depths * sizeof(grid_site_param);
    int v3d_size_bytes = sz.rows * sz.cols * sz.depths * sizeof(*g);

    cl_int err = clEnqueueReadBuffer(gpu.queue, grid_buffer, CL_TRUE, gsp_size_bytes, v3d_size_bytes, g, 0, NULL, NULL);
    clw_print_cl_error(stderr, err, "[ FATAL ] Reading grid vectors with size %d B from GPU", v3d_size_bytes);
}

void v3d_dump(FILE *f, v3d *v, matrix_size sz) {
    fwrite(v, sz.dim[0] * sz.dim[1] * sz.dim[2] * sizeof(*v), 1, f);
}

void grid_full_dump(FILE *f, grid *g) {
    fwrite(&g->g_info, sizeof(g->g_info), 1, f);
    fwrite(g->gsp, sizeof(*g->gsp) * g->g_info.size.dim[0] * g->g_info.size.dim[1] * g->g_info.size.dim[2], 1, f);
    fwrite(g->m, sizeof(*g->m) * g->g_info.size.dim[0] * g->g_info.size.dim[1] * g->g_info.size.dim[2], 1, f);
}