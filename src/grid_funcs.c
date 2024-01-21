#include <errno.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "grid_funcs.h"
#include "constants.h"
#include "string_view.h"
#include "logging.h"

#define CHECK_BOUNDS(rows, cols, row, col) do { if (row >= (int)rows || col >= (int)cols || \
        row < 0 || col < 0) { \
    logging_log(LOG_WARNING, "Location (%d %d) out of bounds (%u %u)", row, col, rows, cols);\
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
    ret.gi.pbc = (pbc_rules){.pbc_x = true, .pbc_y = true, .m = {0}};
    ret.gp = calloc(sizeof(*ret.gp) * rows * cols, 1);
    ret.m = calloc(sizeof(*ret.m) * rows * cols, 1);
    ret.on_gpu = false;

    if (!ret.gp || !ret.m)
        logging_log(LOG_FATAL, "Could not allocate grid. Buy more ram lol");

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
    v3d_fill_with_random(ret.m, rows, cols);

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

void v3d_fill_with_random(v3d *v, unsigned int rows, unsigned int cols) {
    for (unsigned int r = 0; r < rows; ++r)
        for (unsigned int c = 0; c < cols; ++c)
            v3d_set_at_loc(v, rows, cols, r, c, v3d_normalize(v3d_c(shit_random(-1.0, 1.0), shit_random(-1.0, 1.0), shit_random(-1.0, 1.0))));
}

void v3d_create_skyrmion(v3d *v, unsigned int rows, unsigned int cols, int radius, int row, int col, double Q, double P, double theta) {
    double R2 = radius * radius;
    for (int i = row - 2 * radius; i < row + 2 * radius; ++i) {
        double dy = (double)i - row;
        int il = ((i % rows) + rows) % rows;
        for (int j = col - 2 * radius; j < col + 2 * radius; ++j) {
            int jl = ((j % cols) + cols) % cols;

            double dx = (double)j - col;
            double r2 = dx * dx + dy * dy;

            double r = sqrt(r2);

            if (r > (2.0 * radius))
                continue;

            v[il * cols + jl].z = 2.0 * Q * (exp(-r2 / R2) - 0.5);

            if (r != 0) {
                v[il * cols + jl].x = (-dy * cos(theta) + dx * sin(theta)) * P / r * (1.0 - fabs(v[il * cols + jl].z));
                v[il * cols + jl].y = (dx * cos(theta) + dy * sin(theta)) * P / r * (1.0 - fabs(v[il * cols + jl].z));
            } else {
                v[il * cols + jl].x = 0.0;
                v[il * cols + jl].y = 0.0;
            }
        }
    }
}

bool grid_free(grid *g) {
    bool ret = true;
    free(g->gp);
    free(g->m);
    if (g->on_gpu)
        ret = grid_release_from_gpu(g);
    memset(g, 0, sizeof(*g));
    return ret;
}

bool grid_release_from_gpu(grid *g) {
    cl_int err;
    bool ret = true;

    if ((err = clReleaseMemObject(g->gp_buffer)) != CL_SUCCESS) {
        logging_log(LOG_ERROR, "Could not release Grid Parameters Buffer from GPU %d: %s", err, gpu_cl_get_string_error(err));
        ret = false;
    }

    if ((err = clReleaseMemObject(g->m_buffer)) != CL_SUCCESS) {
        logging_log(LOG_ERROR, "Could not release Grid Magnetic Moments from GPU %d: %s", err, gpu_cl_get_string_error(err));
        ret = false;
    }

    g->on_gpu = false;
    return ret;
}

void grid_to_gpu(grid *g, gpu_cl gpu) {
    uint64_t gp_size_bytes = g->gi.rows * g->gi.cols * sizeof(*g->gp);
    uint64_t m_size_bytes = g->gi.rows * g->gi.cols * sizeof(*g->m);

    if (g->on_gpu) {
        logging_log(LOG_WARNING, "Trying to send Grid to GPU with the grid already being on the gpu, only write will be performed");
        goto writing;
    }

    cl_int err;
    g->gp_buffer = clCreateBuffer(gpu.ctx, CL_MEM_READ_WRITE, gp_size_bytes, NULL, &err);
    if (err != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not create Grid Parameters on GPU %d: %s", err, gpu_cl_get_string_error(err));

    g->m_buffer = clCreateBuffer(gpu.ctx, CL_MEM_READ_WRITE, m_size_bytes, NULL, &err);
    if (err != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not create Grid Vectors on GPU %d: %s", err, gpu_cl_get_string_error(err));

    g->on_gpu = true;
    
writing:
    if ((err = clEnqueueWriteBuffer(gpu.queue, g->gp_buffer, CL_TRUE, 0, gp_size_bytes, g->gp, 0, NULL, NULL)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not write to Grid Parameters on GPU %d: %s", err, gpu_cl_get_string_error(err));

    if ((err = clEnqueueWriteBuffer(gpu.queue, g->m_buffer, CL_TRUE, 0, m_size_bytes, g->m, 0, NULL, NULL) != CL_SUCCESS))
        logging_log(LOG_FATAL, "Could not write to Grid Vectors on GPU %d: %s", err, gpu_cl_get_string_error(err));
}

void grid_from_gpu(grid *g, gpu_cl gpu) {
    if (!g->on_gpu) {
        logging_log(LOG_WARNING, "Trying to read Grid from GPU without the Grid being on the GPU");
        return;
    }

    int gp_size_bytes = g->gi.rows * g->gi.cols * sizeof(*g->gp);
    int m_size_bytes = g->gi.rows * g->gi.cols * sizeof(*g->m);
    cl_int err;

    if ((err = clEnqueueReadBuffer(gpu.queue, g->gp_buffer, CL_TRUE, 0, gp_size_bytes, g->gp, 0, NULL, NULL)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not read from Grid Parameters on GPU %d: %s", err, gpu_cl_get_string_error(err));

    if ((err = clEnqueueReadBuffer(gpu.queue, g->m_buffer, CL_TRUE, 0, m_size_bytes, g->m, 0, NULL, NULL) != CL_SUCCESS))
        logging_log(LOG_FATAL, "Could not read from Grid Vectors on GPU %d: %s", err, gpu_cl_get_string_error(err));
}

void v3d_from_gpu(v3d *g, cl_mem buffer, unsigned int rows, unsigned int cols, gpu_cl gpu) {
    int m_size_bytes = rows * cols * sizeof(*g);
    cl_int err;

    if ((err = clEnqueueReadBuffer(gpu.queue, buffer, CL_TRUE, 0, m_size_bytes, g, 0, NULL, NULL) != CL_SUCCESS))
        logging_log(LOG_FATAL, "Could not read from Vectors on GPU %d: %s", err, gpu_cl_get_string_error(err));
}

bool v3d_dump(FILE *f, v3d *v, unsigned int rows, unsigned int cols) {
    return rows * cols * sizeof(*v) == fwrite(v, rows * cols * sizeof(*v), 1, f);
}

bool grid_dump(FILE *f, grid *g) {
    bool ret = sizeof(g->gi) == fwrite(&g->gi, 1, sizeof(g->gi),  f);
    ret = ret && (sizeof(*g->gp) * g->gi.rows * g->gi.cols) == fwrite(g->gp, 1, sizeof(*g->gp) * g->gi.rows * g->gi.cols, f);
    ret = ret && (sizeof(*g->m) * g->gi.rows * g->gi.cols) == fwrite(g->m, 1, sizeof(*g->m) * g->gi.rows * g->gi.cols, f);
    return ret;
}

bool grid_from_file(string_view path, grid *g) {
    string p_ = {0};
    string_add_sv(&p_, path);
    FILE *f = fopen(string_as_cstr(&p_), "rb");
    char *data = NULL;
    bool ret = true;

    if (!g)
        logging_log(LOG_FATAL, "NULL pointer to grid provided");

    if (g->m || g->gp || g->on_gpu || g->gi.cols || g->gi.rows)
        logging_log(LOG_FATAL, "Trying to initialize grid from file with grid already initialized");

    if (!f) {
        logging_log(LOG_WARNING, "Could not open file %.*s: %s. Using defaults", (int)path.len, path.str, strerror(errno));
        *g = grid_init(272, 272);
        return false;
    }
    string_free(&p_);

    if (fseek(f, 0, SEEK_END) < 0) {
        logging_log(LOG_ERROR, "Moving cursor to the end of %.*s failed: %s", (int)path.len, path.str, strerror(errno));
        ret = false;
        goto defer;
    }

    long sz;
    if ((sz = ftell(f)) < 0) {
        logging_log(LOG_ERROR, "Getting cursor position of %.*s failed: %s", (int)path.len, path.str, strerror(errno));
        ret = false;
        goto defer;
    }

    if (fseek(f, 0, SEEK_SET) < 0) {
        logging_log(LOG_ERROR, "Moving cursor to start of %.*s failed: %s", (int)path.len, path.str, strerror(errno));
        ret = false;
        goto defer;
    }

    if (!(data = calloc(sz + 1, 1))) {
        logging_log(LOG_ERROR, "Callocing data from %.*s failed: %s", (int)path.len, path.str, strerror(errno));
        ret = false;
        goto defer;
    }

    char *ptr = data;
    if (fread(data, 1, sz, f) != (uint64_t)sz) {
        logging_log(LOG_ERROR, "Reading data from %.*s failed: %s", (int)path.len, path.str, strerror(errno));
        ret = false;
        goto defer;
    }

    g->gi = *((grid_info*)data);
    ptr += sizeof(grid_info);

    if (!(g->gp = calloc(sizeof(*g->gp) * g->gi.rows * g->gi.cols , 1)))
        logging_log(LOG_FATAL, "Callocing Grid Parameters data from %.*s failed: %s", (int)path.len, path.str, strerror(errno));

    if (!(g->m = calloc(sizeof(*g->m) * g->gi.rows * g->gi.cols, 1)))
        logging_log(LOG_FATAL, "Callocing Grid Vectors data from %.*s failed: %s", (int)path.len, path.str, strerror(errno));

    g->on_gpu = false;

    memcpy(g->gp, ptr, sizeof(*g->gp) * g->gi.rows * g->gi.cols);
    ptr += sizeof(*g->gp) * g->gi.rows * g->gi.cols;

    memcpy(g->m, ptr, sizeof(*g->m) * g->gi.rows * g->gi.cols);
defer:
    fclose(f);
    free(data);
    return ret;
}
