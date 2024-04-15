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
    double dm = 0.18 * QE * 1.0e-3;

    dm_interaction default_dm = (dm_interaction){.dmv_down = v3d_c(dm, 0, 0),
                                                 .dmv_up = v3d_c(-dm, 0, 0),
                                                 .dmv_left = v3d_c(0, -dm, 0),
                                                 .dmv_right = v3d_c(0, dm, 0)};

    grid_site_params default_grid = (grid_site_params){
            .exchange = 1.0e-3 * QE,
            .dm = default_dm,
            .lattice = 5.0e-10,
            .cubic_ani = 0.0,
            .mu = 1.856952954255053e-23,
            .alpha = 0.3,
            .gamma = 1.760859644000000e+11,
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

void grid_set_dm_loc(grid *g, int row, int col, dm_interaction dm) {
    CHECK_BOUNDS(g->gi.rows, g->gi.cols, row, col);
    g->gp[row * g->gi.cols + col].dm = dm;
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

void grid_set_dm(grid *g, dm_interaction dm) {
    for (unsigned int r = 0; r < g->gi.rows; ++r)
        for (unsigned int c = 0; c < g->gi.cols; ++c)
            grid_set_dm_loc(g, r, c, dm);
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

            if (r > (1.0 * radius))
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
    bool ret = true;
    gpu_cl_release_memory(g->gp_buffer);
    gpu_cl_release_memory(g->m_buffer);
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

    g->gp_buffer = gpu_cl_create_buffer(&gpu, gp_size_bytes, CL_MEM_READ_WRITE);
    g->m_buffer = gpu_cl_create_buffer(&gpu, m_size_bytes, CL_MEM_READ_WRITE);
    g->on_gpu = true;
    
writing:
    gpu_cl_write_buffer(&gpu, gp_size_bytes, 0, g->gp, g->gp_buffer);
    gpu_cl_write_buffer(&gpu, m_size_bytes, 0, g->m, g->m_buffer);
}

void grid_from_gpu(grid *g, gpu_cl gpu) {
    if (!g->on_gpu) {
        logging_log(LOG_WARNING, "Trying to read Grid from GPU without the Grid being on the GPU");
        return;
    }

    int gp_size_bytes = g->gi.rows * g->gi.cols * sizeof(*g->gp);
    int m_size_bytes = g->gi.rows * g->gi.cols * sizeof(*g->m);

    gpu_cl_read_buffer(&gpu, gp_size_bytes, 0, g->gp, g->gp_buffer);
    gpu_cl_read_buffer(&gpu, m_size_bytes, 0, g->m, g->m_buffer);
}

void v3d_from_gpu(v3d *g, cl_mem buffer, unsigned int rows, unsigned int cols, gpu_cl gpu) {
    int m_size_bytes = rows * cols * sizeof(*g);
    gpu_cl_read_buffer(&gpu, m_size_bytes, 0, g, buffer);
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

bool grid_from_file(string path, grid *g) {
    if (!g)
        logging_log(LOG_FATAL, "NULL pointer to grid provided");

    if (g->m || g->gp || g->on_gpu || g->gi.cols || g->gi.rows)
        logging_log(LOG_FATAL, "Trying to initialize grid from file with grid already initialized");

    string p_ = str_from_cstr("");
    str_cat_str(&p_, path);
    FILE *f = fopen(str_as_cstr(&p_), "rb");
    char *data = NULL;
    bool ret = true;

    if (!f) {
        logging_log(LOG_WARNING, "Could not open file %.*s: %s. Using defaults", (int)path.len, path.str, strerror(errno));
        *g = grid_init(272, 272);
        return false;
    }
    str_free(&p_);

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

void grid_do_in_rect(grid *g, int64_t x0, int64_t y0, int64_t x1, int64_t y1, void(*fun)(grid *g, uint64_t row, uint64_t col)) {
    for (int64_t y = y0; y < y1; ++y)
        for (int64_t x = x0; x < x1; ++x)
            if (x >= 0 && x < g->gi.cols && y >= 0 && y < g->gi.rows)
                fun(g, y, x);

}

void grid_do_in_ellipse(grid *g, int64_t x0, int64_t y0, int64_t a, int64_t b, void(*fun)(grid *g, uint64_t row, uint64_t col)) {
    for (int64_t y = y0 - b; y < y0 + b; ++y)
        for (int64_t x = x0 - a; x < x0 + a; ++x)
            if (x >= 0 && x < g->gi.cols && y >= 0 && y < g->gi.rows)
                if (((x - x0) * (x - x0) / (double)(a * a) + (y - y0) * (y - y0) / (double)(b * b)) <= 1)
                    fun(g, y, x);
}

static bool triangle_inside(double x, double y, double x0, double y0, double x1, double y1, double x2, double y2) {
	double alpha = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) /
				   ((y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2));
	double beta = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) /
				  ((y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2));
	double gamma = 1.0 - alpha - beta;
	return alpha >= 0 && beta >= 0 && gamma >= 0;
}

static int64_t i64_max(int64_t a, int64_t b) {
    return a > b? a: b;
}

static int64_t i64_min(int64_t a, int64_t b) {
    return a < b? a: b;
}

void grid_do_in_triangle(grid *g, int64_t x0, int64_t y0, int64_t x1, int64_t y1, int64_t x2, int64_t y2, void(*fun)(grid *g, uint64_t row, uint64_t col)) {
    int64_t x_max = i64_max(x0, i64_max(x1, x2));
    int64_t x_min = i64_min(x0, i64_min(x1, x2));

    int64_t y_max = i64_max(y0, i64_max(y1, y2));
    int64_t y_min = i64_min(y0, i64_min(y1, y2));

    for (int64_t y = y_min; y < y_max; ++y)
        for (int64_t x = x_min; x < x_max; ++x)
            if (x >= 0 && x < g->gi.cols && y >= 0 && y < g->gi.rows &&
                triangle_inside(x, y, x0, y0, x1, y1, x2, y2))
                fun(g, y, x);
}

void grid_do_in_line(grid *g, int64_t x0, int64_t y0, int64_t x1, int64_t y1, int64_t thickness, void(*fun)(grid *g, uint64_t row, uint64_t col)) {
    if (x0 == x1) {
        grid_do_in_rect(g, x0 - thickness / 2, y0, x0 + thickness / 2, y1, fun);
        return;
    }

    if (y0 == y1) {
        grid_do_in_rect(g, x0, y0 - thickness / 2, x1, y0 + thickness / 2, fun);
        return;
    }

    double dy = y1 - y0;
    double dx = x1 - x0;
    double ny = dx;
    double nx = -dy;
    double M = sqrt(nx * nx + ny * ny);
    ny /= M;
    nx /= M;
    grid_do_in_triangle(g, x0 - nx * thickness / 2, y0 - ny * thickness / 2, x1 - nx * thickness / 2, y1 - ny * thickness / 2, x0 + nx * thickness / 2, y0 + ny * thickness / 2, fun);
    grid_do_in_triangle(g, x1 + nx * thickness / 2, y1 + ny * thickness / 2, x1 - nx * thickness / 2, y1 - ny * thickness / 2, x0 + nx * thickness / 2, y0 + ny * thickness / 2, fun);
}

