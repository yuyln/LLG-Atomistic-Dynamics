#include <errno.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <float.h>

#include "grid_funcs.h"
#include "constants.h"
#include "string_view.h"
#include "logging.h"
#include "allocator.h"

#define CHECK_BOUNDS(rows, cols, row, col) do { if (row >= (int)rows || col >= (int)cols || \
        row < 0 || col < 0) { \
    logging_log(LOG_WARNING, "Location (%d %d) out of bounds (%u %u)", row, col, rows, cols);\
    row = ((row % (int)rows) + (int)rows) % (int)rows; \
    col = ((col % (int)cols) + (int)cols) % (int)cols; \
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
    ret.gp = mmalloc(sizeof(*ret.gp) * rows * cols);
    ret.m = mmalloc(sizeof(*ret.m) * rows * cols);
    ret.clusters = (cluster_centers){0};
    ret.queue = (cluster_queue){0};
    ret.points = mmalloc(sizeof(*ret.points) * rows * cols);
    ret.seen = mmalloc(sizeof(*ret.seen) * rows * cols);
    ret.on_gpu = false;

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
    ani.dir = v3d_normalize(ani.dir);
    g->gp[row * g->gi.cols + col].ani = ani;
}

void grid_set_pinning_loc(grid *g, int row, int col, pinning pin) {
    CHECK_BOUNDS(g->gi.rows, g->gi.cols, row, col);
    pin.dir = v3d_normalize(pin.dir);
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

void v3d_create_skyrmion_at_old(v3d *v, unsigned int rows, unsigned int cols, int radius, int row, int col, double Q, double P, double theta) {
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

void v3d_create_skyrmion_at(v3d *v, unsigned int rows, unsigned int cols, double radius, double dw_width, double ix, double iy, double Q, double vor, double _gamma) {
    int dr = radius + dw_width;
    for (int i = -dr; i <= dr; ++i) {
        for (int j = -dr; j <= dr; ++j) {
            double r = sqrt(i * i + j * j);
            if (r > dr)
                continue;
            int x = ix + j;
            int y = iy + i;
            x = ((x % (int)cols) + (int)cols) % (int)cols;
            y = ((y % (int)rows) + (int)rows) % (int)rows;
            double phi = atan2(i, j) + M_PI;
            phi = phi * vor + _gamma;
            double theta = 2.0 * atan(pow(sinh(radius / dw_width) / sinh(r / dw_width), -Q));
            v[y * cols + x] = v3d_c(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
        }
    }
}

void v3d_create_biskyrmion_at(v3d *v, unsigned int rows, unsigned int cols, double radius, double dw, double ix, double iy, double dr, double angle, double Q, double vor, double g) {
    angle = angle - floor(angle / M_PI) * M_PI;
    for (int i = -radius - dw - dr * sin(angle); i <= radius + dw + dr * sin(angle); ++i) {
        for (int j = -radius - dw - dr * cos(angle); j <= radius + dw + dr * cos(angle); ++j) {
            int x = ix + j;
            int y = iy + i;
            x = ((x % (int)cols) + (int)cols) % (int)cols;
            y = ((y % (int)rows) + (int)rows) % (int)rows;

            int x0 = ix + dr / 2.0 * cos(angle);
            int y0 = iy + dr / 2.0 * sin(angle);

            int x1 = ix - dr / 2.0 * cos(angle);
            int y1 = iy - dr / 2.0 * sin(angle);

            x0 = ((x0 % (int)cols) + (int)cols) % (int)cols;
            y0 = ((y0 % (int)rows) + (int)rows) % (int)rows;

            x1 = ((x1 % (int)cols) + (int)cols) % (int)cols;
            y1 = ((y1 % (int)rows) + (int)rows) % (int)rows;

            double phi0 = atan2(y - y0, x - x0) + M_PI;
            double Phi0 = phi0 * vor;

            double phi1 = atan2(y - y1, x - x1) + M_PI;
            double Phi1 = phi1 * vor;

            double phi = Phi1 + Phi0 + g - angle;

            double r0 = sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0));
            double r1 = sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1));

            double inside_theta0 = pow(sinh(radius / dw) / sinh(r0 / dw), -Q);
            double inside_theta1 = pow(sinh(radius / dw) / sinh(r1 / dw), -Q);

            double theta = 2.0 * atan(inside_theta0 * inside_theta1);
            v[y * cols + x] = v3d_c(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
        }
    }
}


void v3d_uniform(v3d *v, unsigned int rows, unsigned int cols, v3d dir) {
    for (unsigned int i = 0; i < rows * cols; ++i)
        v[i] = v3d_normalize(dir);
}

void grid_fill_with_random(grid *g) {
    v3d_fill_with_random(g->m, g->gi.rows, g->gi.cols);
}

void grid_create_skyrmion_at_old(grid *g, int radius, int row, int col, double Q, double P, double theta) {
    v3d_create_skyrmion_at_old(g->m, g->gi.rows, g->gi.cols, radius, row, col, Q, P, theta);
}

void grid_create_skyrmion_at(grid *g, double radius, double dw_width, double ix, double iy, double Q, double vor, double _gamma) {
    v3d_create_skyrmion_at(g->m, g->gi.rows, g->gi.cols, radius, dw_width, ix, iy, Q, vor, _gamma);
}

void grid_uniform(grid *g, v3d dir) {
    v3d_uniform(g->m, g->gi.rows, g->gi.cols, dir);
}

void grid_create_biskyrmion_at(grid *g, double radius, double dw_width, double ix, double iy, double dr, double angle, double Q, double vorticity, double _gamma) {
    v3d_create_biskyrmion_at(g->m, g->gi.rows, g->gi.cols, radius, dw_width, ix, iy, dr, angle, Q, vorticity, _gamma);
}

bool grid_free(grid *g) {
    bool ret = true;
    mfree(g->gp);
    mfree(g->m);
    mfree(g->points);
    mfree(g->seen);

    if (g->on_gpu)
        ret = grid_release_from_gpu(g);
    
    if (g->clusters.items)
        mfree(g->clusters.items);

    if (g->queue.items)
        mfree(g->queue.items);

    memset(g, 0, sizeof(*g));
    return ret;
}

bool grid_release_from_gpu(grid *g) {
    bool ret = true;
    if (g->on_gpu) {
        gpu_cl_release_memory(g->gp_gpu);
        gpu_cl_release_memory(g->m_gpu);
    } else
        logging_log(LOG_WARNING, "Trying to release a grid that was already freed from the gpu");
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

    g->gp_gpu = gpu_cl_create_gpu(&gpu, gp_size_bytes, CL_MEM_READ_WRITE);
    g->m_gpu = gpu_cl_create_gpu(&gpu, m_size_bytes, CL_MEM_READ_WRITE);
    g->on_gpu = true;
    
writing:
    gpu_cl_write_gpu(&gpu, gp_size_bytes, 0, g->gp, g->gp_gpu);
    gpu_cl_write_gpu(&gpu, m_size_bytes, 0, g->m, g->m_gpu);
}

void grid_from_gpu(grid *g, gpu_cl gpu) {
    if (!g->on_gpu) {
        logging_log(LOG_WARNING, "Trying to read Grid from GPU without the Grid being on the GPU");
        return;
    }

    int gp_size_bytes = g->gi.rows * g->gi.cols * sizeof(*g->gp);
    int m_size_bytes = g->gi.rows * g->gi.cols * sizeof(*g->m);

    gpu_cl_read_gpu(&gpu, gp_size_bytes, 0, g->gp, g->gp_gpu);
    gpu_cl_read_gpu(&gpu, m_size_bytes, 0, g->m, g->m_gpu);
}

void v3d_from_gpu(v3d *g, cl_mem buffer, unsigned int rows, unsigned int cols, gpu_cl gpu) {
    int m_size_bytes = rows * cols * sizeof(*g);
    gpu_cl_read_gpu(&gpu, m_size_bytes, 0, g, buffer);
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
    FILE *f = mfopen(str_as_cstr(&p_), "rb");
    char *data = NULL;
    bool ret = true;

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

    data = mmalloc(sz + 1);

    char *ptr = data;
    if (fread(data, 1, sz, f) != (uint64_t)sz) {
        logging_log(LOG_ERROR, "Reading data from %.*s failed: %s", (int)path.len, path.str, strerror(errno));
        ret = false;
        goto defer;
    }

    g->gi = *((grid_info*)data);
    ptr += sizeof(grid_info);

    g->gp = mmalloc(sizeof(*g->gp) * g->gi.rows * g->gi.cols);
    g->m = mmalloc(sizeof(*g->m) * g->gi.rows * g->gi.cols);
    g->clusters = (cluster_centers){0};
    g->queue = (cluster_queue){0};
    g->points = mmalloc(sizeof(*g->points) * g->gi.rows * g->gi.cols);
    g->seen = mmalloc(sizeof(*g->seen) * g->gi.rows * g->gi.cols);

    g->on_gpu = false;

    memcpy(g->gp, ptr, sizeof(*g->gp) * g->gi.rows * g->gi.cols);
    ptr += sizeof(*g->gp) * g->gi.rows * g->gi.cols;

    memcpy(g->m, ptr, sizeof(*g->m) * g->gi.rows * g->gi.cols);
defer:
    mfclose(f);
    mfree(data);
    return ret;
}

static int64_t i64_max(int64_t a, int64_t b) {
    return a > b? a: b;
}

static int64_t i64_min(int64_t a, int64_t b) {
    return a < b? a: b;
}

void grid_do_in_rect(grid *g, int64_t x0, int64_t y0, int64_t x1, int64_t y1, void(*fun)(grid*, uint64_t, uint64_t, void*), void *user_data) {
    int64_t x_max = i64_max(x0, x1);
    int64_t x_min = i64_min(x0, x1);

    int64_t y_max = i64_max(y0, y1);
    int64_t y_min = i64_min(y0, y1);

    for (int64_t y = y_min; y < y_max; ++y)
        for (int64_t x = x_min; x < x_max; ++x) {
            int64_t xl = ((x % (int64_t)g->gi.cols) + (int64_t)g->gi.cols) % g->gi.cols;
            int64_t yl = ((y % (int64_t)g->gi.rows) + (int64_t)g->gi.rows) % g->gi.rows;
            fun(g, yl, xl, user_data);
        }
}

void grid_do_in_ellipse(grid *g, int64_t x0, int64_t y0, int64_t a, int64_t b, void(*fun)(grid*, uint64_t, uint64_t, void*), void *user_data) {
    for (int64_t y = y0 - b; y < y0 + b; ++y)
        for (int64_t x = x0 - a; x < x0 + a; ++x)
                if (((x - x0) * (x - x0) / (double)(a * a) + (y - y0) * (y - y0) / (double)(b * b)) <= 1) {
                    int64_t xl = ((x % (int64_t)g->gi.cols) + (int64_t)g->gi.cols) % g->gi.cols;
                    int64_t yl = ((y % (int64_t)g->gi.rows) + (int64_t)g->gi.rows) % g->gi.rows;
                    fun(g, yl, xl, user_data);
                }
}

static bool triangle_inside(double x, double y, double x0, double y0, double x1, double y1, double x2, double y2) {
	double alpha = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) /
				   ((y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2));
	double beta = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) /
				  ((y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2));
	double gamma = 1.0 - alpha - beta;
	return alpha >= 0 && beta >= 0 && gamma >= 0;
}


void grid_do_in_triangle(grid *g, int64_t x0, int64_t y0, int64_t x1, int64_t y1, int64_t x2, int64_t y2, void(*fun)(grid*, uint64_t, uint64_t, void*), void *user_data) {
    int64_t x_max = i64_max(x0, i64_max(x1, x2));
    int64_t x_min = i64_min(x0, i64_min(x1, x2));

    int64_t y_max = i64_max(y0, i64_max(y1, y2));
    int64_t y_min = i64_min(y0, i64_min(y1, y2));

    for (int64_t y = y_min; y < y_max; ++y)
        for (int64_t x = x_min; x < x_max; ++x) {
            int64_t xl = ((x % (int64_t)g->gi.cols) + (int64_t)g->gi.cols) % g->gi.cols;
            int64_t yl = ((y % (int64_t)g->gi.rows) + (int64_t)g->gi.rows) % g->gi.rows;
            if (triangle_inside(x, y, x0, y0, x1, y1, x2, y2))
                fun(g, yl, xl, user_data);
        }
}

void grid_do_in_line(grid *g, int64_t x0, int64_t y0, int64_t x1, int64_t y1, int64_t thickness, void(*fun)(grid*, uint64_t, uint64_t, void*), void *user_data) {
    if (x0 == x1) {
        grid_do_in_rect(g, x0 - thickness / 2, y0, x0 + thickness / 2, y1, fun, user_data);
        return;
    }

    if (y0 == y1) {
        grid_do_in_rect(g, x0, y0 - thickness / 2, x1, y0 + thickness / 2, fun, user_data);
        return;
    }

    double dy = y1 - y0;
    double dx = x1 - x0;
    double ny = dx;
    double nx = -dy;
    double M = sqrt(nx * nx + ny * ny);
    ny /= M;
    nx /= M;
    grid_do_in_triangle(g, x0 - nx * thickness / 2, y0 - ny * thickness / 2, x1 - nx * thickness / 2, y1 - ny * thickness / 2, x0 + nx * thickness / 2, y0 + ny * thickness / 2, fun, user_data);
    grid_do_in_triangle(g, x1 + nx * thickness / 2, y1 + ny * thickness / 2, x1 - nx * thickness / 2, y1 - ny * thickness / 2, x0 + nx * thickness / 2, y0 + ny * thickness / 2, fun, user_data);
}

dm_interaction dm_interfacial(double dm) {
    return (dm_interaction){.dmv_down = v3d_c(-dm, 0.0, 0.0),
                            .dmv_up = v3d_c(dm, 0.0, 0.0),
                            .dmv_left = v3d_c(0.0, dm, 0.0),
                            .dmv_right = v3d_c(0.0, -dm, 0.0)};
}

dm_interaction dm_bulk(double dm) {
    return (dm_interaction){.dmv_down = v3d_c(0.0, -dm, 0.0),
                            .dmv_up = v3d_c(0.0, dm, 0.0),
                            .dmv_left = v3d_c(dm, 0.0, 0.0),
                            .dmv_right = v3d_c(-dm, 0.0, 0.0)};
}

anisotropy anisotropy_z_axis(double value) {
    return (anisotropy){.dir = v3d_c(0, 0, 1), .ani = value};
}

#define da_append(da, item) do { \
    if ((da)->len >= (da)->cap) { \
        if ((da)->cap <= 1) \
            (da)->cap = (da)->len + 2;\
        else\
            (da)->cap *= 1.5; \
        (da)->items = realloc((da)->items, sizeof(*(da)->items) * (da)->cap); \
        if (!(da)->items) \
            logging_log(LOG_FATAL, "%s:%d Could not append item to dynamic array. Allocation failed. Buy more RAM I guess, lol", __FILE__, __LINE__); \
        memset(&(da)->items[(da)->len], 0, sizeof(*(da)->items) * ((da)->cap - (da)->len));\
    } \
    (da)->items[(da)->len] = (item); \
    (da)->len += 1; \
} while(0)

#define da_remove(da, idx) do { \
    if ((idx) >= (da)->len || (idx) < 0) { \
        logging_log(LOG_ERROR, "%s:%d Trying to remove out of range idx %ll from dynamic array", __FILE__, __LINE__, (int64_t)idx); \
        break; \
    } \
    memmove(&((da)->items[(idx)]), &((da)->items[(idx) + 1]), sizeof(*((da)->items)) * ((da)->len - (idx) - 1)); \
    if ((da)->len <= (da)->cap / 2) { \
        (da)->cap /= 1.5; \
        (da)->items = realloc((da)->items, sizeof(*((da)->items)) * (da)->cap); \
        if (!(da)->items) \
            logging_log(LOG_FATAL, "%s:%d Could not append item to dynamic array. Allocation failed. Buy more RAM I guess, lol", __FILE__, __LINE__); \
        memset(&((da)->items[(da)->len]), 0, sizeof(*((da)->items)) * ((da)->cap - (da)->len));\
    } \
    (da)->len -= 1;\
} while(0)

//https://iopscience.iop.org/article/10.1088/2633-1357/abad0c/pdf
double q_ijk(v3d mi, v3d mj, v3d mk) {
    double num = v3d_dot(mi, v3d_cross(mj, mk));
    double den = 1.0 + v3d_dot(mi, mj) + v3d_dot(mi, mk) + v3d_dot(mj, mk);
    return 2.0 * atan2(num, den);
}

double charge_lattice(v3d m, v3d left, v3d right, v3d up, v3d down) {
    double q_012 = q_ijk(m, right, up);
    double q_023 = q_ijk(m, up, left);
    double q_034 = q_ijk(m, left, down);
    double q_041 = q_ijk(m, down, right);
    return 1.0 / (8.0 * M_PI) * (q_012 + q_023 + q_034 + q_041);
}

static double metric(v3d *v, int rows, int cols, int i0, int j0, int i1, int j1) {
    UNUSED(rows);
    //double charge_0 = 0;
    //double charge_1 = 0;
    //{
    //    uint64_t x = j0;
    //    uint64_t y = i0;

    //    uint64_t right = (((x + 1) % cols) + cols) % cols;
    //    uint64_t ridx = y * cols + right;
    //    v3d mright = v[ridx];

    //    uint64_t left = ((((int64_t)x - 1) % (int64_t)cols) + cols) % cols;
    //    uint64_t lidx = y * cols + left;
    //    v3d mleft = v[lidx];

    //    uint64_t up = (((y + 1) % rows) + rows) % rows;;
    //    uint64_t uidx = up * cols + x;
    //    v3d mup = v[uidx];

    //    uint64_t down = ((((int64_t)y - 1) % (int64_t)rows) + rows) % rows;
    //    uint64_t didx = down * cols + x;
    //    v3d mdown = v[didx];

    //    v3d m = v[i0 * cols + j0];
    //    charge_0 = charge_lattice(m, mleft, mright, mup, mdown);
    //}

    //{
    //    uint64_t x = j1;
    //    uint64_t y = i1;

    //    uint64_t right = (((x + 1) % cols) + cols) % cols;
    //    uint64_t ridx = y * cols + right;
    //    v3d mright = v[ridx];

    //    uint64_t left = ((((int64_t)x - 1) % (int64_t)cols) + cols) % cols;
    //    uint64_t lidx = y * cols + left;
    //    v3d mleft = v[lidx];

    //    uint64_t up = (((y + 1) % rows) + rows) % rows;;
    //    uint64_t uidx = up * cols + x;
    //    v3d mup = v[uidx];

    //    uint64_t down = ((((int64_t)y - 1) % (int64_t)rows) + rows) % rows;
    //    uint64_t didx = down * cols + x;
    //    v3d mdown = v[didx];

    //    v3d m = v[i1 * cols + j1];
    //    charge_1 = charge_lattice(m, mleft, mright, mup, mdown);
    //}
    v3d m0 = v[i0 * cols + j0];
    v3d m1 = v[i1 * cols + j1];
    m0.z = m0.z < 0? -1: 1;
    m1.z = m1.z < 0? -1: 1;
    return fabs(m1.z - m0.z);
}

void grid_cluster(grid *g, double eps, uint64_t min_pts) {
    uint64_t rows = g->gi.rows;
    uint64_t cols = g->gi.cols;
    g->clusters.len = 0;
    double avg_mz = 0;
    for (uint64_t i = 0; i < rows; ++i) {
        for (uint64_t j = 0; j < cols; ++j) {
            g->points[i * cols + j] = (cluster_point){.cluster = 0, .row = i, .col = j, .label = UNDEFINED};
            avg_mz += g->m[i * cols + j].z / (double)(rows * cols);
        }
    }

    for (uint64_t i = 0; i < rows * cols; ++i) {
        cluster_point *it = &g->points[i];

        if (it->label != UNDEFINED)
            continue;

        g->queue.len = 0;
        memset(g->seen, 0, sizeof(*g->seen) * rows * cols);
        da_append(&g->queue, i);
        uint64_t count = 0;
        while (g->queue.len) {
            cluster_point *qt = &g->points[g->queue.items[0]];
            uint64_t x = qt->col;
            uint64_t y = qt->row;
            bool *saw = &g->seen[y * cols + x];
            if (qt->label != UNDEFINED || *saw) {
                da_remove(&g->queue, 0);
                continue;
            }
            *saw = true;

            uint64_t right = x + 1;
            right = ((right % cols) + cols) % cols;
            uint64_t ridx = y * cols + right;

            if (metric(g->m, rows, cols, y, right, y, x) < eps && !g->seen[ridx])
                da_append(&g->queue, ridx);

            uint64_t left = ((((int64_t)x - 1) % (int64_t)cols) + cols) % cols;
            uint64_t lidx = y * cols + left;

            if (metric(g->m, rows, cols, y, left, y, x) < eps && !g->seen[lidx])
                da_append(&g->queue, lidx);

            uint64_t up = y + 1;
            up = ((up % rows) + rows) % rows;
            uint64_t uidx = up * cols + x;

            if (metric(g->m, rows, cols, up, x, y, x) < eps && !g->seen[uidx])
                da_append(&g->queue, uidx);

            uint64_t down = ((((int64_t)y - 1) % (int64_t)rows) + rows) % rows;
            uint64_t didx = down * cols + x;

            if (metric(g->m, rows, cols, down, x, y, x) < eps && !g->seen[didx])
                da_append(&g->queue, didx);

            da_remove(&g->queue, 0);
            count++;
        }

        int label = UNDEFINED;
        if (count < min_pts) {
            label = NOISE;
            //logging_log(LOG_INFO, "%e %e %llu", (double)i / g->gi.cols, (double)(i % g->gi.cols), count);
        }
        else {
            da_append(&g->clusters, ((cluster_center){.x = 0, .y = 0, .id = g->clusters.len, .count = 0, .avg_m = v3d_s(0), .sum_weight = 0}));
            label = CLUSTER;
        }

        g->queue.len = 0;
        memset(g->seen, 0, sizeof(*g->seen) * rows * cols);
        da_append(&g->queue, i);
        while (g->queue.len) {
            cluster_point *qt = &g->points[g->queue.items[0]];
            uint64_t x = qt->col;
            uint64_t y = qt->row;
            bool *saw = &g->seen[y * cols + x];
            if (qt->label != UNDEFINED || *saw) {
                da_remove(&g->queue, 0);
                continue;
            }
            *saw = true;

            qt->label = label;
            if (qt->label == CLUSTER) {
                qt->cluster = g->clusters.len - 1;
                uint64_t c = qt->cluster;
                g->clusters.items[c].x += x * fabs(g->m[y * cols + x].z);
                g->clusters.items[c].y += y * fabs(g->m[y * cols + x].z);
                g->clusters.items[c].count += 1;
                g->clusters.items[c].avg_m = v3d_sum(g->clusters.items[c].avg_m, g->m[y * cols + x]);
                g->clusters.items[c].sum_weight += fabs(g->m[y * cols + x].z);
            }

            uint64_t right = x + 1;
            right = ((right % cols) + cols) % cols;
            uint64_t ridx = y * cols + right;

            if (metric(g->m, rows, cols, y, right, y, x) < eps && !g->seen[ridx])
                da_append(&g->queue, ridx);

            uint64_t left = ((((int64_t)x - 1) % (int64_t)cols) + cols) % cols;
            uint64_t lidx = y * cols + left;

            if (metric(g->m, rows, cols, y, left, y, x) < eps && !g->seen[lidx])
                da_append(&g->queue, lidx);

            uint64_t up = y + 1;
            up = ((up % rows) + rows) % rows;
            uint64_t uidx = up * cols + x;

            if (metric(g->m, rows, cols, up, x, y, x) < eps && !g->seen[uidx])
                da_append(&g->queue, uidx);

            uint64_t down = ((((int64_t)y - 1) % (int64_t)rows) + rows) % rows;
            uint64_t didx = down * cols + x;

            if (metric(g->m, rows, cols, down, x, y, x) < eps && !g->seen[didx])
                da_append(&g->queue, didx);

            da_remove(&g->queue, 0);
        }
    }

    for (uint64_t i = 0; i < g->clusters.len; ++i) {
        cluster_center *it = &g->clusters.items[i];
        if (it->count > 0) {
            it->avg_m = v3d_scalar(it->avg_m, 1.0 / (double)it->count);
            it->x /= (double)it->sum_weight;
            it->y /= (double)it->sum_weight;
        }
    }

    //for (uint64_t i = 0; i < ret.len; ++i) {
    //    center *it = &ret.items[i];
    //    if (CLOSE_ENOUGH(it->avg_m.z, avg_mz, 0.1))
    //        da_remove(&ret, i);
    //}
}

void grid_cluster_kmeans(grid *g, uint64_t n_clusters, uint64_t niter) {
    uint64_t rows = g->gi.rows;
    uint64_t cols = g->gi.cols;
    v3d avg_m = {0};
    static bool first = true;
    for (uint64_t i = 0; i < rows; ++i) {
        for (uint64_t j = 0; j < cols; ++j) {
            g->points[i * cols + j] = (cluster_point){.cluster = 0, .row = i, .col = j, .label = UNDEFINED};
            avg_m = v3d_sum(avg_m, g->m[i * cols + j]);
        }
    }
    avg_m = v3d_scalar(avg_m, 1.0 / (double)(rows * cols));

    if (first) {
        for (uint64_t i = 0; i < n_clusters; ++i)
            da_append(&g->clusters, ((cluster_center){.id = i, .count = 0, .x = shit_random(0, cols), .y = shit_random(0, rows)}));
    }
    first = false;

    for (uint64_t iter = 0; iter < niter; ++iter) {
        for (uint64_t i = 0; i < g->clusters.len; ++i) {
            g->clusters.items[i].count = 0;
            g->clusters.items[i].sum_weight = 0;
        }

        for (uint64_t i = 0; i < rows * cols; ++i) {
            uint64_t y = i / cols;
            uint64_t x = i % cols;

            double min_dist = FLT_MAX;
            uint64_t min_dist_idx = 0;

            for (uint64_t j = 0; j < g->clusters.len; ++j) {
                double dx = x - g->clusters.items[j].x;
                double dy = y - g->clusters.items[j].y;

                dx /= (double)g->gi.cols;
                dy /= (double)g->gi.rows;
                double dist_real2 = dx * dx + dy + dy;

                v3d dm = v3d_sub(g->clusters.items[j].avg_m, g->m[i]);
                double dist_m2 = v3d_dot(dm, dm);

                double dist = sqrt(dist_real2 + dist_m2);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_dist_idx = j;
                }
            }
            g->points[i].cluster = min_dist_idx;
        }

        for (uint64_t i = 0; i < rows * cols; ++i) {
            uint64_t y = i / cols;
            uint64_t x = i % cols;
            uint64_t idx = g->points[i].cluster;
            v3d diff = v3d_sub(g->m[i], avg_m);
            double weight = 1;
            if (g->m[i].z > -0.5)
                continue;
            g->clusters.items[idx].count += 1;
            g->clusters.items[idx].x += x * weight;
            g->clusters.items[idx].y += y * weight;
            g->clusters.items[idx].sum_weight += weight;
            g->clusters.items[idx].avg_m = v3d_sum(g->clusters.items[idx].avg_m, v3d_scalar(g->m[i], weight));
        }

        for (uint64_t i = 0; i < g->clusters.len; ++i) {
            if (g->clusters.items[i].count > 0) {
                g->clusters.items[i].x /= g->clusters.items[i].sum_weight;
                g->clusters.items[i].y /= g->clusters.items[i].sum_weight;
                g->clusters.items[i].avg_m = (v3d_scalar(g->clusters.items[i].avg_m, 1.0 / (g->clusters.items[i].sum_weight)));
            }
            else {
                g->clusters.items[i] = (cluster_center){.avg_m = v3d_normalize(v3d_c(shit_random(-1, 1), shit_random(-1, 1), shit_random(-1, 1))), .id = i, .x = shit_random(0, cols), .y = shit_random(0, rows), .count = 0};
            }
        }

    }
}
