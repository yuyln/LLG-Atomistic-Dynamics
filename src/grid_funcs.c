#include <errno.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <float.h>
#include <inttypes.h>

#include "grid_funcs.h"
#include "constants.h"
#include "logging.h"
#include "allocator.h"
#include "profiler.h"
#include "utils.h"

#define CHECK_BOUNDS(rows, cols, depth, i, j, k) do { if (i >= (int)rows || j >= (int)cols || k >= (int)depth || \
        i < 0 || j < 0 || k < 0) { \
    logging_log(LOG_WARNING, "Location (%d %d %d) out of bounds (%u %u %u)", i, j, k, rows, cols, depth);\
    i = ((i % (int)rows) + (int)rows) % (int)rows; \
    j = ((j % (int)cols) + (int)cols) % (int)cols; \
    k = ((k % (int)depth) + (int)depth) % (int)depth; \
}} while(0)

double shit_random(double from, double to) {
    double r = (double)rand() / (double)RAND_MAX;
    return from + r * (to - from);
}

static void grid_allocate(grid *g) {
    g->gp = mmalloc(sizeof(*g->gp) * g->gi.rows * g->gi.cols * g->gi.depth);
    g->m = mmalloc(sizeof(*g->m) * g->gi.rows * g->gi.cols * g->gi.depth);
    g->clusters = (cluster_centers){0};
    g->points = mmalloc(sizeof(*g->points) * g->gi.rows * g->gi.cols * g->gi.depth);
    g->seen = mmalloc(sizeof(*g->seen) * g->gi.rows * g->gi.cols * g->gi.depth);
    g->on_gpu = false;

    g->queue.cap = g->gi.rows * g->gi.cols * g->gi.depth;
    g->queue.len = 0;
    g->queue.items = mmalloc(sizeof(*g->queue.items) * g->queue.cap);
    g->queue.start = 0;
    g->on_gpu = false;
    g->dimensions = g->gi.rows * g->gi.cols * g->gi.depth;
}

grid grid_init(unsigned int rows, unsigned int cols, unsigned int depth) {
    grid ret = {0};
    ret.gi.rows = rows;
    ret.gi.cols = cols;
    ret.gi.depth = depth;
    ret.gi.pbc = (pbc_rules){.pbc_x = true, .pbc_y = true, .pbc_z = true, .m = {0}};
    ret.gi.lattice = 0.5e-9;
    grid_allocate(&ret);

    double dm = 0.2 * QE * 1.0e-3;

    grid_site_params default_grid = (grid_site_params){
            .exchange = isotropic_exchange(1.0e-3 * QE),
            .dm = dm_bulk(dm),
            .cubic_ani = 0.0,
            .mu = 1.856952954255053e-23,
            .alpha = 0.3,
            .gamma = 1.760859644000000e+11,
            .ani = anisotropy_z_axis(0.05 * 1.0e-3 * QE),
            .pin = {{0}},
    };

    for (unsigned int i = 0; i < rows; ++i) { 
        for (unsigned int j = 0; j < cols; ++j) {
            for (unsigned int k = 0; k < depth; ++k) {
                ret.gp[k * rows * cols + i * cols + j] = default_grid;
                ret.gp[k * rows * cols + i * cols + j].i = i;
                ret.gp[k * rows * cols + i * cols + j].j = j;
                ret.gp[k * rows * cols + i * cols + j].k = k;
                ret.m[k * rows * cols + i * cols + j] = v3d_normalize(v3d_c(shit_random(-1, 1), shit_random(-1, 1), shit_random(-1, 1)));
            }
        }
    }
    return ret;
}

void grid_set_exchange_loc(grid *g, int i, int j, int k, exchange_interaction exchange) {
    CHECK_BOUNDS(g->gi.rows, g->gi.cols, g->gi.depth, i, j, k);
    V_AT(g->gp, i, j, k, g->gi.rows, g->gi.cols).exchange = exchange;
}

void grid_set_dm_loc(grid *g, int i, int j, int k, dm_interaction dm) {
    CHECK_BOUNDS(g->gi.rows, g->gi.cols, g->gi.depth, i, j, k);
    V_AT(g->gp, i, j, k, g->gi.rows, g->gi.cols).dm = dm;
}

void grid_set_cubic_anisotropy_loc(grid *g, int i, int j, int k, double cubic_ani) {
    CHECK_BOUNDS(g->gi.rows, g->gi.cols, g->gi.depth, i, j, k);
    V_AT(g->gp, i, j, k, g->gi.rows, g->gi.cols).cubic_ani = cubic_ani;
}

void grid_set_mu_loc(grid *g, int i, int j, int k, double mu) {
    CHECK_BOUNDS(g->gi.rows, g->gi.cols, g->gi.depth, i, j, k);
    V_AT(g->gp, i, j, k, g->gi.rows, g->gi.cols).mu = mu;
}

void grid_set_alpha_loc(grid *g, int i, int j, int k, double alpha) {
    CHECK_BOUNDS(g->gi.rows, g->gi.cols, g->gi.depth, i, j, k);
    V_AT(g->gp, i, j, k, g->gi.rows, g->gi.cols).alpha = alpha;
}

void grid_set_gamma_loc(grid *g, int i, int j, int k, double gamma) {
    CHECK_BOUNDS(g->gi.rows, g->gi.cols, g->gi.depth, i, j, k);
    V_AT(g->gp, i, j, k, g->gi.rows, g->gi.cols).gamma = gamma;
}

void grid_set_anisotropy_loc(grid *g, int i, int j, int k, anisotropy ani) {
    CHECK_BOUNDS(g->gi.rows, g->gi.cols, g->gi.depth, i, j, k);
    ani.dir = v3d_normalize(ani.dir);
    V_AT(g->gp, i, j, k, g->gi.rows, g->gi.cols).ani = ani;
}

void grid_set_pinning_loc(grid *g, int i, int j, int k, pinning pin) {
    CHECK_BOUNDS(g->gi.rows, g->gi.cols, g->gi.depth, i, j, k);
    pin.dir = v3d_normalize(pin.dir);
    V_AT(g->gp, i, j, k, g->gi.rows, g->gi.cols).pin = pin;
}

void v3d_set_at_loc(v3d *g, unsigned int rows, unsigned int cols, unsigned int depth, int i, int j, int k, v3d m) {
    CHECK_BOUNDS(rows, cols, depth, i, j, k);
    V_AT(g, i, j, k, rows, cols) = v3d_normalize(m);
}

void grid_set_exchange(grid *g, exchange_interaction exchange) {
    for (unsigned int r = 0; r < g->gi.rows; ++r)
        for (unsigned int c = 0; c < g->gi.cols; ++c)
            for (unsigned int k = 0; k < g->gi.depth; ++k)
                grid_set_exchange_loc(g, r, c, k, exchange);
}

void grid_set_dm(grid *g, dm_interaction dm) {
    for (unsigned int r = 0; r < g->gi.rows; ++r)
        for (unsigned int c = 0; c < g->gi.cols; ++c)
            for (unsigned int k = 0; k < g->gi.depth; ++k)
                grid_set_dm_loc(g, r, c, k, dm);
}

void grid_set_lattice(grid *g, double lattice) {
    g->gi.lattice = lattice;
}

void grid_set_cubic_anisotropy(grid *g, double cubic_ani) {
    for (unsigned int r = 0; r < g->gi.rows; ++r)
        for (unsigned int c = 0; c < g->gi.cols; ++c)
            for (unsigned int k = 0; k < g->gi.depth; ++k)
                grid_set_cubic_anisotropy_loc(g, r, c, k, cubic_ani);
}

void grid_set_mu(grid *g, double mu) {
    for (unsigned int r = 0; r < g->gi.rows; ++r)
        for (unsigned int c = 0; c < g->gi.cols; ++c)
            for (unsigned int k = 0; k < g->gi.depth; ++k)
                grid_set_mu_loc(g, r, c, k, mu);
}

void grid_set_alpha(grid *g, double alpha) {
    for (unsigned int r = 0; r < g->gi.rows; ++r)
        for (unsigned int c = 0; c < g->gi.cols; ++c)
            for (unsigned int k = 0; k < g->gi.depth; ++k)
                grid_set_alpha_loc(g, r, c, k, alpha);

}

void grid_set_gamma(grid *g, double gamma) {
    for (unsigned int r = 0; r < g->gi.rows; ++r)
        for (unsigned int c = 0; c < g->gi.cols; ++c)
            for (unsigned int k = 0; k < g->gi.depth; ++k)
                grid_set_gamma_loc(g, r, c, k, gamma);
}

void grid_set_anisotropy(grid *g, anisotropy ani) {
    for (unsigned int r = 0; r < g->gi.rows; ++r)
        for (unsigned int c = 0; c < g->gi.cols; ++c)
            for (unsigned int k = 0; k < g->gi.depth; ++k)
                grid_set_anisotropy_loc(g, r, c, k, ani);
}

void v3d_fill_with_random(v3d *v, unsigned int rows, unsigned int cols, unsigned int depth) {
    for (unsigned int r = 0; r < rows; ++r)
        for (unsigned int c = 0; c < cols; ++c)
            for (unsigned int k = 0; k < depth; ++k)
                v3d_set_at_loc(v, rows, cols, depth, r, c, k, v3d_normalize(v3d_c(shit_random(-1.0, 1.0), shit_random(-1.0, 1.0), shit_random(-1.0, 1.0))));
}

void v3d_create_skyrmion_at(v3d *v, unsigned int rows, unsigned int cols, unsigned int depth, double radius, double dw_width, double ix, double iy, double Q, double vor, double _gamma) {
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
            for (uint64_t k = 0; k < depth; ++k)
                V_AT(v, y, x, k, rows, cols) = v3d_c(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
        }
    }
}

void v3d_create_biskyrmion_at(v3d *v, unsigned int rows, unsigned int cols, unsigned int depth, double radius, double dw, double ix, double iy, double dr, double angle, double Q, double vor, double g) {
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
            for (uint64_t k = 0; k < depth; ++k)
                V_AT(v, y, x, k, rows, cols) = v3d_c(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
        }
    }
}

void v3d_create_skyrmionium_at(v3d *v, unsigned int rows, unsigned int cols, unsigned int depth, double radius, double dw_width, double ix, double iy, double Q, double vor, double _gamma) {
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
            double theta = 4.0 * atan(pow(sinh(radius / dw_width) / sinh(r / dw_width), -Q));
            for (uint64_t k = 0; k < depth; ++k)
                V_AT(v, y, x, k, rows, cols) = v3d_c(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
        }
    }
}

void v3d_create_hopfion_at(v3d *v, unsigned int rows, unsigned int cols, unsigned int depth, double radius, double height, double c, double ix, double iy, double iz) {
    int dh = height;
    int dr = 4.0 * dh;
    for (int i = -dr; i <= dr; ++i) {
        for (int j = -dr; j <= dr; ++j) {
            for (int k = -dh; k <= dh; ++k) {
                double r = sqrt(i * i + j * j + k * k) + EPS;
                if (r > dr)
                    continue;
                int x = ix + j;
                int y = iy + i;
                int z = iz + k;
                x = ((x % (int)cols) + (int)cols) % (int)cols;
                y = ((y % (int)rows) + (int)rows) % (int)rows;
                z = ((z % (int)depth) + (int)depth) % (int)depth;
                v3d m = {0};
#if 0
                double theta = atan2(i, j) + M_PI;
                double phi = acos(k / r);
                double xi = c * M_PI / sqrt((radius * radius / (r * r)) + c * c);
                double t = acos(-2.0 * sin(theta) * sin(theta) * sin(xi) * sin(xi) + 1.0);
                double f = phi - atan2(1.0, cos(theta) * tan(xi));
                m.x = sin(t) * cos(f);
                m.y = sin(t) * sin(f);
                m.z = cos(t);
#else
                double f = exp(-r * r / (4.0 * height * height)) * M_PI;
                m.x = j / r * sin(2 * f) + 2.0 * i * k / (r * r) * sin(f) * sin(f);
                m.y = i / r * sin(2 * f) - 2.0 * j * k / (r * r) * sin(f) * sin(f);
                m.z = cos(2.0 * f) + 2.0 * k * k / (r * r) * sin(f) * sin(f);
#endif
                V_AT(v, y, x, z, rows, cols) = v3d_normalize(m);
            }
        }
    }
}

void v3d_uniform(v3d *v, unsigned int rows, unsigned int cols, unsigned int depth, v3d dir) {
    for (unsigned int i = 0; i < rows * cols * depth; ++i)
        v[i] = v3d_normalize(dir);
}

void grid_fill_with_random(grid *g) {
    v3d_fill_with_random(g->m, g->gi.rows, g->gi.cols, g->gi.depth);
}

void grid_create_skyrmion_at(grid *g, double radius, double dw_width, double ix, double iy, double Q, double vor, double _gamma) {
    v3d_create_skyrmion_at(g->m, g->gi.rows, g->gi.cols, g->gi.depth, radius, dw_width, ix, iy, Q, vor, _gamma);
}

void grid_uniform(grid *g, v3d dir) {
    v3d_uniform(g->m, g->gi.rows, g->gi.cols, g->gi.depth, dir);
}

void grid_create_biskyrmion_at(grid *g, double radius, double dw_width, double ix, double iy, double dr, double angle, double Q, double vorticity, double _gamma) {
    v3d_create_biskyrmion_at(g->m, g->gi.rows, g->gi.cols, g->gi.depth, radius, dw_width, ix, iy, dr, angle, Q, vorticity, _gamma);
}

void grid_create_skyrmionium_at(grid *g, double radius, double dw_width, double ix, double iy, double Q, double vor, double _gamma) {
    v3d_create_skyrmionium_at(g->m, g->gi.rows, g->gi.cols, g->gi.depth, radius, dw_width, ix, iy, Q, vor, _gamma);
}

void grid_create_hopfion_at(grid *g, double radius, double height, double c, double ix, double iy, double iz) {
    v3d_create_hopfion_at(g->m, g->gi.rows, g->gi.cols, g->gi.depth, radius, height, c, ix, iy, iz);
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
    uint64_t gp_size_bytes = g->dimensions * sizeof(*g->gp);
    uint64_t m_size_bytes = g->dimensions * sizeof(*g->m);

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

    uint64_t gp_size_bytes = g->dimensions * sizeof(*g->gp);
    uint64_t m_size_bytes = g->dimensions * sizeof(*g->m);

    gpu_cl_read_gpu(&gpu, gp_size_bytes, 0, g->gp, g->gp_gpu);
    gpu_cl_read_gpu(&gpu, m_size_bytes, 0, g->m, g->m_gpu);
}

void v3d_from_gpu(v3d *g, cl_mem buffer, unsigned int rows, unsigned int cols, unsigned int depth, gpu_cl gpu) {
    uint64_t m_size_bytes = rows * cols * depth * sizeof(*g);
    gpu_cl_read_gpu(&gpu, m_size_bytes, 0, g, buffer);
}

bool v3d_dump(FILE *f, v3d *v, unsigned int rows, unsigned int cols, unsigned int depth) {
    return rows * cols * depth * sizeof(*v) == fwrite(v, rows * cols * depth * sizeof(*v), 1, f);
}

bool grid_dump(FILE *f, grid *g) {
    bool ret = sizeof(g->gi) == fwrite(&g->gi, 1, sizeof(g->gi),  f);
    ret = ret && (sizeof(*g->gp) * g->dimensions) == fwrite(g->gp, 1, sizeof(*g->gp) * g->dimensions, f);
    ret = ret && (sizeof(*g->m) * g->dimensions) == fwrite(g->m, 1, sizeof(*g->m) * g->dimensions, f);
    return ret;
}

bool grid_from_file(const char *path, grid *g) {
    if (!g)
        logging_log(LOG_FATAL, "NULL pointer to grid provided");

    if (g->m || g->gp || g->on_gpu || g->gi.cols || g->gi.rows || g->gi.depth)
        logging_log(LOG_FATAL, "Trying to initialize grid from file with grid already initialized");

    FILE *f = mfopen(path, "rb");
    char *data = NULL;
    bool ret = true;

    if (fseek(f, 0, SEEK_END) < 0) {
        logging_log(LOG_ERROR, "Moving cursor to the end of %s failed: %s", path, strerror(errno));
        ret = false;
        goto defer;
    }

    long sz;
    if ((sz = ftell(f)) < 0) {
        logging_log(LOG_ERROR, "Getting cursor position of % failed: %s", path, strerror(errno));
        ret = false;
        goto defer;
    }

    if (fseek(f, 0, SEEK_SET) < 0) {
        logging_log(LOG_ERROR, "Moving cursor to start of %s failed: %s", path, strerror(errno));
        ret = false;
        goto defer;
    }

    data = mmalloc(sz + 1);

    char *ptr = data;
    if (fread(data, 1, sz, f) != (uint64_t)sz) {
        logging_log(LOG_ERROR, "Reading data from \"%s\" failed: %s", path, strerror(errno));
        ret = false;
        goto defer;
    }

    g->gi = *((grid_info*)data);
    g->dimensions = g->gi.rows * g->gi.cols * g->gi.depth;
    ptr += sizeof(grid_info);

    grid_allocate(g);

    memcpy(g->gp, ptr, sizeof(*g->gp) * g->dimensions);
    ptr += sizeof(*g->gp) * g->dimensions;

    memcpy(g->m, ptr, sizeof(*g->m) * g->dimensions);
defer:
    mfclose(f);
    mfree(data);
    return ret;
}

bool grid_from_animation_bin(const char *path, grid *g, int64_t frame) {
    if (!g)
        logging_log(LOG_FATAL, "NULL pointer to grid provided");

    if (g->m || g->gp || g->on_gpu || g->gi.cols || g->gi.rows || g->gi.depth)
        logging_log(LOG_FATAL, "Trying to initialize grid from file with grid already initialized");

    FILE *f = mfopen(path, "rb");
    bool ret = true;

    uint64_t frames = 0;
    if (fread(&frames, 1, sizeof(frames), f) != sizeof(frames)) {
        logging_log(LOG_ERROR, "Reading data from \"%s\" failed: %s", path, strerror(errno));
        ret = false;
        goto defer;
    }

    if (fread(&g->gi, 1, sizeof(g->gi), f) != sizeof(g->gi)) {
        logging_log(LOG_ERROR, "Reading data from \"%s\" failed: %s", path, strerror(errno));
        ret = false;
        goto defer;
    }
    g->dimensions = g->gi.rows * g->gi.cols * g->gi.depth;

    long where_to_come_back = ftell(f);
    if (where_to_come_back < 0) {
        logging_log(LOG_ERROR, "Getting current position from \"%s\" failed: %s", path, strerror(errno));
        ret = false;
        goto defer;
    }

    if (fseek(f, 0, SEEK_END) < 0) {
        logging_log(LOG_ERROR, "Seeking to end of \"%s\" failed: %s", path, strerror(errno));
        ret = false;
        goto defer;
    }

    long end_of_file = ftell(f);
    if (end_of_file < 0) {
        logging_log(LOG_ERROR, "Getting current position from \"%s\" failed: %s", path, strerror(errno));
        ret = false;
        goto defer;
    }

    uint64_t frames_size = end_of_file - sizeof(frames) - sizeof(g->gi) - sizeof(*g->gp) * g->dimensions;
    if (frames_size % (sizeof(*g->m) * g->dimensions)) {
        logging_log(LOG_ERROR, "Size of file isn't multiple of size of grid, probably corrupted");
        ret = false;
        goto defer;
    }

    frames = frames_size / (sizeof(*g->m) * g->dimensions);
    frame = ((frame % (int64_t)frames) + frames) % frames;

    if (fseek(f, where_to_come_back, SEEK_SET) < 0) {
        logging_log(LOG_ERROR, "Seeking to beginning of \"%s\" failed: %s", path, strerror(errno));
        ret = false;
        goto defer;
    }

    grid_allocate(g);

    if (fread(g->gp, 1, sizeof(*g->gp) * g->dimensions, f) != (sizeof(*g->gp) * g->dimensions)) {
        logging_log(LOG_ERROR, "Reading data from \"%s\" failed: %s", path, strerror(errno));
        ret = false;
        goto defer;
    }

    if (fseek(f, frame * g->dimensions * sizeof(*g->m), SEEK_CUR) < 0) {
        logging_log(LOG_ERROR, "Advancing to %"PRIi64" from \"%s\" failed: %s", frame, path, strerror(errno));
        ret = false;
        goto defer;
    }

    if (fread(g->m, 1, sizeof(*g->m) * g->dimensions, f) != (sizeof(*g->m) * g->dimensions)) {
        logging_log(LOG_ERROR, "Reading data from \"%s\" failed: %s", path, strerror(errno));
        ret = false;
        goto defer;
    }
defer:
    mfclose(f);
    return ret;
}

static int64_t i64_max(int64_t a, int64_t b) {
    return a > b? a: b;
}

static int64_t i64_min(int64_t a, int64_t b) {
    return a < b? a: b;
}

exchange_interaction isotropic_exchange(double value) {
    return (exchange_interaction){.J_up = value,
                                  .J_down = value,
                                  .J_left = value,
                                  .J_right = value,
                                  .J_front = value,
                                  .J_back = value};
}

dm_interaction dm_interfacial(double dm) {
    return (dm_interaction){.dmv_down = v3d_c(-dm, 0.0, 0.0),
                            .dmv_up = v3d_c(dm, 0.0, 0.0),
                            .dmv_left = v3d_c(0.0, dm, 0.0),
                            .dmv_right = v3d_c(0.0, -dm, 0.0),
                            .dmv_front = v3d_s(0), 
                            .dmv_back = v3d_s(0)};
}

dm_interaction dm_bulk(double dm) {
    return (dm_interaction){.dmv_down = v3d_c(0.0, -dm, 0.0),
                            .dmv_up = v3d_c(0.0, dm, 0.0),
                            .dmv_left = v3d_c(-dm, 0.0, 0.0),
                            .dmv_right = v3d_c(dm, 0.0, 0.0),
                            .dmv_front = v3d_c(0, 0, dm),
                            .dmv_back = v3d_c(0, 0, -dm)};
}

anisotropy anisotropy_z_axis(double value) {
    return (anisotropy){.dir = v3d_c(0, 0, 1), .ani = value};
}

//https://iopscience.iop.org/article/10.1088/2633-1357/abad0c/pdf
static double q_ijk(v3d mi, v3d mj, v3d mk) {
    double num = v3d_dot(mi, v3d_cross(mj, mk));
    double den = 1.0 + v3d_dot(mi, mj) + v3d_dot(mi, mk) + v3d_dot(mj, mk);
    return 2.0 * atan2(num, den);
}

static double charge_lattice(v3d m, v3d left, v3d right, v3d up, v3d down) {
    double q_012 = q_ijk(m, right, up);
    double q_023 = q_ijk(m, up, left);
    double q_034 = q_ijk(m, left, down);
    double q_041 = q_ijk(m, down, right);
    return 1.0 / (8.0 * M_PI) * (q_012 + q_023 + q_034 + q_041);
}

static double default_metric(grid *g, uint64_t i0, uint64_t j0, uint64_t k0, uint64_t i1, uint64_t j1, uint64_t k1, void *user_data) {
    UNUSED(user_data);
    v3d m0 = V_AT(g->m, i0, j0, k0, g->gi.rows, g->gi.cols);
    v3d m1 = V_AT(g->m, i1, j1, k1, g->gi.rows, g->gi.cols);
    m0.z = m0.z < 0.0? -1: 1;
    m1.z = m1.z < 0.0? -1: 1;
    return fabs(m1.z - m0.z);
}

static double default_weight(grid *g, uint64_t i, uint64_t j, uint64_t k, void *user_data) {
    UNUSED(user_data);
    return V_AT(g->m, i, j, k, g->gi.rows, g->gi.cols).z;
}

INCEPTION("DA -> [ CLUSTER ] -> 5.700138278e-03 sec")
INCEPTION("RB -> [ CLUSTER ] -> 3.818874674e-03 sec")
INCEPTION("RB with custom metric etc -> [ CLUSTER ] -> 4.142878783e-03 sec")
void grid_cluster(grid *g, double eps, uint64_t min_pts, double(*metric)(grid*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, void*), double(*weight_f)(grid*, uint64_t, uint64_t, uint64_t, void*), void *user_data_metric, void *user_data_weight) {
    uint64_t rows = g->gi.rows;
    uint64_t cols = g->gi.cols;
    uint64_t depth = g->gi.depth;

    if (!metric)
        metric = default_metric;

    if (!weight_f)
        weight_f = default_weight;

    g->clusters.len = 0;

    v3d avg_m = {0};
    for (uint64_t i = 0; i < rows; ++i) {
        for (uint64_t j = 0; j < cols; ++j) {
            for (uint64_t k = 0; k < depth; ++k) {
                V_AT(g->points, i, j, k, rows, cols) = (cluster_point){.i = i, .j = j, .k = k, .label = UNDEFINED};
                avg_m = v3d_sum(avg_m, V_AT(g->m, i, j, k, g->gi.rows, g->gi.cols));
            }
        }
    }
    avg_m = v3d_scalar(avg_m, 1.0 / (rows * cols * depth));

    for (uint64_t i = 0; i < rows * cols * depth; ++i) {
        cluster_point *it = &g->points[i];

        if (it->label != UNDEFINED)
            continue;

        g->queue.len = 0;
        g->queue.start = 0;
        memset(g->seen, 0, sizeof(*g->seen) * rows * cols);

        rb_append(&g->queue, i);

        uint64_t count = 0;
        while (g->queue.len) {
            cluster_point *qt = &g->points[rb_at(&g->queue, 0)];
            uint64_t y = qt->i;
            uint64_t x = qt->j;
            uint64_t z = qt->k;
            bool *saw = &V_AT(g->seen, y, x, z, rows, cols);
            if (qt->label != UNDEFINED || *saw) {
                g->queue.start += 1;
                g->queue.len -= 1;
                continue;
            }
            *saw = true;

            uint64_t right = x + 1;
            right = ((right % cols) + cols) % cols;
            uint64_t ridx = z * rows * cols + y * cols + right;
            if (x < cols - 1 || g->gi.pbc.pbc_x)
                if (metric(g, y, right, z, y, x, z, user_data_metric) < eps && !g->seen[ridx])
                    rb_append(&g->queue, ridx);


            uint64_t left = ((((int64_t)x - 1) % (int64_t)cols) + cols) % cols;
            uint64_t lidx = z * rows * cols + y * cols + left;

            if (x > 0 || g->gi.pbc.pbc_x)
                if (metric(g, y, left, z, y, x, z, user_data_metric) < eps && !g->seen[lidx])
                    rb_append(&g->queue, lidx);

            uint64_t up = y + 1;
            up = ((up % rows) + rows) % rows;
            uint64_t uidx = z * rows * cols + up * cols + x;

            if (y < rows - 1 || g->gi.pbc.pbc_y)
                if (metric(g, up, x, z, y, x, z, user_data_metric) < eps && !g->seen[uidx])
                    rb_append(&g->queue, uidx);

            uint64_t down = ((((int64_t)y - 1) % (int64_t)rows) + rows) % rows;
            uint64_t didx = z * rows * cols + down * cols + x;

            if (y > 0 || g->gi.pbc.pbc_y)
                if (metric(g, down, x, z, y, x, z, user_data_metric) < eps && !g->seen[didx])
                    rb_append(&g->queue, didx);

            uint64_t front = z + 1;
            front = ((front % depth) + depth) % depth;
            uint64_t fidx = front * rows * cols + y * cols + x;

            if (z < depth - 1 || g->gi.pbc.pbc_z)
                if (metric(g, y, x, front, y, x, z, user_data_metric) < eps && !g->seen[fidx])
                    rb_append(&g->queue, fidx);

            uint64_t back = ((((int64_t)z - 1) % (int64_t)depth) + depth) % depth;
            uint64_t bidx = back * rows * cols + y * cols + x;

            if (z > 0 || g->gi.pbc.pbc_z)
                if (metric(g, y, x, back, y, x, z, user_data_metric) < eps && !g->seen[bidx])
                    rb_append(&g->queue, bidx);

            g->queue.start += 1;
            g->queue.len -= 1;
            count++;
        }
        int label = UNDEFINED;
        if (count < min_pts)
            label = NOISE;
        else {
            uint64_t c = g->clusters.len;
            da_append(&g->clusters, ((cluster_center){.id = c, .x = 0, .y = 0, .z = 0}));
            label = CLUSTER;
        }

        g->queue.len = 0;
        g->queue.start = 0;
        memset(g->seen, 0, sizeof(*g->seen) * rows * cols * depth);
        rb_append(&g->queue, i);
        it->x = it->j * g->gi.lattice;
        it->y = it->i * g->gi.lattice;
        it->z = it->k * g->gi.lattice;

        while (g->queue.len) {
            cluster_point *qt = &g->points[rb_at(&g->queue, 0)];
            uint64_t x = qt->j;
            uint64_t y = qt->i;
            uint64_t z = qt->k;
            bool *saw = &V_AT(g->seen, y, x, z, rows, cols);
            if (qt->label != UNDEFINED || *saw) {
                g->queue.start += 1;
                g->queue.len -= 1;
                continue;
            }
            *saw = true;

            qt->label = label;
            if (qt->label == CLUSTER) {
                qt->cluster = g->clusters.len - 1;
                uint64_t c = qt->cluster;
                double weight = weight_f(g, y, x, z, user_data_weight);
                g->clusters.items[c].x += qt->x * weight;
                g->clusters.items[c].y += qt->y * weight;
                g->clusters.items[c].z += qt->z * weight;

                g->clusters.items[c].j += x * weight;
                g->clusters.items[c].i += y * weight;
                g->clusters.items[c].k += z * weight;

                g->clusters.items[c].count += 1;
                g->clusters.items[c].avg_m = v3d_sum(g->clusters.items[c].avg_m, v3d_scalar(V_AT(g->m, y, x, z, rows, cols), weight));
                g->clusters.items[c].sum_weight += weight;
            }

            uint64_t right = x + 1;
            right = ((right % cols) + cols) % cols;
            uint64_t ridx = z * rows * cols + y * cols + right;
            if (x < cols - 1 || g->gi.pbc.pbc_x) {
                if (metric(g, y, right, z, y, x, z, user_data_metric) < eps && !g->seen[ridx]) {
                    rb_append(&g->queue, ridx);
                    g->points[ridx].x = qt->x + g->gi.lattice;
                    g->points[ridx].y = qt->y;
                    g->points[ridx].z = qt->z;
                }
            }

            uint64_t left = ((((int64_t)x - 1) % (int64_t)cols) + cols) % cols;
            uint64_t lidx = z * rows * cols + y * cols + left;

            if (x > 0 || g->gi.pbc.pbc_x) {
                if (metric(g, y, left, z, y, x, z, user_data_metric) < eps && !g->seen[lidx]) {
                    rb_append(&g->queue, lidx);
                    g->points[lidx].x = qt->x - g->gi.lattice;
                    g->points[lidx].y = qt->y;
                    g->points[ridx].z = qt->z;
                }
            }

            uint64_t up = y + 1;
            up = ((up % rows) + rows) % rows;
            uint64_t uidx = z * rows * cols + up * cols + x;

            if (y < rows - 1 || g->gi.pbc.pbc_y) {
                if (metric(g, up, x, z, y, x, z, user_data_metric) < eps && !g->seen[uidx]) {
                    rb_append(&g->queue, uidx);
                    g->points[uidx].y = qt->y + g->gi.lattice;
                    g->points[uidx].x = qt->x;
                    g->points[ridx].z = qt->z;
                }
            }

            uint64_t down = ((((int64_t)y - 1) % (int64_t)rows) + rows) % rows;
            uint64_t didx = z * rows * cols + down * cols + x;

            if (y > 0 || g->gi.pbc.pbc_y) {
                if (metric(g, down, x, z, y, x, z, user_data_metric) < eps && !g->seen[didx]) {
                    rb_append(&g->queue, didx);
                    g->points[didx].y = qt->y - g->gi.lattice;
                    g->points[didx].x = qt->x;
                    g->points[ridx].z = qt->z;
                }
            }

            uint64_t front = z + 1;
            front = ((front % depth) + depth) % depth;
            uint64_t fidx = front * rows * cols + y * cols + x;

            if (z < depth - 1 || g->gi.pbc.pbc_z) {
                if (metric(g, y, x, front, y, x, z, user_data_metric) < eps && !g->seen[fidx]) {
                    rb_append(&g->queue, fidx);
                    g->points[fidx].z = qt->z + g->gi.lattice;
                    g->points[fidx].x = qt->x;
                    g->points[fidx].y = qt->y;
                }
            }

            uint64_t back = ((((int64_t)z - 1) % (int64_t)depth) + depth) % depth;
            uint64_t bidx = back * rows * cols + y * cols + x;

            if (z > 0 || g->gi.pbc.pbc_z) {
                if (metric(g, y, x, back, y, x, z, user_data_metric) < eps && !g->seen[bidx]) {
                    rb_append(&g->queue, bidx);
                    g->points[bidx].z = qt->z - g->gi.lattice;
                    g->points[bidx].x = qt->x;
                    g->points[bidx].y = qt->y;
                }
            }

            g->queue.start += 1;
            g->queue.len -= 1;
        }
    }

    for (uint64_t i = 0; i < g->clusters.len; ++i) {
        cluster_center *it = &g->clusters.items[i];
        if (!CLOSE_ENOUGH(it->sum_weight, 0, EPS)) {
            it->avg_m = v3d_normalize(v3d_scalar(it->avg_m, 1.0 / it->sum_weight));

            it->x /= it->sum_weight;
            it->y /= it->sum_weight;
            it->z /= it->sum_weight;

            it->i /= it->sum_weight;
            it->j /= it->sum_weight;
            it->k /= it->sum_weight;

            it->x = it->x - floor(it->x / (cols * g->gi.lattice)) * cols * g->gi.lattice;
            it->y = it->y - floor(it->y / (rows * g->gi.lattice)) * rows * g->gi.lattice;
            it->z = it->z - floor(it->z / (depth * g->gi.lattice)) * depth * g->gi.lattice;
        }

        if (it->count >= (0.6 * rows * cols * depth)) {
            it->x = -1;
            it->y = -1;
            it->z = -1;
        }
    }
}

double exchange_from_micromagnetic(double A, double lattice, double atoms_per_cell) {
    return lattice * A * atoms_per_cell;
}

double dm_from_micromagnetic(double D, double lattice, double atoms_per_cell) {
    return D * (lattice * atoms_per_cell) * (lattice * atoms_per_cell);
}

double anisotropy_from_micromagnetic(double K, double lattice, double atoms_per_cell) {
    return K * (lattice * atoms_per_cell) * (lattice * atoms_per_cell) * (lattice * atoms_per_cell);
}

double mu_from_micromagnetic(double Ms, double lattice, double atoms_per_cell) {
    return Ms * (lattice * atoms_per_cell) * (lattice * atoms_per_cell) * (lattice * atoms_per_cell);
}
