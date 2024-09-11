#ifndef __GRID_FUNCS_H
#define __GRID_FUNCS_H

#include "grid_types.h"
#include "gpu.h"

typedef struct {
    uint64_t row;
    uint64_t col;
    enum {
        UNDEFINED,
        NOISE,
        CLUSTER
    } label;
    uint64_t cluster;
    double x;
    double y;
} cluster_point;

typedef struct {
    double x;
    double y;

    double row;
    double col;

    uint64_t id;
    uint64_t count;
    v3d avg_m;
    double sum_weight;
} cluster_center;

typedef struct {
    cluster_center *items;
    uint64_t len;
    uint64_t cap;
} cluster_centers;

typedef struct {
    uint64_t *items;
    uint64_t start;
    uint64_t len;
    uint64_t cap;
} cluster_queue;

typedef struct {
    grid_info gi;
    grid_site_params *gp;
    v3d *m;

    cl_mem gp_gpu;
    cl_mem m_gpu;

    bool on_gpu;

    cluster_point *points;
    cluster_centers clusters;
    cluster_queue queue;
    bool *seen;
} grid;

double shit_random(double from, double to);
grid grid_init(unsigned int rows, unsigned int cols);

void grid_set_exchange_loc(grid *g, int row, int col, double exchange);
void grid_set_dm_loc(grid *g, int row, int col, dm_interaction dm);
void grid_set_lattice_loc(grid *g, int row, int col, double lattice);
void grid_set_cubic_anisotropy_loc(grid *g, int row, int col, double cubic_ani);
void grid_set_mu_loc(grid *g, int row, int col, double mu);
void grid_set_alpha_loc(grid *g, int row, int col, double alpha);
void grid_set_gamma_loc(grid *g, int row, int col, double gamma);
void grid_set_anisotropy_loc(grid *g, int row, int col, anisotropy ani);
void grid_set_pinning_loc(grid *g, int row, int col, pinning pin);
void v3d_set_at_loc(v3d *v, unsigned int rows, unsigned int cols, int row, int col, v3d m);

void grid_set_exchange(grid *g, double exchange);
void grid_set_dm(grid *g, dm_interaction dm);
void grid_set_lattice(grid *g, double lattice);
void grid_set_cubic_anisotropy(grid *g, double cubic_ani);
void grid_set_mu(grid *g, double mu);
void grid_set_alpha(grid *g, double alpha);
void grid_set_gamma(grid *g, double gamma);
void grid_set_anisotropy(grid *g, anisotropy ani);

void v3d_fill_with_random(v3d *v, unsigned int rows, unsigned int cols);
void v3d_create_skyrmion_at_old(v3d *v, unsigned int rows, unsigned int cols, int radius, int row, int col, double Q, double P, double theta);
void v3d_create_skyrmion_at(v3d *v, unsigned int rows, unsigned int cols, double radius, double dw_width, double ix, double iy, double Q, double vorticity, double _gamma);
void v3d_create_biskyrmion_at(v3d *v, unsigned int rows, unsigned int cols, double radius, double dw_width, double ix, double iy, double dr, double angle, double Q, double vorticity, double _gamma);
void v3d_create_skyrmionium_at(v3d *v, unsigned int rows, unsigned int cols, double radius, double dw_width, double ix, double iy, double Q, double vorticity, double _gamma);
void v3d_create_target_skyrmion_at(v3d *v, unsigned int rows, unsigned int cols, double radius, double dw_width, double ix, double iy, double Q, double vorticity, double _gamma, int turns);
void v3d_uniform(v3d *v, unsigned int rows, unsigned int cols, v3d dir);

void grid_fill_with_random(grid *g);
void grid_create_skyrmion_at_old(grid *g, int radius, int row, int col, double Q, double P, double theta);
void grid_create_skyrmion_at(grid *g, double radius, double dw_width, double ix, double iy, double Q, double vorticity, double _gamma);
void grid_create_biskyrmion_at(grid *g, double radius, double dw_width, double ix, double iy, double dr, double angle, double Q, double vorticity, double _gamma);
void grid_create_skyrmionium_at(grid *g, double radius, double dw_width, double ix, double iy, double Q, double vorticity, double _gamma);
void grid_create_target_skyrmion_at(grid *g, double radius, double dw_width, double ix, double iy, double Q, double vorticity, double _gamma, int turns);
void grid_uniform(grid *g, v3d dir);

bool grid_free(grid *g);
bool grid_release_from_gpu(grid *g);

void grid_to_gpu(grid *g, gpu_cl gpu);
void grid_from_gpu(grid *g, gpu_cl gpu);
void v3d_from_gpu(v3d *g, cl_mem buffer, unsigned int rows, unsigned int cols, gpu_cl gpu);

bool v3d_dump(FILE *f, v3d *v, unsigned int rows, unsigned int cols);
bool grid_dump(FILE *f, grid *g);
bool grid_dump_path(const char *path, grid *g);
bool grid_from_file(const char *path, grid *g);
bool grid_from_animation_bin(const char *path, grid *g, int64_t frame);

void grid_do_in_rect(grid *g, int64_t x0, int64_t y0, int64_t x1, int64_t y1, void(*fun)(grid*, uint64_t, uint64_t, void*), void *user_data);
void grid_do_in_ellipse(grid *g, int64_t x0, int64_t y0, int64_t a, int64_t b, void(*fun)(grid*, uint64_t, uint64_t, void*), void *user_data);
void grid_do_in_triangle(grid *g, int64_t x0, int64_t y0, int64_t x1, int64_t y1, int64_t x2, int64_t y2, void(*fun)(grid*, uint64_t, uint64_t, void*), void *user_data);
void grid_do_in_line(grid *g, int64_t x0, int64_t y0, int64_t x1, int64_t y1, int64_t thickness, void(*fun)(grid*, uint64_t, uint64_t, void*), void *user_data);

dm_interaction dm_interfacial(double value);
dm_interaction dm_bulk(double value);
anisotropy anisotropy_z_axis(double value);

void grid_cluster(grid *g, double eps, uint64_t min_pts, double(*metric)(grid*, uint64_t, uint64_t, uint64_t, uint64_t, void*), double(*weight_f)(grid*, uint64_t, uint64_t, void*), void *user_data_metric, void *user_data_weight);

double exchange_from_micromagnetic(double A, double lattice, double atoms_per_cell);
double dm_from_micromagnetic(double D, double lattice, double atoms_per_cell);
double anisotropy_from_micromagnetic(double K, double lattice, double atoms_per_cell);
double mu_from_micromagnetic(double Ms, double lattice, double atoms_per_cell);

#endif
