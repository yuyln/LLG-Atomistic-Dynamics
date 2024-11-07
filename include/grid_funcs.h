#ifndef __GRID_FUNCS_H
#define __GRID_FUNCS_H

#include "grid_types.h"
#include "gpu.h"

typedef struct {
    uint64_t i;
    uint64_t j;
    uint64_t k;
    enum {
        UNDEFINED,
        NOISE,
        CLUSTER
    } label;
    uint64_t cluster;
    double x;
    double y;
    double z;
} cluster_point;

typedef struct {
    double x;
    double y;
    double z;

    double i;
    double j;
    double k;

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
    uint64_t dimensions;
} grid;

double shit_random(double from, double to);
grid grid_init(unsigned int rows, unsigned int cols, unsigned int depth);

void grid_set_exchange_loc(grid *g, int i, int j, int k, exchange_interaction exchange);
void grid_set_dm_loc(grid *g, int i, int j, int k, dm_interaction dm);
void grid_set_cubic_anisotropy_loc(grid *g, int i, int j, int k, double cubic_ani);
void grid_set_mu_loc(grid *g, int i, int j, int k, double mu);
void grid_set_alpha_loc(grid *g, int i, int j, int k, double alpha);
void grid_set_gamma_loc(grid *g, int i, int j, int k, double gamma);
void grid_set_anisotropy_loc(grid *g, int i, int j, int k, anisotropy ani);
void grid_set_pinning_loc(grid *g, int i, int j, int k, pinning pin);
void v3d_set_at_loc(v3d *v, unsigned int rows, unsigned int cols, unsigned int depth, int i, int j, int k, v3d m);

void grid_set_exchange(grid *g, exchange_interaction exchange);
void grid_set_dm(grid *g, dm_interaction dm);
void grid_set_lattice(grid *g, double lattice);
void grid_set_cubic_anisotropy(grid *g, double cubic_ani);
void grid_set_mu(grid *g, double mu);
void grid_set_alpha(grid *g, double alpha);
void grid_set_gamma(grid *g, double gamma);
void grid_set_anisotropy(grid *g, anisotropy ani);

void v3d_fill_with_random(v3d *v, unsigned int rows, unsigned int cols, unsigned int depth);
void v3d_create_skyrmion_at(v3d *v, unsigned int rows, unsigned int cols, unsigned int depth, double radius, double dw_width, double ix, double iy, double Q, double vorticity, double _gamma);
void v3d_create_biskyrmion_at(v3d *v, unsigned int rows, unsigned int cols, unsigned int depth, double radius, double dw_width, double ix, double iy, double dr, double angle, double Q, double vorticity, double _gamma);
void v3d_create_skyrmionium_at(v3d *v, unsigned int rows, unsigned int cols, unsigned int depth, double radius, double dw_width, double ix, double iy, double Q, double vorticity, double _gamma);
void v3d_create_hopfion_at(v3d *v, unsigned int rows, unsigned int cols, unsigned int depth, double radius, double height, double ix, double iy, double iz, double factor);
void v3d_uniform(v3d *v, unsigned int rows, unsigned int cols, unsigned int depth, v3d dir);

void grid_fill_with_random(grid *g);
void grid_create_skyrmion_at(grid *g, double radius, double dw_width, double ix, double iy, double Q, double vorticity, double _gamma);
void grid_create_biskyrmion_at(grid *g, double radius, double dw_width, double ix, double iy, double dr, double angle, double Q, double vorticity, double _gamma);
void grid_create_skyrmionium_at(grid *g, double radius, double dw_width, double ix, double iy, double Q, double vorticity, double _gamma);
void grid_create_hopfion_at(grid *g, double radius, double height, double ix, double iy, double iz, double factor);
void grid_uniform(grid *g, v3d dir);

bool grid_free(grid *g);
bool grid_release_from_gpu(grid *g);

void grid_to_gpu(grid *g, gpu_cl gpu);
void grid_from_gpu(grid *g, gpu_cl gpu);
void v3d_from_gpu(v3d *g, cl_mem buffer, unsigned int rows, unsigned int cols, unsigned int depth, gpu_cl gpu);

bool v3d_dump(FILE *f, v3d *v, unsigned int rows, unsigned int cols, unsigned int depth);
bool grid_dump(FILE *f, grid *g);
bool grid_dump_path(const char *path, grid *g);
bool grid_from_file(const char *path, grid *g);
bool grid_from_animation_bin(const char *path, grid *g, int64_t frame);

dm_interaction dm_interfacial(double value);
dm_interaction dm_bulk(double value);
exchange_interaction isotropic_exchange(double value);
anisotropy anisotropy_z_axis(double value);

void grid_cluster(grid *g, double eps, uint64_t min_pts, double(*metric)(grid*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, void*), double(*weight_f)(grid*, uint64_t, uint64_t, uint64_t, void*), void *user_data_metric, void *user_data_weight);

double exchange_from_micromagnetic(double A, double lattice, double atoms_per_cell);
double dm_from_micromagnetic(double D, double lattice, double atoms_per_cell);
double anisotropy_from_micromagnetic(double K, double lattice, double atoms_per_cell);
double mu_from_micromagnetic(double Ms, double lattice, double atoms_per_cell);

void grid_do_in_prism(grid *g, v3d p0, v3d p1, v3d p2, v3d p3, void(*func)(grid*, uint64_t, uint64_t, uint64_t, void*), void *user_data);
void grid_do_in_triangle(grid *g, v3d p0, v3d p1, v3d p2, void(*func)(grid*, uint64_t, uint64_t, uint64_t, void*), void *user_data);
//you may be wondering: why the heck does this rect function take 3 positions instead of 4?
//To which I respond: 4 points do not define a plane. This should be obvious, since I used 4 points to create a prism. Therefore this function can not accept 4 points. Well, in reality this function does accept 4 points, but the 4 point is created by (0-index) p3=p0+(p1-p0)+(p2-p0), this way a plane is defined and everything is fine.
//Graphics programming goes burrrr /j
void grid_do_in_rect(grid *g, v3d p0, v3d p1, v3d p2, void(*func)(grid*, uint64_t, uint64_t, uint64_t, void*), void *user_data);
void grid_do_in_ellipsoid(grid *g, v3d center, v3d size, void(*func)(grid*, uint64_t, uint64_t, uint64_t, void*), void *user_data);
void grid_do_in_8pts(grid *g, v3d p0, v3d p1, v3d p2, v3d p3, v3d p4, v3d p5, v3d p6, v3d p7, void(*func)(grid*, uint64_t, uint64_t, uint64_t, void*), void *user_data); //TODO: Better name

#endif
