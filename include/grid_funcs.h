#ifndef __GRID_FUNCS_H
#define __GRID_FUNCS_H

#include "grid_types.h"
#include "gpu.h"

typedef struct {
    grid_info gi;
    grid_site_params *gp;
    v3d *m;

    cl_mem gp_buffer;
    cl_mem m_buffer;

    bool on_gpu;
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
void v3d_create_skyrmion(v3d *v, unsigned int rows, unsigned int cols, int radius, int row, int col, double Q, double P, double theta);

bool grid_free(grid *g);
bool grid_release_from_gpu(grid *g);

void grid_to_gpu(grid *g, gpu_cl gpu);
void grid_from_gpu(grid *g, gpu_cl gpu);
void v3d_from_gpu(v3d *g, cl_mem buffer, unsigned int rows, unsigned int cols, gpu_cl gpu);

bool v3d_dump(FILE *f, v3d *v, unsigned int rows, unsigned int cols);
bool grid_dump(FILE *f, grid *g);
bool grid_from_file(string path, grid *g);

void grid_do_in_rect(grid *g, int64_t x0, int64_t y0, int64_t x1, int64_t y1, void(*fun)(grid *g, uint64_t row, uint64_t col));
void grid_do_in_ellipse(grid *g, int64_t x0, int64_t y0, int64_t a, int64_t b, void(*fun)(grid *g, uint64_t row, uint64_t col));
void grid_do_in_line(grid *g, int64_t x0, int64_t y0, int64_t x1, int64_t y1, int64_t thickness, void(*fun)(grid *g, uint64_t row, uint64_t col));
void grid_do_in_triangle(grid *g, int64_t x0, int64_t y0, int64_t x1, int64_t y1, int64_t x2, int64_t y2, void(*fun)(grid *g, uint64_t row, uint64_t col));
#endif
