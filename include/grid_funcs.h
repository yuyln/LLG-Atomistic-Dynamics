#ifndef __GRID_FUNCS_H
#define __GRID_FUNCS_H
#include "grid_types.h"
#include "gpu.h"

typedef struct {
    grid_info g_info;
    grid_site_param *gsp;
    v3d *m;

    cl_mem gsp_buffer;
    cl_mem m_buffer;
} grid;

double shit_random(double from, double to);
grid grid_init(unsigned int rows, unsigned int cols, unsigned int depths);

void grid_set_exchange_loc(grid *g, int row, int col, int depth, double exchange);
void grid_set_dm_loc(grid *g, int row, int col, int depth, double dm, double dm_ani, dm_symmetry dm_sym);
void grid_set_lattice_loc(grid *g, int row, int col, int depth, double lattice);
void grid_set_cubic_anisotropy_loc(grid *g, int row, int col, int depth, double cubic_ani);
void grid_set_mu_loc(grid *g, int row, int col, int depth, double mu);
void grid_set_alpha_loc(grid *g, int row, int col, int depth, double alpha);
void grid_set_gamma_loc(grid *g, int row, int col, int depth, double gamma);
void grid_set_anisotropy_loc(grid *g, int row, int col, int depth, anisotropy ani);
void grid_set_pinning_loc(grid *g, int row, int col, int depth, pinning pin);
void v3d_set_at_loc(v3d *v, unsigned int rows, unsigned int cols, unsigned int depths, int row, int col, int depth, v3d m);

void grid_set_exchange(grid *g, double exchange);
void grid_set_dm(grid *g, double dm, double dm_ani, dm_symmetry dm_sym);
void grid_set_lattice(grid *g, double lattice);
void grid_set_cubic_anisotropy(grid *g, double cubic_ani);
void grid_set_mu(grid *g, double mu);
void grid_set_alpha(grid *g, double alpha);
void grid_set_gamma(grid *g, double gamma);
void grid_set_anisotropy(grid *g, anisotropy ani);

void grid_free(grid *g);

void grid_to_gpu(grid *g, gpu_cl gpu);
void grid_from_gpu(grid *g, cl_mem g_buffer, gpu_cl gpu);
void v3d_from_gpu(v3d *g, unsigned int rows, unsigned int cols, unsigned int depths, cl_mem grid_buffer, gpu_cl gpu);

void v3d_dump(FILE *f, v3d *v, matrix_size sz);
void grid_full_dump(FILE *f, grid *g);
#endif
