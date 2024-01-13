#ifndef __GRID_RENDER_H
#define __GRID_RENDER_H
#include "grid_types.h"
#include "render.h"
#include "grid_funcs.h"
#include "gpu.h"
#include "openclwrapper.h"
#include <stdint.h>

typedef struct {
    grid *g;
    gpu_cl *gpu;

    cl_mem rgba_gpu;
    RGBA32 *rgba_cpu;
    unsigned int width, height;

    uint64_t grid_hsl_id;
    uint64_t grid_bwr_id;
    uint64_t energy_id;
    uint64_t charge_id;
    uint64_t calc_energy_id;
    uint64_t calc_charge_id;

    double *buffer_cpu;
    cl_mem buffer_gpu;

} grid_renderer;

grid_renderer grid_renderer_init(grid *g, gpu_cl *gpu);
void grid_renderer_close(grid_renderer *gr);
void grid_renderer_hsl(grid_renderer *gr);
void grid_renderer_bwr(grid_renderer *gr);
void grid_renderer_energy(grid_renderer *gr, double time);
void grid_renderer_charge(grid_renderer *gr);

/*void grid_renderer_exchange_energy(grid_renderer *gr);
  void grid_renderer_eletric_field(grid_renderer *gr);
void grid_renderer_dm_energy(grid_renderer *gr);
void grid_renderer_field_energy(grid_renderer *gr);
void grid_renderer_anisotropy_energy(grid_renderer *gr);
void grid_renderer_cubic_anisotropy_energy(grid_renderer *gr);*/
#endif
