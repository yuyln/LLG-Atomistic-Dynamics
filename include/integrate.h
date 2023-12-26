#ifndef __INTEGRATE_H
#define __INTEGRATE_H
#include <stdint.h>
#include <stdlib.h>
#include "gpu.h"
#include "grid_funcs.h"
#include "string_view.h"
#include "constants.h"
#include "stb_image_write.h"

void integrate(grid *grid, double dt, double duration, int interval_info, int interval_grid, string_view func_current, string_view func_field, const char *dir_out);
void integrate_step(double time, gpu_cl *gpu, uint64_t step_id, uint64_t exchange_id, uint64_t global, uint64_t local);

#endif
