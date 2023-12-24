#include <time.h>
#include <stdlib.h>

#include "grid_funcs.h"
#include "integrate.h"

//@TODO: Change openclwrapper to print file and location correctly
//@TODO: Check uint64_t->int changes
//@TODO: Do 3D
int main(void) {
    srand(time(NULL));
    double dt = 1.0e-15;
    grid g = grid_init(272, 272);
    v3d_fill_with_random(g.m, g.gi.rows, g.gi.cols);
    grid_set_anisotropy(&g, (anisotropy){.ani=0.02 * QE * 1.0e-3, .dir = v3d_c(0.0, 0.0, 1.0)});
    grid_set_dm(&g, 0.1 * QE * 1.0e-3, 0.0, 0);
    integrate(&g, dt, 1.0 * NS, 100, 100000, (string_view){0}, (string_view){0}, "./output/");

    grid_free(&g);
    return 0;
}
