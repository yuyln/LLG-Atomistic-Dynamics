#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define __SHAPES_C
#include "shapes.h"

#define __GEN_LATTICE_UTILS_C
#include "lattice_gen_utils.h"

// PRIMITIVE TYPES FOR DRAWING
// ##################
//     triangle     #
//     quad         #
//     line         #
//     n_side       #
//     circle       #
//     ellipse      #
// ##################

int main(void) {
    FILE *f_out = fopen("./starting.in", "wb");
    int cols = 64;
    int rows = 64;
    v3d *grid = (v3d*)calloc(cols * rows, sizeof(v3d));
    double rskyr = 4.0;
    for (int I = 0; I < cols * rows; ++I)
        grid[I] = (v3d){0.0, 0.0, 1.0};
    create_skyrmion_neel(grid, rows, cols, cols / 2.0, rows / 2.0, rskyr, 1.0, -1.0);

    dump_grid(f_out, grid, rows, cols);
    free(grid);
    fclose(f_out);
    return 0;
}
