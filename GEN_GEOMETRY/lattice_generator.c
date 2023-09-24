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
	FILE *f_out = fopen("./starting.in", "w");
	int cols = 272 * 4;
	int rows = 272 / 4;
	v3d *grid = (v3d*)calloc(cols * rows, sizeof(v3d));
	double rskyr = 4.0;

	for (int i = 0; i < cols * rows; ++i)
		grid[i] = (v3d){0.0, 0.0, 1.0};


    int npin = 32;
	double rpin = 11.0 / 4.0;
	double d = (cols - 2.0 * npin * rpin) / (double)npin;

    int skyr_rows = 1;
    for (int s = 0; s < skyr_rows; ++s) {
        double cy = (rows - 1) / 2;
        for (int i = 0; i < npin; ++i) {
            if (i == 2) continue;
            double x = d / 2.0 + i * d + (2 * i + 1) * rpin;
            create_skyrmion_neel(grid, rows, cols, x + rpin + rskyr, cy, rskyr, 1.0, -1.0);
        }
    }

    print_grid(f_out, grid, rows, cols);
    free(grid);
    fclose(f_out);
    return 0;
}
