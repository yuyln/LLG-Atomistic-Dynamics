#include <stdio.h>
#include <stdlib.h>

#define __SHAPES_C
#include "shapes.h"

// PRIMITIVE TYPES FOR DRAWING
// ##################
//     triangle     #
//     quad         #
//     line         #
//     n_side       #
//     circle       #
//     ellipse      #
// ##################

int main(void)
{
	FILE *f_out = fopen("./pinning.in", "w");
	int cols = 128;
	int rows = 64;

    quad upper = quad_center_angle(i_v2d(cols / 2, rows - 5), i_v2d(cols + 1, 10), 0);
    quad bottom = quad_center_angle(i_v2d(cols / 2, 5), i_v2d(cols + 1, 10), 0);

    quad_discrete_to_file(f_out, upper, 0, cols, 0, rows, "0.0\t0.0\t1.0");
    quad_discrete_to_file(f_out, bottom, 0, cols, 0, rows, "0.0\t0.0\t1.0");

	fclose(f_out);
	return 0;
}
