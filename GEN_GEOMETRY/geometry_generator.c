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
	int rows = 112;

	ellipse e = ellipse_center_angle(i_v2d(cols / 2, rows / 2), i_v2d(15, 30), RAD(0));
	ellipse_discrete_to_file(f_out, e, 0, cols, 0, rows, NULL);

	quad q = quad_center_angle(i_v2d(cols / 4, rows / 4), i_v2d(15, 30), RAD(0));
	quad_discrete_to_file(f_out, q, 0, cols, 0, rows, NULL);

	q = quad_center_angle(i_v2d(3 * cols / 4, rows / 4), i_v2d(15, 30), RAD(60));
	quad_discrete_to_file(f_out, q, 0, cols, 0, rows, NULL);

	triangle t = triangle_center_angle(i_v2d(3 * cols / 4, 3 * rows / 4), i_v2d(30, 60), RAD(30), RAD(0));
	triangle_discrete_to_file(f_out, t, 0, cols, 0, rows, "0.0\t0.0\t1.0");

	fclose(f_out);
	return 0;
}
