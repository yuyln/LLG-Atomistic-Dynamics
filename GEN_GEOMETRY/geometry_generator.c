#include <stdio.h>
#include <stdlib.h>


#define PRINT_FORMAT "%d\t%d\t0.0\t0.0\t-1.0\n", y, x
#define __SHAPES_C
#include "shapes.h"

//PRIMITIVE TYPES FOR DRAWING
//##################
//    triangle     #
//    quad         #
//    line         #
//    n_side       #
//    circle       #
//    ellipse      #
//##################

int main(void) {
	FILE *f_out = fopen("./pinning.in", "w");
	int cols = 272;
	int rows = 272;

	ellipse e = ellipse_center_angle(i_v2d(cols / 2, rows / 2), i_v2d(60, 120), RAD(0));
	ellipse_discrete_to_file(f_out, e, 0, cols, 0, rows);

	fclose(f_out);
	return 0;
}