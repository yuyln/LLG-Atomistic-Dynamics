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

int main(void) {
	FILE *f_out = fopen("./pinning.in", "w");
	int cols = 272;
	int rows = 272;

    int nqx = 5;
    int nqy = 5;
    int pad = 20;

    int sqx = (cols - nqx * pad) / nqx;
    int sqy = (rows - nqy * pad) / nqy;
#if 0
    for (int qy = 0; qy < nqy; qy++) {
        int b = pad / 2 + qy * (sqy + pad);
        for (int qx = 0; qx < nqx; qx++) {
            int l = pad / 2 + qx * (sqx + pad);
            quad q = (quad){.p1 = i_v2d(l, b),
                            .p2 = i_v2d(l + sqx, b),
                            .p3 = i_v2d(l + sqx, b + sqy),
                            .p4 = i_v2d(l, b + sqy)};
            quad_discrete_to_file(f_out, q, 0, cols, 0, rows, NULL);
        }
    }
#else
    for (int qy = 0; qy < nqy; qy++) {
        int cy = pad / 2 + qy * (sqy + pad) + sqy / 2;
        for (int qx = 0; qx < nqx; qx++) {
            int cx = pad / 2 + qx * (sqx + pad) + sqx / 2;
            ellipse e = (ellipse){.center = i_v2d(cx, cy),
                                  .ab = i_v2d(sqx / 2.0, sqy / 2.0),
                                  .angle = 0};
            ellipse_discrete_to_file(f_out, e, 0, cols, 0, rows, NULL);
        }
    }
#endif

	fclose(f_out);
	return 0;
}
