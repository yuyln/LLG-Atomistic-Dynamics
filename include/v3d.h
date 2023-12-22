#ifndef __V3D_H
#define __V3D_H

typedef struct {
    double x, y, z;
} v3d;

v3d v3d_c(double x, double y, double z);
v3d v3d_s(double x);
v3d v3d_scalar(v3d v, double s);
v3d v3d_sum(v3d v1, v3d v2);
v3d v3d_sub(v3d v1, v3d v2);
v3d v3d_cross(v3d v1, v3d v2);
v3d v3d_normalize(v3d v);
double v3d_dot(v3d v1, v3d v2);

#endif
