#ifndef __v3d
#define __v3d

#ifndef OPENCLCOMP
#include <math.h>
#include <stdlib.h>
#endif

typedef struct {
    double x, y, z;
} v3d;

v3d v3d_s(double scalar) {
    return (v3d){scalar, scalar, scalar};
}

v3d v3d_c(double x, double y, double z) {
    return (v3d){x, y, z};
}

v3d v3d_add(v3d v1, v3d v2) {
    v1.x += v2.x;
    v1.y += v2.y;
    v1.z += v2.z;
    return v1;
}

v3d v3d_sub(v3d v1, v3d v2) {
    v1.x -= v2.x;
    v1.y -= v2.y;
    v1.z -= v2.z;
    return v1;
}

v3d v3d_scalar(v3d v1, double scalar) {
    v1.x *= scalar;
    v1.y *= scalar;
    v1.z *= scalar;
    return v1;
}

v3d v3d_normalize(v3d v) {
    double v2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (v2 > 0.0) {
        double v1 = 1.0 / sqrt(v2);
        return v3d_scalar(v, v1);
    }
    return v;
}

v3d v3d_normalize_to(v3d v, double V) {
    double v2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (v2 > 0.0) {
        double v1 = V / sqrt(v2);
        return v3d_scalar(v, v1);
    }
    return v;
}

v3d v3d_cross(v3d v1, v3d v2) {
    v3d ret;
    ret.x = v1.y * v2.z - v1.z * v2.y;
    ret.y = v1.z * v2.x - v1.x * v2.z;
    ret.z = v1.x * v2.y - v1.y * v2.x;
    return ret;
}

double v3d_dot(v3d v1, v3d v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
#endif
