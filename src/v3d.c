#include "v3d.h"

v3d v3d_c(double x, double y, double z) {
    return (v3d){.x=x, .y=y, .z=z};
}

v3d v3d_s(double x) {
    return (v3d){.x=x, .y=x, .z=x};
}

v3d v3d_scalar(v3d v, double s) {
    v.x *= s;
    v.y *= s;
    v.z *= s;
    return v;
}

v3d v3d_sum(v3d v1, v3d v2) {
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

v3d v3d_cross(v3d v1, v3d v2) {
    v3d ret = {0};
    ret.x = v1.y * v2.z - v1.z * v2.y;
    ret.y = v1.z * v2.x - v1.x * v2.z;
    ret.z = v1.x * v2.y - v1.y * v2.x;
    return ret;
}
