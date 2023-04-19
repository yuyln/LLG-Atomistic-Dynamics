#ifndef __VEC
#define __VEC

#ifndef OPENCLCOMP
#include <math.h>
#include <stdlib.h>
#endif

typedef struct
{
    double x, y, z;
} Vec;

Vec VecFromScalar(double scalar)
{
    return (Vec){scalar, scalar, scalar};
}

Vec VecFrom(double x, double y, double z)
{
    return (Vec){x, y, z};
}

Vec VecAdd(Vec v1, Vec v2)
{
    v1.x += v2.x;
    v1.y += v2.y;
    v1.z += v2.z;
    return v1;
}

Vec VecSub(Vec v1, Vec v2)
{
    v1.x -= v2.x;
    v1.y -= v2.y;
    v1.z -= v2.z;
    return v1;
}

Vec VecScalar(Vec v1, double scalar)
{
    v1.x *= scalar;
    v1.y *= scalar;
    v1.z *= scalar;
    return v1;
}

Vec VecNormalize(Vec v)
{
    double v2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (v2 > 0.0)
    {
        double v1 = 1.0 / sqrt(v2);
        return VecScalar(v, v1);
    }
    return v;
}

Vec VecNormalizeTo(Vec v, double V)
{
    double v2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (v2 > 0.0)
    {
        double v1 = V / sqrt(v2);
        return VecScalar(v, v1);
    }
    return v;
}

Vec VecCross(Vec v1, Vec v2)
{
    Vec ret;
    ret.x = v1.y * v2.z - v1.z * v2.y;
    ret.y = v1.z * v2.x - v1.x * v2.z;
    ret.z = v1.x * v2.y - v1.y * v2.x;
    return ret;
}

double VecDot(Vec v1, Vec v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
#endif
