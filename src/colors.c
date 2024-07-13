#include "colors.h"

RGBA32 linear_mapping(double t, v3d start, v3d middle, v3d end) {
    v3d color;
    if (t < 0.5) {
        color = v3d_sum(v3d_scalar(v3d_sub(middle, start), 2.0 * t), start);
    } else {
        color = v3d_sum(v3d_scalar(v3d_sub(end, middle), 2.0 * t - 1.0), middle);
    }
    //RGBA -> BGRA
    return (RGBA32){.x = color.z * 255, .y = color.y * 255, .z = color.x * 255, .w = 255};
}

RGBA32 m_bwr_mapping(v3d m) {
    //v3d start = v3d_c(0x03 / 255.0, 0x7f / 255.0, 0xff / 255.0);
    //v3d middle = v3d_c(1, 1, 1);
    //v3d end = v3d_c(0xf4 / 255.0, 0x05 / 255.0, 0x01 / 255.0);

    v3d start = v3d_c(0, 0, 0xff / 255.0);
    v3d middle = v3d_c(1, 1, 1);
    v3d end = v3d_c(0xff / 255.0, 0, 0);

    double mz = m.z;
    return linear_mapping(0.5 * mz + 0.5, start, middle, end);
}

double _v(double m1, double m2, double hue) {
    //hue = hue % 1.0;
    int hue_i = floor(hue);
    hue = hue - hue_i;
    if (hue < (1.0 / 6.0))
        return m1 + (m2 - m1) * hue * 6.0;
    if (hue < 0.5)
        return m2;
    if (hue < (2.0 / 3.0))
        return m1 + (m2 - m1) * (2.0 / 3.0 - hue) * 6.0;
    return m1;
}

RGBA32 hsl_to_rgb(double h, double s, double l) {
    if (CLOSE_ENOUGH(s, 0.0, EPS))
        return (RGBA32){.x = 255 * l, .y = 255 * l, .z = 255 * l, .w = 255};
    double m2;
    if (l <= 0.5)
        m2 = l * (1.0 + s);
    else
        m2 = l + s - l * s;
    double m1 = 2.0 * l - m2;
    RGBA32 ret = (RGBA32){.x = _v(m1, m2, h + 1.0 / 3.0) * 255, .y = _v(m1, m2, h) * 255, .z = _v(m1, m2, h - 1.0 / 3.0) * 255, .w = 255};
    //RGBA -> BGRA
    return (RGBA32){.x = ret.z, .y = ret.y, .z = ret.x, .w = ret.w};
}

RGBA32 m_to_hsl(v3d m) {
    //m = v3d_normalize(m);
    double angle = atan2(m.y, m.x) / M_PI;
    angle = (angle + 1.0) / 2.0;
    double l = (m.z + 1.0) / 2.0;
    double s = 1.0;
    return hsl_to_rgb(angle, s, l);
}
