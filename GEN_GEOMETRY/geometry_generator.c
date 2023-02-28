#include <stdio.h>
#include <stdlib.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include <stdbool.h>
#include <float.h>

typedef struct {int x, y;} v2i;
typedef struct {double x, y;} v2d;
typedef struct {v2d p1, p2, p3;} triangle;
typedef struct {v2d p1, p2, p3, p4;} quad;
typedef struct {v2d p1, p2; double thick;} line;
typedef struct {int sides; v2d l, center, max_d, min_d; double rotation;} n_side;

v2d i_v2d(double x, double y);
v2d s_v2d(double x);
v2d v2d_add(v2d a, v2d b);
v2d v2d_sub(v2d a, v2d b);
v2d v2d_sca(v2d a, double s);
v2d rotate(v2d o, double angle);

triangle triangle_center_angle(v2d center, v2d size, double angle, double rot);
void triangle_discrete_to_file(FILE *file, triangle t, int xmin, int xmax, int ymin, int ymax);
bool triangle_inside(v2d p, triangle t);

quad quad_center_angle(v2d center, v2d size, double angle);
void quad_discrete_to_file(FILE *file, quad q, int xmin, int xmax, int ymin, int ymax);
bool quad_inside(v2d p, quad q);

line line_points(v2d p1, v2d p2, double thickness);
void line_discrete_to_file_smooth(FILE *file, line l, int xmin, int xmax, int ymin, int ymax);
void line_discrete_to_file_quad(FILE *file, line l, int xmin, int xmax, int ymin, int ymax);
bool line_inside_smooth(v2d p, line l);
bool line_inside_quad(v2d p, line l);

n_side n_side_center_angle(v2d center, v2d l, int n, double rot);
void n_side_discrete_to_file(FILE *file, n_side t, int xmin, int xmax, int ymin, int ymax);
bool n_side_inside(v2d p, n_side t);



int main(void) {
	FILE *f_out = fopen("./pinning.in", "w");
	int cols = 272;
	int rows = 272;

	n_side n = n_side_center_angle(i_v2d(cols / 2, rows / 2), i_v2d(100, 50), 8, -M_PI / 8.0 + M_PI / 4.0);
	n_side_discrete_to_file(f_out, n, 0, cols, 0, rows);

	fclose(f_out);
}


v2d i_v2d(double x, double y) {
    return (v2d){x, y};
}

v2d s_v2d(double x) {
    return (v2d){x, x};
}

v2d v2d_add(v2d a, v2d b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

v2d v2d_sub(v2d a, v2d b) {
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

v2d v2d_sca(v2d a, double s) {
    a.x *= s;
    a.y *= s;
    return a;
}

v2d rotate(v2d o, double angle) {
	v2d ret;
	ret.x = cos(angle) * o.x - sin(angle) * o.y;
	ret.y = sin(angle) * o.x + cos(angle) * o.y;
	return ret;
}

bool triangle_inside(v2d p, triangle t) {
	double alpha = ((t.p2.y - t.p3.y)*(p.x - t.p3.x) + (t.p3.x - t.p2.x)*(p.y - t.p3.y)) /
        ((t.p2.y - t.p3.y)*(t.p1.x - t.p3.x) + (t.p3.x - t.p2.x)*(t.p1.y - t.p3.y));
	double beta = ((t.p3.y - t.p1.y)*(p.x - t.p3.x) + (t.p1.x - t.p3.x)*(p.y - t.p3.y)) /
		((t.p2.y - t.p3.y)*(t.p1.x - t.p3.x) + (t.p3.x - t.p2.x)*(t.p1.y - t.p3.y));
	double gamma = 1.0f - alpha - beta;
	return alpha >= 0 && beta >= 0 && gamma >= 0;
}

triangle triangle_center_angle(v2d center, v2d size, double angle, double rot) {
	triangle ret = {0};
	v2d p1, p2, p3;

	p1 = s_v2d(0.0f);
	p2 = i_v2d(size.y * cos(angle), size.y * sin(angle));
	p3 = i_v2d(size.x, 0);

    v2d C = v2d_sca(i_v2d(p1.x + p2.x + p3.x, p1.y + p2.y + p3.y), 1.0f / 3.0f);

    p1 = v2d_sub(p1, C);
    p2 = v2d_sub(p2, C);
    p3 = v2d_sub(p3, C);

	p1 = rotate(p1, rot);
	p2 = rotate(p2, rot);
	p3 = rotate(p3, rot);

    ret.p1 = v2d_add(p1, center);
    ret.p2 = v2d_add(p2, center);
    ret.p3 = v2d_add(p3, center);

	return ret;
}

void triangle_discrete_to_file(FILE *file, triangle t, int xmin, int xmax, int ymin, int ymax) {
	double min_y = t.p1.y;
	min_y = t.p2.y < min_y? t.p2.y : min_y;
	min_y = t.p3.y < min_y? t.p3.y : min_y;
	double max_y = t.p1.y;
	max_y = t.p2.y > max_y? t.p2.y : max_y;
	max_y = t.p3.y > max_y? t.p3.y : max_y;

	double min_x = t.p1.x;
	min_x = t.p2.x < min_x? t.p2.x : min_x;
	min_x = t.p3.x < min_x? t.p3.x : min_x;
	double max_x = t.p1.x;
	max_x = t.p2.x > max_x? t.p2.x : max_x;
	max_x = t.p3.x > max_x? t.p3.x : max_x;

	for (int y = (int)min_y; y < (int)max_y; ++y) {
		if (y < ymin || y >= ymax) continue;
		for (int x = (int)min_x; x < (int)max_x; ++x) {
			if (x < xmin || x >= xmax) continue;
			v2d p = i_v2d(x, y);
			if (triangle_inside(p, t)) fprintf(file, "%d\t%d\t0.0\t0.0\t-1.0\n", y, x);
		}
	}	
}

quad quad_center_angle(v2d center, v2d size, double angle) {
    quad ret = {0};
    v2d tmp1 = v2d_sub(s_v2d(0), v2d_sca(size, 0.5));
    v2d tmp2 = v2d_add(s_v2d(0), i_v2d(size.x * 0.5, -size.y * 0.5));
    v2d tmp3 = v2d_add(s_v2d(0), v2d_sca(size, 0.5));
    v2d tmp4 = v2d_add(s_v2d(0), i_v2d(-size.x * 0.5, size.y * 0.5));

    tmp1 = rotate(tmp1, angle);
    tmp2 = rotate(tmp2, angle);
    tmp3 = rotate(tmp3, angle);
    tmp4 = rotate(tmp4, angle);

    ret.p1 = v2d_add(tmp1, center);
    ret.p2 = v2d_add(tmp2, center);
    ret.p3 = v2d_add(tmp3, center);
    ret.p4 = v2d_add(tmp4, center);
    return ret;
}

bool quad_inside(v2d p, quad q) {
    v2d a = v2d_sub(q.p2, q.p1);
    v2d b = v2d_sub(q.p3, q.p2);
    p = v2d_sub(p, q.p1);
    double lambda = b.x * p.y - b.y * p.x;
    double mu = a.y * p.x - a.x * p.y;
    double den = a.y * b.x - a.x * b.y;
    lambda /= den;
    mu /= den;
    return lambda < 1.0f && mu < 1.0f && lambda > 0.0f && mu > 0.0f;
}

void quad_discrete_to_file(FILE *file, quad q, int xmin, int xmax, int ymin, int ymax) {
	double min_y = q.p1.y;
	min_y = q.p2.y < min_y? q.p2.y : min_y;
	min_y = q.p3.y < min_y? q.p3.y : min_y;
	min_y = q.p4.y < min_y? q.p4.y : min_y;
	double max_y = q.p1.y;
	max_y = q.p2.y > max_y? q.p2.y : max_y;
	max_y = q.p3.y > max_y? q.p3.y : max_y;
	max_y = q.p4.y > max_y? q.p4.y : max_y;

	double min_x = q.p1.x;
	min_x = q.p2.x < min_x? q.p2.x : min_x;
	min_x = q.p3.x < min_x? q.p3.x : min_x;
	min_x = q.p4.x < min_x? q.p4.x : min_x;
	double max_x = q.p1.x;
	max_x = q.p2.x > max_x? q.p2.x : max_x;
	max_x = q.p3.x > max_x? q.p3.x : max_x;
	max_x = q.p4.x > max_x? q.p4.x : max_x;

	for (int y = (int)min_y; y < (int)max_y; ++y) {
		if (y < ymin || y >= ymax) continue;
		for (int x = (int)min_x; x < (int)max_x; ++x) {
			if (x < xmin || x >= xmax) continue;
			v2d p = i_v2d(x, y);
			if (quad_inside(p, q)) fprintf(file, "%d\t%d\t0.0\t0.0\t-1.0\n", y, x);
		}
	}	
}

line line_points(v2d p1, v2d p2, double thickness) {
	line ret = {0};
	ret.p1 = p1;
	ret.p2 = p2;
	ret.thick = thickness;
	return ret;
}

v2d normal_to_line(line l) {
	v2d d = v2d_sub(l.p2, l.p1);
	double N = sqrt(d.x * d.x + d.y * d.y);
	d = v2d_sca(d, 1.0 / N);
	return rotate(d, M_PI / 2.0);
}

double distance2_line_point(v2d p, line l) {
	v2d dir = v2d_sub(l.p2, l.p1);
	double N = sqrt(dir.x * dir.x + dir.y * dir.y);
	dir = v2d_sca(dir, 1.0 / N);

	v2d p1_p = v2d_sub(p, l.p1);

	double proj = dir.x * p1_p.x + dir.y * p1_p.y;

	if (proj > N) {
		v2d distance = v2d_sub(l.p2, p);
		return distance.x * distance.x + distance.y * distance.y;
	} else if (proj < 0.0) {
		v2d distance = v2d_sub(l.p1, p);
		return distance.x * distance.x + distance.y * distance.y;
	}

	v2d pl = v2d_add(l.p1, v2d_sca(dir, proj));
	v2d distance = v2d_sub(p, pl);
	return distance.x * distance.x + distance.y * distance.y;
}

void line_discrete_to_file_smooth(FILE *file, line l, int xmin, int xmax, int ymin, int ymax) {
	double min_y = l.p1.y;
	min_y = l.p2.y < min_y? l.p2.y : min_y;
	double max_y = l.p1.y;
	max_y = l.p2.y > max_y? l.p2.y : max_y;

	double min_x = l.p1.x;
	min_x = l.p2.x < min_x? l.p2.x : min_x;

	double max_x = l.p1.x;
	max_x = l.p2.x > max_x? l.p2.x : max_x;

	for (int y = (int)min_y - l.thick; y < (int)max_y + l.thick; ++y) {
		if (y < ymin || y >= ymax) continue;
		for (int x = (int)min_x - l.thick; x < (int)max_x + l.thick; ++x) {
			if (x < xmin || x >= xmax) continue;
			v2d p = i_v2d(x, y);
			if (line_inside_smooth(p, l)) fprintf(file, "%d\t%d\t0.0\t0.0\t-1.0\n", y, x);
		}
	}
}

void line_discrete_to_file_quad(FILE *file, line l, int xmin, int xmax, int ymin, int ymax) {
	double min_y = l.p1.y;
	min_y = l.p2.y < min_y? l.p2.y : min_y;
	double max_y = l.p1.y;
	max_y = l.p2.y > max_y? l.p2.y : max_y;

	double min_x = l.p1.x;
	min_x = l.p2.x < min_x? l.p2.x : min_x;

	double max_x = l.p1.x;
	max_x = l.p2.x > max_x? l.p2.x : max_x;

	for (int y = (int)min_y - l.thick; y < (int)max_y + l.thick; ++y) {
		if (y < ymin || y >= ymax) continue;
		for (int x = (int)min_x - l.thick; x < (int)max_x + l.thick; ++x) {
			if (x < xmin || x >= xmax) continue;
			v2d p = i_v2d(x, y);
			if (line_inside_quad(p, l)) fprintf(file, "%d\t%d\t0.0\t0.0\t-1.0\n", y, x);
		}
	}
}

bool line_inside_smooth(v2d p, line l) {
	double d2 = distance2_line_point(p, l);
	double R = l.thick / 2.0;
	return d2 < (R * R);
}

bool line_inside_quad(v2d p, line l) {
	quad ql = {0};
	v2d n = normal_to_line(l);
	ql.p1 = v2d_add(l.p1, v2d_sca(n, l.thick / 2.0));
	ql.p2 = v2d_sub(l.p1, v2d_sca(n, l.thick / 2.0));
	ql.p3 = v2d_add(l.p2, v2d_sca(n, l.thick / 2.0));
	ql.p4 = v2d_sub(l.p2, v2d_sca(n, l.thick / 2.0));
	return quad_inside(p, ql);
}

n_side n_side_center_angle(v2d center, v2d l, int n, double rot) {
	n_side ret = {0};
	ret.center = center;
	ret.l = l;
	ret.rotation = rot;
	ret.sides = n;

	ret.min_d = s_v2d(INFINITY);
	ret.max_d = s_v2d(-INFINITY);

	double deph = 2.0 * M_PI / (double)ret.sides;
	double dt = deph;
	for (int i = 0; i < ret.sides; ++i) {
		double theta = i * dt - deph / 2.0;
		triangle piece = {
			.p1 = i_v2d(0.0, 0.0),
			.p2 = i_v2d(cos(theta) * ret.l.x, sin(theta) * ret.l.y),
			.p3 = i_v2d(cos(theta + dt) * ret.l.x, sin(theta + dt) * ret.l.y)
		};

		piece.p1 = v2d_add(ret.center, rotate(piece.p1, ret.rotation));
		piece.p2 = v2d_add(ret.center, rotate(piece.p2, ret.rotation));
		piece.p3 = v2d_add(ret.center, rotate(piece.p3, ret.rotation));

		ret.min_d.y = ret.min_d.y < piece.p1.y? ret.min_d.y: piece.p1.y;
		ret.min_d.y = ret.min_d.y < piece.p2.y? ret.min_d.y: piece.p2.y;
		ret.min_d.y = ret.min_d.y < piece.p3.y? ret.min_d.y: piece.p3.y;

		ret.min_d.x = ret.min_d.x < piece.p1.x? ret.min_d.x: piece.p1.x;
		ret.min_d.x = ret.min_d.x < piece.p2.x? ret.min_d.x: piece.p2.x;
		ret.min_d.x = ret.min_d.x < piece.p3.x? ret.min_d.x: piece.p3.x;

		ret.max_d.y = ret.max_d.y > piece.p1.y? ret.max_d.y: piece.p1.y;
		ret.max_d.y = ret.max_d.y > piece.p2.y? ret.max_d.y: piece.p2.y;
		ret.max_d.y = ret.max_d.y > piece.p3.y? ret.max_d.y: piece.p3.y;

		ret.max_d.x = ret.max_d.x > piece.p1.x? ret.max_d.x: piece.p1.x;
		ret.max_d.x = ret.max_d.x > piece.p2.x? ret.max_d.x: piece.p2.x;
		ret.max_d.x = ret.max_d.x > piece.p3.x? ret.max_d.x: piece.p3.x;
	}

	return ret;
}

void n_side_discrete_to_file(FILE *file, n_side t, int xmin, int xmax, int ymin, int ymax) {
	for (int y = t.min_d.y; y < t.max_d.y; ++y) {
		if (y < ymin || y >= ymax) continue;
		for (int x = t.min_d.x; x < t.max_d.x; ++x) {
			if (x < xmin || x >= xmax) continue;
			v2d p = i_v2d(x, y);
			if (n_side_inside(p, t)) fprintf(file, "%d\t%d\t0.0\t0.0\t-1.0\n", y, x);
		}
	}

}

bool n_side_inside(v2d p, n_side t) {
	double deph = 2.0 * M_PI / (double)t.sides;
	double dt = deph;
	for (int i = 0; i < t.sides; ++i) {
		double theta = i * dt - deph / 2.0;
		triangle piece = {
			.p1 = i_v2d(0.0, 0.0),
			.p2 = i_v2d(cos(theta) * t.l.x, sin(theta) * t.l.y),
			.p3 = i_v2d(cos(theta + dt) * t.l.x, sin(theta + dt) * t.l.y)
		};

		piece.p1 = v2d_add(t.center, rotate(piece.p1, t.rotation));
		piece.p2 = v2d_add(t.center, rotate(piece.p2, t.rotation));
		piece.p3 = v2d_add(t.center, rotate(piece.p3, t.rotation));

		if (triangle_inside(p, piece)) return true;

	}
	return false;
}
