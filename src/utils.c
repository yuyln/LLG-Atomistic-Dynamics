#include "utils.h"
#include <stdint.h>
#include <float.h>
#include <math.h>

typedef struct {
    double x;
    double y;
    double z;
    double size;
    double vx; //TODO
    double vy; //TODO
} center;

typedef struct {
    center *items;
    uint64_t len;
    uint64_t cap;
} centers;

bool organize_clusters_inplace(const char *in_path, double sample_x, double sample_y, double sample_z, double d2_threshold) {
    return organize_clusters(in_path, in_path, sample_x, sample_y, sample_z, d2_threshold);
}

bool organize_clusters(const char *in_path, const char *out_path, double sample_x, double sample_y, double sample_z, double d2_threshold) {
    FILE *f_in = mfopen(in_path, "rb");
    char *buffer = NULL;

    uint64_t len = 0;
    char c;
    while ((c = fgetc(f_in)) != EOF)
        len += 1;
    buffer = mmalloc(len + 1);
    rewind(f_in);
    fread(buffer, len, 1, f_in);
    mfclose(f_in);

    uint64_t max_n = 0;
    uint64_t n = 0;
    for (uint64_t i = 0; i < len; ++i) {
        if (buffer[i] == ',')
            n += 1;
        if (buffer[i] == '\n') {
            max_n = n >= max_n? n: max_n;
            n = 0;
        }
    }
    bool has_size = max_n % 4 == 0;
    uint64_t div_fac = has_size? 4: 3;

    if (max_n % div_fac) {
        logging_log(LOG_ERROR, "Number of commas is not a multiple of %llu. \"%s\" is probably corrupted", div_fac, in_path);
        return false;
    }

    max_n /= div_fac;

    centers cs[3] = {0};

    for (uint64_t i = 0; i < max_n; ++i) {
        da_append(&cs[0], ((center){.x = -1, .y = -1, .z = -1}));
        da_append(&cs[1], ((center){.x = -1, .y = -1, .z = -1}));
        da_append(&cs[2], ((center){.x = -1, .y = -1, .z = -1}));
    }

    char *ptr = buffer;

    FILE *fout = mfopen(out_path, "wb");
    {
        char *line_end = ptr;
        while (*line_end != '\n')
            line_end += 1;

        char *first_comma = NULL;
        double time = strtod(ptr, &first_comma);

        char *data = first_comma + 1;
        for (uint64_t counter = 0; data < line_end && counter < max_n; ++counter) {
            char *aux = NULL;
            cs[0].items[counter].x = strtod(data, &aux);
            data = aux + 1;

            aux = NULL;
            cs[0].items[counter].y = strtod(data, &aux);
            data = aux + 1;

            aux = NULL;
            cs[0].items[counter].z = strtod(data, &aux);
            data = aux + 1;

            if (has_size) {
                aux = NULL;
                cs[0].items[counter].size = strtod(data, &aux);
                data = aux + 1;
            }
        }
        ptr = line_end + 1;

        fprintf(fout, "%.15e,", time);
        for (uint64_t i = 0; i < cs[0].len - 1; ++i) {
            fprintf(fout, "%.15e,%.15e,%.15e,", cs[0].items[i].x, cs[0].items[i].y, cs[0].items[i].z);
            if (has_size) {
                fprintf(fout, "%.15e,", cs[0].items[i].size);
            }
        }
        uint64_t i = cs[0].len - 1;
        if (has_size) {
            fprintf(fout, "%.15e,%.15e,%.15e,%.15e\n", cs[0].items[i].x, cs[0].items[i].y, cs[0].items[i].z, cs[0].items[i].size);
        } else {
            fprintf(fout, "%.15e,%.15e,%.15e\n", cs[0].items[i].x, cs[0].items[i].y, cs[0].items[i].z);
        }

    }

    while (ptr < buffer + len) {
        char *line_end = ptr;
        while (*line_end != '\n')
            line_end += 1;

        char *first_comma = NULL;
        double time = strtod(ptr, &first_comma);
        fprintf(fout, "%.15e,", time);

        char *data = first_comma + 1;
        for (uint64_t counter = 0; counter < max_n; ++counter) {
            if (data < line_end) {
                char *aux = NULL;
                cs[1].items[counter].x = strtod(data, &aux);
                data = aux + 1;

                aux = NULL;
                cs[1].items[counter].y = strtod(data, &aux);
                data = aux + 1;

                aux = NULL;
                cs[1].items[counter].z = strtod(data, &aux);
                data = aux + 1;
                
                if (has_size) {
                    aux = NULL;
                    cs[1].items[counter].size = strtod(data, &aux);
                    data = aux + 1;
                }
            }
            cs[2].items[counter] = cs[1].items[counter];
        }
        ptr = line_end + 1;

        for (uint64_t i = 0; i < cs[1].len; ++i) {
            double min_d2 = FLT_MAX;
            uint64_t min_idx = 0;
            for (uint64_t j = 0; j < cs[0].len; ++j) {
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        for (int dk = -1; dk <= 1; ++dk) {
                            double dx = (cs[1].items[i].x - cs[0].items[j].x - dj * sample_x) / sample_x;
                            double dy = (cs[1].items[i].y - cs[0].items[j].y - di * sample_y) / sample_y;
                            double dz = (cs[1].items[i].z - cs[0].items[j].z - dk * sample_z) / sample_z;
                            double ds = cs[1].items[i].size - cs[0].items[j].size;
                            double d2 = dx * dx + dy * dy + dz * dz;
                            min_idx = d2 < min_d2? j: min_idx;
                            min_d2 = d2 < min_d2? d2: min_d2;
                        }
                    }
                }
            }
            cs[2].items[min_idx] = cs[1].items[i];
            cs[0].items[min_idx] = (center){.x = -1, .y = -1, .z = -1};

            if (min_d2 >= d2_threshold) {
                cs[2].items[min_idx].x = -1;
                cs[2].items[min_idx].y = -1;
                cs[2].items[min_idx].z = -1;
                cs[2].items[min_idx].size = 0;
            }
        }

        {
            for (uint64_t i = 0; i < cs[2].len - 1; ++i) {
                cs[0].items[i] = cs[2].items[i];
                cs[1].items[i] = cs[2].items[i];
                fprintf(fout, "%.15e,%.15e,%.15e,", cs[0].items[i].x, cs[0].items[i].y, cs[0].items[i].z);
                if (has_size) {
                    fprintf(fout, "%.15e,", cs[0].items[i].size);
                }
            }
            uint64_t i = cs[2].len - 1;
            cs[0].items[i] = cs[2].items[i];
            cs[1].items[i] = cs[2].items[i];
            if (has_size) {
                fprintf(fout, "%.15e,%.15e,%.15e,%.15e\n", cs[0].items[i].x, cs[0].items[i].y, cs[0].items[i].z, cs[0].items[i].size);
            } else {
                fprintf(fout, "%.15e,%.15e,%.15e\n", cs[0].items[i].x, cs[0].items[i].y, cs[0].items[i].z);
            }
        }
    }

    mfclose(fout);
    mfree(buffer);
    mfree(cs[0].items);
    mfree(cs[1].items);
    mfree(cs[2].items);

    return true;
}

const char *str_fmt_tmp(const char *fmt, ...) {
    static char strs[MAX_STRS][MAX_STR_LEN] = {0};
    static uint64_t idx = 0;
    uint64_t ret_idx = idx;

    va_list arg_list;
    va_start(arg_list, fmt);
    if (!fmt) {
        logging_log(LOG_WARNING, "Format string provided is NULL");
        goto end;
    }

    if (vsnprintf(strs[idx], MAX_STR_LEN, fmt, arg_list) > MAX_STR_LEN)
        logging_log(LOG_WARNING, "String written with len greater than MAX_STR_LEN");

    va_end(arg_list);
    idx += 1;
    if (idx >= MAX_STR_LEN) {
        idx = 0;
        logging_log(LOG_WARNING, "Surpassed MAX_STRS limit. Strings are going to be overwritten, starting with %s", strs[0]);
    }
end:
    return strs[ret_idx];
}

double min_double(double a, double b) {
    return a < b? a: b;
}

double max_double(double a, double b) {
    return a > b? a: b;
}

bool barycentric_4pts(v3d p0, v3d p1, v3d p2, v3d p3, double x, double y, double z, double *b0, double *b1, double *b2, double *b3) {
    double l0 = (-(p1.x*(p2.y*p3.z-p3.y*p2.z))+p2.x*(p1.y*p3.z-p3.y*p1.z)-p3.x*(p1.y*p2.z-p2.y*p1.z))/(p0.x*(p1.y*(p2.z-p3.z)-p2.y*(p1.z-p3.z)+p3.y*(p1.z-p2.z))+p1.x*(-(p0.y*(p2.z-p3.z))+p2.y*(p0.z-p3.z)-p3.y*(p0.z-p2.z))+p2.x*(p0.y*(p1.z-p3.z)-p1.y*(p0.z-p3.z)+p3.y*(p0.z-p1.z))+p3.x*(-(p0.y*(p1.z-p2.z))+p1.y*(p0.z-p2.z)-p2.y*(p0.z-p1.z)))+(x*(p1.y*(p2.z-p3.z)-p2.y*(p1.z-p3.z)+p3.y*(p1.z-p2.z)))/(p0.x*(p1.y*(p2.z-p3.z)-p2.y*(p1.z-p3.z)+p3.y*(p1.z-p2.z))+p1.x*(-(p0.y*(p2.z-p3.z))+p2.y*(p0.z-p3.z)-p3.y*(p0.z-p2.z))+p2.x*(p0.y*(p1.z-p3.z)-p1.y*(p0.z-p3.z)+p3.y*(p0.z-p1.z))+p3.x*(-(p0.y*(p1.z-p2.z))+p1.y*(p0.z-p2.z)-p2.y*(p0.z-p1.z)))+(y*(-(p1.x*(p2.z-p3.z))+p2.x*(p1.z-p3.z)-p3.x*(p1.z-p2.z)))/(p0.x*(p1.y*(p2.z-p3.z)-p2.y*(p1.z-p3.z)+p3.y*(p1.z-p2.z))+p1.x*(-(p0.y*(p2.z-p3.z))+p2.y*(p0.z-p3.z)-p3.y*(p0.z-p2.z))+p2.x*(p0.y*(p1.z-p3.z)-p1.y*(p0.z-p3.z)+p3.y*(p0.z-p1.z))+p3.x*(-(p0.y*(p1.z-p2.z))+p1.y*(p0.z-p2.z)-p2.y*(p0.z-p1.z)))+((p1.x*(p2.y-p3.y)-p2.x*(p1.y-p3.y)+p3.x*(p1.y-p2.y))*z)/(p0.x*(p1.y*(p2.z-p3.z)-p2.y*(p1.z-p3.z)+p3.y*(p1.z-p2.z))+p1.x*(-(p0.y*(p2.z-p3.z))+p2.y*(p0.z-p3.z)-p3.y*(p0.z-p2.z))+p2.x*(p0.y*(p1.z-p3.z)-p1.y*(p0.z-p3.z)+p3.y*(p0.z-p1.z))+p3.x*(-(p0.y*(p1.z-p2.z))+p1.y*(p0.z-p2.z)-p2.y*(p0.z-p1.z)));
    
    double l1 = (p0.x*(p2.y*p3.z-p3.y*p2.z)-p2.x*(p0.y*p3.z-p3.y*p0.z)+p3.x*(p0.y*p2.z-p2.y*p0.z))/(p0.x*(p1.y*(p2.z-p3.z)-p2.y*(p1.z-p3.z)+p3.y*(p1.z-p2.z))+p1.x*(-(p0.y*(p2.z-p3.z))+p2.y*(p0.z-p3.z)-p3.y*(p0.z-p2.z))+p2.x*(p0.y*(p1.z-p3.z)-p1.y*(p0.z-p3.z)+p3.y*(p0.z-p1.z))+p3.x*(-(p0.y*(p1.z-p2.z))+p1.y*(p0.z-p2.z)-p2.y*(p0.z-p1.z)))+(x*(-(p0.y*(p2.z-p3.z))+p2.y*(p0.z-p3.z)-p3.y*(p0.z-p2.z)))/(p0.x*(p1.y*(p2.z-p3.z)-p2.y*(p1.z-p3.z)+p3.y*(p1.z-p2.z))+p1.x*(-(p0.y*(p2.z-p3.z))+p2.y*(p0.z-p3.z)-p3.y*(p0.z-p2.z))+p2.x*(p0.y*(p1.z-p3.z)-p1.y*(p0.z-p3.z)+p3.y*(p0.z-p1.z))+p3.x*(-(p0.y*(p1.z-p2.z))+p1.y*(p0.z-p2.z)-p2.y*(p0.z-p1.z)))+(y*(p0.x*(p2.z-p3.z)-p2.x*(p0.z-p3.z)+p3.x*(p0.z-p2.z)))/(p0.x*(p1.y*(p2.z-p3.z)-p2.y*(p1.z-p3.z)+p3.y*(p1.z-p2.z))+p1.x*(-(p0.y*(p2.z-p3.z))+p2.y*(p0.z-p3.z)-p3.y*(p0.z-p2.z))+p2.x*(p0.y*(p1.z-p3.z)-p1.y*(p0.z-p3.z)+p3.y*(p0.z-p1.z))+p3.x*(-(p0.y*(p1.z-p2.z))+p1.y*(p0.z-p2.z)-p2.y*(p0.z-p1.z)))+((-(p0.x*(p2.y-p3.y))+p2.x*(p0.y-p3.y)-p3.x*(p0.y-p2.y))*z)/(p0.x*(p1.y*(p2.z-p3.z)-p2.y*(p1.z-p3.z)+p3.y*(p1.z-p2.z))+p1.x*(-(p0.y*(p2.z-p3.z))+p2.y*(p0.z-p3.z)-p3.y*(p0.z-p2.z))+p2.x*(p0.y*(p1.z-p3.z)-p1.y*(p0.z-p3.z)+p3.y*(p0.z-p1.z))+p3.x*(-(p0.y*(p1.z-p2.z))+p1.y*(p0.z-p2.z)-p2.y*(p0.z-p1.z)));

    double l2 = (-(p0.x*(p1.y*p3.z-p3.y*p1.z))+p1.x*(p0.y*p3.z-p3.y*p0.z)-p3.x*(p0.y*p1.z-p1.y*p0.z))/(p0.x*(p1.y*(p2.z-p3.z)-p2.y*(p1.z-p3.z)+p3.y*(p1.z-p2.z))+p1.x*(-(p0.y*(p2.z-p3.z))+p2.y*(p0.z-p3.z)-p3.y*(p0.z-p2.z))+p2.x*(p0.y*(p1.z-p3.z)-p1.y*(p0.z-p3.z)+p3.y*(p0.z-p1.z))+p3.x*(-(p0.y*(p1.z-p2.z))+p1.y*(p0.z-p2.z)-p2.y*(p0.z-p1.z)))+(x*(p0.y*(p1.z-p3.z)-p1.y*(p0.z-p3.z)+p3.y*(p0.z-p1.z)))/(p0.x*(p1.y*(p2.z-p3.z)-p2.y*(p1.z-p3.z)+p3.y*(p1.z-p2.z))+p1.x*(-(p0.y*(p2.z-p3.z))+p2.y*(p0.z-p3.z)-p3.y*(p0.z-p2.z))+p2.x*(p0.y*(p1.z-p3.z)-p1.y*(p0.z-p3.z)+p3.y*(p0.z-p1.z))+p3.x*(-(p0.y*(p1.z-p2.z))+p1.y*(p0.z-p2.z)-p2.y*(p0.z-p1.z)))+(y*(-(p0.x*(p1.z-p3.z))+p1.x*(p0.z-p3.z)-p3.x*(p0.z-p1.z)))/(p0.x*(p1.y*(p2.z-p3.z)-p2.y*(p1.z-p3.z)+p3.y*(p1.z-p2.z))+p1.x*(-(p0.y*(p2.z-p3.z))+p2.y*(p0.z-p3.z)-p3.y*(p0.z-p2.z))+p2.x*(p0.y*(p1.z-p3.z)-p1.y*(p0.z-p3.z)+p3.y*(p0.z-p1.z))+p3.x*(-(p0.y*(p1.z-p2.z))+p1.y*(p0.z-p2.z)-p2.y*(p0.z-p1.z)))+((p0.x*(p1.y-p3.y)-p1.x*(p0.y-p3.y)+p3.x*(p0.y-p1.y))*z)/(p0.x*(p1.y*(p2.z-p3.z)-p2.y*(p1.z-p3.z)+p3.y*(p1.z-p2.z))+p1.x*(-(p0.y*(p2.z-p3.z))+p2.y*(p0.z-p3.z)-p3.y*(p0.z-p2.z))+p2.x*(p0.y*(p1.z-p3.z)-p1.y*(p0.z-p3.z)+p3.y*(p0.z-p1.z))+p3.x*(-(p0.y*(p1.z-p2.z))+p1.y*(p0.z-p2.z)-p2.y*(p0.z-p1.z)));

    double l3 = (p0.x*(p1.y*p2.z-p2.y*p1.z)-p1.x*(p0.y*p2.z-p2.y*p0.z)+p2.x*(p0.y*p1.z-p1.y*p0.z))/(p0.x*(p1.y*(p2.z-p3.z)-p2.y*(p1.z-p3.z)+p3.y*(p1.z-p2.z))+p1.x*(-(p0.y*(p2.z-p3.z))+p2.y*(p0.z-p3.z)-p3.y*(p0.z-p2.z))+p2.x*(p0.y*(p1.z-p3.z)-p1.y*(p0.z-p3.z)+p3.y*(p0.z-p1.z))+p3.x*(-(p0.y*(p1.z-p2.z))+p1.y*(p0.z-p2.z)-p2.y*(p0.z-p1.z)))+(x*(-(p0.y*(p1.z-p2.z))+p1.y*(p0.z-p2.z)-p2.y*(p0.z-p1.z)))/(p0.x*(p1.y*(p2.z-p3.z)-p2.y*(p1.z-p3.z)+p3.y*(p1.z-p2.z))+p1.x*(-(p0.y*(p2.z-p3.z))+p2.y*(p0.z-p3.z)-p3.y*(p0.z-p2.z))+p2.x*(p0.y*(p1.z-p3.z)-p1.y*(p0.z-p3.z)+p3.y*(p0.z-p1.z))+p3.x*(-(p0.y*(p1.z-p2.z))+p1.y*(p0.z-p2.z)-p2.y*(p0.z-p1.z)))+(y*(p0.x*(p1.z-p2.z)-p1.x*(p0.z-p2.z)+p2.x*(p0.z-p1.z)))/(p0.x*(p1.y*(p2.z-p3.z)-p2.y*(p1.z-p3.z)+p3.y*(p1.z-p2.z))+p1.x*(-(p0.y*(p2.z-p3.z))+p2.y*(p0.z-p3.z)-p3.y*(p0.z-p2.z))+p2.x*(p0.y*(p1.z-p3.z)-p1.y*(p0.z-p3.z)+p3.y*(p0.z-p1.z))+p3.x*(-(p0.y*(p1.z-p2.z))+p1.y*(p0.z-p2.z)-p2.y*(p0.z-p1.z)))+((-(p0.x*(p1.y-p2.y))+p1.x*(p0.y-p2.y)-p2.x*(p0.y-p1.y))*z)/(p0.x*(p1.y*(p2.z-p3.z)-p2.y*(p1.z-p3.z)+p3.y*(p1.z-p2.z))+p1.x*(-(p0.y*(p2.z-p3.z))+p2.y*(p0.z-p3.z)-p3.y*(p0.z-p2.z))+p2.x*(p0.y*(p1.z-p3.z)-p1.y*(p0.z-p3.z)+p3.y*(p0.z-p1.z))+p3.x*(-(p0.y*(p1.z-p2.z))+p1.y*(p0.z-p2.z)-p2.y*(p0.z-p1.z)));
    if (b0 != NULL)
	*b0 = l0;
    if (b1 != NULL)
	*b1 = l1;
    if (b2 != NULL)
	*b2 = l2;
    if (b3 != NULL)
	*b3 = l3;
    return l0 >= 0 && l1 >= 0 && l2 >= 0 && l3 >= 0;	
}

bool barycentric_8pts(v3d p0, v3d p1, v3d p2, v3d p3, v3d p4, v3d p5, v3d p6, v3d p7, double x, double y, double z, double *b0, double *b1, double *b2, double *b3, double *b4, double *b5, double *b6, double *b7) {
    bool part1 = barycentric_4pts(p0, p1, p2, p3, x, y, z, b0, b1, b2, b3);
    bool part2 = barycentric_4pts(p4, p5, p6, p7, x, y, z, b4, b5, b6, b7);
    return part1 || part2;
}
