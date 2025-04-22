#include "utils.h"

#include <float.h>
#include <math.h>
#include <assert.h>

typedef struct {
    double x;
    double y;
    double size;
    double vx; //TODO
    double vy; //TODO
    double _min_distance;
} center;

typedef struct {
    center *items;
    uint64_t len;
    uint64_t cap;
} centers;

bool organize_clusters_inplace(const char *in_path, double sample_x, double sample_y, double d2_threshold) {
    return organize_clusters(in_path, in_path, sample_x, sample_y, d2_threshold);
}

bool organize_clusters(const char *in_path, const char *out_path, double sample_x, double sample_y, double d2_threshold) {
    FILE *f_in = mfopen(in_path, "rb");
    if (!f_in)
        return false;
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
    bool has_size = max_n % 3 == 0;
    uint64_t div_fac = has_size? 3: 2;

    if (max_n % div_fac) {
        logging_log(LOG_ERROR, "Number of commas is not a multiple of %llu. \"%s\" is probably corrupted", div_fac, in_path);
        return false;
    }

    max_n /= div_fac;

    centers cs[3] = {0};

    for (uint64_t i = 0; i < max_n; ++i) {
        da_append(&cs[0], ((center){.x = -1, .y = -1}));
        da_append(&cs[1], ((center){.x = -1, .y = -1}));
        da_append(&cs[2], ((center){.x = -1, .y = -1}));
    }

    char *ptr = buffer;

    FILE *fout = mfopen(out_path, "wb");
    if (!fout)
        return false;
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

            if (has_size) {
                aux = NULL;
                cs[0].items[counter].size = strtod(data, &aux);
                data = aux + 1;
            }
        }
        ptr = line_end + 1;

        fprintf(fout, "%.15e,", time);
        for (uint64_t i = 0; i < cs[0].len - 1; ++i) {
            fprintf(fout, "%.15e,%.15e,", cs[0].items[i].x, cs[0].items[i].y);
            if (has_size) {
                fprintf(fout, "%.15e,", cs[0].items[i].size);
            }
        }
        uint64_t i = cs[0].len - 1;
        if (has_size) {
            fprintf(fout, "%.15e,%.15e,%.15e\n", cs[0].items[i].x, cs[0].items[i].y, cs[0].items[i].size);
        } else {
            fprintf(fout, "%.15e,%.15e\n", cs[0].items[i].x, cs[0].items[i].y);
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
        cs[1].len = 0;
        cs[2].len = 0;
        for (uint64_t counter = 0; counter < max_n; ++counter) {
            da_append(&cs[2], ((center){.x = -1, .y = -1, ._min_distance = FLT_MAX}));
            if (data < line_end) {
                da_append(&cs[1], ((center){0}));
                char *aux = NULL;
                center *it = &cs[1].items[cs[1].len - 1];
                it->x = strtod(data, &aux);
                data = aux + 1;

                aux = NULL;
                it->y = strtod(data, &aux);
                data = aux + 1;

                if (has_size) {
                    aux = NULL;
                    it->size = strtod(data, &aux);
                    data = aux + 1;
                }
            } else {
                da_append(&cs[1], ((center){.x = -1, .y = -1, ._min_distance = FLT_MAX}));
            }
        }
        ptr = line_end + 1;
        assert(cs[1].len == max_n);
        assert(cs[2].len == max_n);

        for (uint64_t i = 0; i < cs[1].len; ++i) {
            double min_d2 = FLT_MAX;
            uint64_t min_idx = 0;
            for (uint64_t j = 0; j < cs[0].len; ++j) {
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        double dx = (cs[1].items[i].x - cs[0].items[j].x - dj * sample_x) / sample_x;
                        double dy = (cs[1].items[i].y - cs[0].items[j].y - di * sample_y) / sample_y;
                        double ds = cs[1].items[i].size - cs[0].items[j].size;
                        double d2 = dx * dx + dy * dy + ds * ds;
                        min_idx = d2 < min_d2? j: min_idx;
                        min_d2 = d2 < min_d2? d2: min_d2;
                    }
                }
            }
            if (min_d2 < cs[2].items[min_idx]._min_distance) {
                cs[2].items[min_idx] = cs[1].items[i];
                cs[2].items[min_idx]._min_distance = min_d2;
            }
            cs[0].items[min_idx] = (center){.x = -1, .y = -1};

            //if (min_d2 >= d2_threshold) {
            //    cs[2].items[min_idx].x = -1;
            //    cs[2].items[min_idx].y = -1;
            //    cs[2].items[min_idx].size = 0;
            //    cs[2].items[min_idx]._min_distance = FLT_MAX;
            //}
        }

        {
            for (uint64_t i = 0; i < cs[2].len; ++i) {
                cs[0].items[i] = cs[2].items[i];
                cs[1].items[i] = cs[2].items[i];

                fprintf(fout, "%.15e,%.15e", cs[0].items[i].x, cs[0].items[i].y);
                if (has_size) {
                    fprintf(fout, ",%.15e", cs[0].items[i].size);
                }
                if ((i + 1) == cs[2].len)
                    fprintf(fout, "\n");
                else
                    fprintf(fout, ",");
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
    if (idx >= MAX_STRS) {
        idx = 0;
        logging_log(LOG_WARNING, "Surpassed MAX_STRS limit. Strings are going to be overwritten, starting with %s", strs[0]);
    }
end:
    return strs[ret_idx];
}

double shit_random(double from, double to) {
    double r = (double)rand() / (double)RAND_MAX;
    return from + r * (to - from);
}

uint64_t xorshift64_u64(xorshift64_state *state) {
	uint64_t x = *state;
	x ^= x << 13;
	x ^= x >> 7;
	x ^= x << 17;
	return *state = x;
}

double xorshift64_double(xorshift64_state *state) {
    uint64_t rng = xorshift64_u64(state);
    return rng / (double)UINT64_MAX;
}

double xorshift64_range(xorshift64_state *state, double from, double to) {
    double r = xorshift64_double(state);
    return from + r * (to - from);
}

double xorshift64_normal_distribution(xorshift64_state *state) {
    for (;;) {
        double U = xorshift64_double(state);
        double V = xorshift64_double(state);
        double X = sqrt(8.0 / M_E) * (V - 0.5) / U;
        double X2 = X * X;
        if (X2 <= (5.0 - 4.0 * exp(0.25) * U))
            return X;
        else if (X2 >= (4.0 * exp(-1.35) / U + 1.4))
            continue;
        else if (X2 <= (-4.0 * log(U)))
            return X;
    }
}
