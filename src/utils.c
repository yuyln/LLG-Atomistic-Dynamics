#include "utils.h"

typedef struct {
    double x;
    double y;
} center;

typedef struct {
    center *items;
    uint64_t len;
    uint64_t cap;
} centers;

bool organize_clusters_inplace(const char *in_path) {
}

bool organize_clusters(const char *in_path, const char *out_path) {
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
    if (max_n % 2) {
        logging_log(LOG_ERROR, "Number of commnas not a multiple of two. \"%s\" is probably corrupted", in_path);
        return false;
    }

    max_n >>= 1;

    centers cs[2] = {0};

    for (uint64_t i = 0; i < max_n; ++i) {
        da_append(&cs[0], ((center){.x = -1, .y = -1}));
        da_append(&cs[1], ((center){.x = -1, .y = -1}));
    }

    char *ptr = buffer;
    while (*ptr) {
        double time[2] = {0};
        for (int i = 0; i < 2; ++i) {
            char *line_end = ptr;
            while (*line_end != '\n')
                line_end += 1;

            char *first_comma = NULL;
            time[i] = strtod(ptr, &first_comma);

            char *data = first_comma + 1;
            uint64_t aux = 0;
            for (uint64_t counter = 0; data < line_end && counter < max_n; ++counter) {
                char *aux = NULL;
                cs[i].items[counter].x = strtod(data, &aux);
                data = aux + 1;

                aux = NULL;
                cs[i].items[counter].y = strtod(data, &aux);
                data = aux + 1;
            }
            ptr = line_end + 1;
        }
        logging_log(LOG_INFO, "%e, %e, %e", time[0], cs[0].items[1].x, cs[0].items[1].y);
        logging_log(LOG_INFO, "%e, %e, %e", time[1], cs[1].items[1].x, cs[1].items[1].y);
    }

    return true;
}
