#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

char *shift_args(int *argc, const char ***argv) {
    if (*argc <= 0)
        return NULL;
    *argc -= 1;
    char *ret = (char*)**argv;
    *argv += 1;
    return ret;
}

char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[ WARNING ] Could not read file %s: %s\n", path, strerror(errno));
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    uint64_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *tmp = calloc(size + 1, 1);
    char *tmp2 = calloc(size + 1, 1);
    fread(tmp, size, 1, f);

    char *ptr = tmp;
    char *ptr2 = tmp2;

    char include[] = "#include";
    while (*ptr) {
        if (strncmp(ptr, include, sizeof(include) - 1) == 0) {
            while (*ptr != '\n' && *ptr)
                ptr++;
        }
        *ptr2++ = *ptr;
        ptr++;
    }
    char *ret = calloc(strlen(tmp2) + 1, 1);
    memcpy(ret, tmp2, strlen(tmp2));

    free(tmp);
    free(tmp2);
    fclose(f);
    return ret;
}

int main(int argc, const char **argv) {
    const char *program = shift_args(&argc, &argv);
    if (argc == 0) {
        fprintf(stderr, "Usage %s file1 file2 file3 ...\n", program);
        return 1;
    }

    FILE *f = fopen("./src/complete_kernel.c", "wb");
    fprintf(f, "#include \"complete_kernel.h\"\n\nconst char *complete_kernel = ");
    while(argc > 0) {
        char *data = read_file(shift_args(&argc, &argv));
        char *ptr = data;
        while (*ptr)
            fprintf(f, " \"\\x%02x\" ", *ptr++);
        free(data);
    }
    fprintf(f, ";");
    fclose(f);

    return 0;
}
