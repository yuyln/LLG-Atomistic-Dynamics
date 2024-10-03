#define __ALLOCATOR_C
#include "allocator.h"
#include "./src/logging.c"

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
    FILE *f = mfopen(path, "rb");
    massert(f);

    fseek(f, 0, SEEK_END);
    uint64_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *tmp = mmalloc(size + 1);
    char *tmp2 = mmalloc(size + 1);
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
    char *ret = mmalloc(strlen(tmp2) + 1);
    memcpy(ret, tmp2, strlen(tmp2));

    mfree(tmp);
    mfree(tmp2);
    mfclose(f);
    return ret;
}

int main(int argc, const char **argv) {
    const char *program = shift_args(&argc, &argv);
    if (argc == 0) {
        fprintf(stderr, "Usage %s file1 file2 file3 ...\n", program);
        return 1;
    }

    const char *path = "./src/complete_kernel.c";
    FILE *f = mfopen(path, "wb");
    massert(f);
    fprintf(f, "#include \"complete_kernel.h\"\n\nconst char *complete_kernel = \"//I hate clover\\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\" ");
    while(argc > 0) {
        char *data = read_file(shift_args(&argc, &argv));
        char *ptr = data;
        while (*ptr)
            fprintf(f, " \"\\x%02x\" ", *ptr++);
        mfree(data);
    }
    fprintf(f, ";");
    mfclose(f);

    return 0;
}
