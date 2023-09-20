#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#define OPENCLWRAPPER_IMPLEMENTATION
#include <opencl_wrapper.h>

static const char kernel_name[] = "kernel.c";

int main()
{
    FILE *kernel = fopen(kernel_name, "rb");
    if (!kernel)
    {
        fprintf(stderr, "Could not open file %s: %s\n", kernel_name, strerror(errno));
        return 1;
    }

    char *kernel_data;
    fseek(kernel, 0, SEEK_SET);
    fseek(kernel, 0, SEEK_END);
    long kernel_size = ftell(kernel);

    fseek(kernel, 0, SEEK_SET);
    kernel_data = (char*)malloc(kernel_size + 1);
    fread(kernel_data, 1, kernel_size, kernel);
    kernel_data[kernel_size] = '\0';
    fclose(kernel);

    kernel = fopen("./headers/opencl_kernel.h", "w");
    if (!kernel)
    {
        fprintf(stderr, "Could not open file %s: %s\n", "opencl_kernel.h", strerror(errno));
        return 1;
    }
    
    fprintf(kernel, "#ifndef __OPEN_CL_KERNEL\n");
    fprintf(kernel, "#define __OPEN_CL_KERNEL\n");
    fprintf(kernel, "/*static*/ const char kernel_data[] = \"\\\n");

    char *ptr = kernel_data;
    for (; *ptr; ++ptr)
    {
        if (*ptr == '\n')
        {
            fprintf(kernel, "\\n\\\n");
            continue;
        }

        if (*ptr == '\r')
            continue;

        fputc(*ptr, kernel);
    }
    
    fprintf(kernel, "\";");
    fprintf(kernel, "\n#endif");

    if (kernel_data)
        free(kernel_data);

    fclose(kernel);
    return 0;
}
