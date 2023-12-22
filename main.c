#include "gpu.h"
#include "string_view.h"

//@TODO: Change openclwrapper to print file and location correctly
//@TODO: Check uint64_t->int changes
//@TODO: Optimize for 3D->2D
int main(void) {
    gpu_cl gpu = gpu_cl_init(0, 0);
    char *kernel;
    clw_read_file("./kernel_complete.cl", &kernel);
    const char cmp[] = "-DOPENCL_COMPILATION -DNBULK";
    string_view kernel_view = (string_view){.str = kernel, .len = strlen(kernel)};
    string_view compile_opt = (string_view){.str = cmp, .len = strlen(cmp) + 1};
    gpu_cl_compile_source(&gpu, kernel_view, compile_opt);
    free(kernel);
    gpu_cl_close(&gpu);
    return 0;
}
