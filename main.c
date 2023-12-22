#include <stdio.h>
#include "gpu.h"

//@TODO: Change openclwrapper to print file and location correctly
int main(void) {
    gpu_cl gpu = gpu_cl_init(0, 0);
    gpu_cl_close(&gpu);
    return 0;
}
