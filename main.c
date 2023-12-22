#include <stdio.h>
#include "string_view.h"
#include "grid_funcs.h"
#define OPENCLWRAPPER_IMPLEMENTATION
#include "openclwrapper.h"

//@TODO: Change openclwrapper to print file and location correctly
int main(void) {
    grid g = grid_init((matrix_size){.dim = {10, 10, 1}});
    return 0;
}
