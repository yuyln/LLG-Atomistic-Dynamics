#!/bin/sh
set -xe
CFLAGS="-Wall -Wextra -pedantic -O0 -ggdb -I ./include -DCL_TARGET_OPENCL_VERSION=300 -DCL_USE_DEPRECATED_OPENCL_1_2_APIS"
LIBS="-lm `pkg-config --static --libs OpenCL`"
FILES="`find ./src -type f -name "*.c"`"


gcc $CFLAGS $FILES create_kernel.c -o create_kernel $LIBS
./create_kernel ./include/constants.h ./include/v3d.h ./include/grid_types.h ./include/simulation_funcs.h ./src/v3d.c ./src/simulation_funcs.c kernel.cl

gcc $CFLAGS $FILES main.c -o main $LIBS

