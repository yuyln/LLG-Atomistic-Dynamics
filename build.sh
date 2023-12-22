#!/bin/sh
set -xe
CFLAGS="-Wall -Wextra -pedantic -O0 -ggdb -I ./include -DCL_TARGET_OPENCL_VERSION=300 -DCL_USE_DEPRECATED_OPENCL_1_2_APIS"
LIBS="-lm `pkg-config --static --libs OpenCL`"
FILES="`find ./src -type f -name "*.c"`"


gcc $CFLAGS $FILES create_kernel.c -o create_kernel $LIBS
./create_kernel

gcc $CFLAGS $FILES main.c -o main $LIBS

