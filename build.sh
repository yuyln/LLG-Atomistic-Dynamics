#!/bin/sh
set -xe
CFLAGS="-Wall -Wextra -pedantic -O0 -ggdb -I ./include -DCL_TARGET_OPENCL_VERSION=300 -DCL_USE_DEPRECATED_OPENCL_1_2_APIS"
LIBS="-lm `pkg-config --static --libs OpenCL`"
FILES="main.c `find ./src -type f -name "*.c"`"


gcc $CFLAGS $FILES -o main $LIBS
