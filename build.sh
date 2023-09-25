#!/bin/sh
set -xe

CFLAGS="-DRK4 -DCL_TARGET_OPENCL_VERSION=300 -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -Wall -Wextra -pedantic -O3 -ggdb"
LIBS="-lm `pkg-config --static --libs OpenCL` -fopenmp"

gcc $CFLAGS prepare.c -o prepare $LIBS
./prepare

gcc $CFLAGS main.c -o main $LIBS

gcc $CFLAGS analyze.c -o analyze $LIBS

