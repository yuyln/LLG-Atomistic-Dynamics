#!/bin/sh
set -xe
CFLAGS="-DnPROFILING -Wall -Wextra -pedantic -O0 -ggdb -I ./include -DCL_TARGET_OPENCL_VERSION=300 -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -Wno-overlength-strings"
LIBS="-lm `pkg-config --static --libs OpenCL x11 xext`"
FILES="`find ./src -type f -name "*.c"`"


gcc $CFLAGS create_kernel.c -o create_kernel
./create_kernel ./tyche_i.cl ./include/constants.h ./include/v3d.h ./include/grid_types.h ./include/simulation_funcs.h ./src/v3d.c ./src/simulation_funcs.c kernel.c
rm ./create_kernel

gcc $CFLAGS -c $FILES $LIBS

FILES_OBJ="`find -type f -name "*.o"`"
ar cr libatomistic.a $FILES_OBJ

rm *.o

gcc -L./ $CFLAGS main.c -o main -latomistic $LIBS
