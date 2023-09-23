#!/bin/sh
set -xe

CFLAGS="-DRK4 -Wall -Wextra -pedantic -O3 -ggdb"
LIBS="-lm `pkg-config --static --libs OpenCL` -fopenmp"

gcc $CFLAGS prepare.c -o prepare $LIBS
./prepare

gcc $CFLAGS main.c -o main $LIBS

gcc $CFLAGS analyze.c -o analyze $LIBS

