#!/bin/sh
set -xe

CFLAGS="-DWIN -DRK4 -Wall -Wextra -pedantic -Ofast -ggdb -I ./headers -I ./OpenCL/include -L ./OpenCL/lib -static-libgcc"
LIBS="-lm -lOpenCL -fopenmp"
CC=x86_64-w64-mingw32-gcc

#$CC $CFLAGS prepare.c -o prepare $LIBS
#WINEPREFIX=~/WINE64 wine ./prepare.exe
#
#$CC $CFLAGS main.c -o main $LIBS

$CC $CFLAGS analyze.c -o analyze $LIBS

