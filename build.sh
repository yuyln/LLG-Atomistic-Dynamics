#!/bin/sh
set -xe
CFLAGS="-Wall -Wextra -pedantic -O0 -ggdb -I ./include"
LIBS="-lm `pkg-config --static --libs OpenCL`"
FILES="main.c `find ./src -type f -name "*.c"`"


gcc $CFLAGS main.c -o main $LIBS
