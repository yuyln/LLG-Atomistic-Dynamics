#!/bin/sh

set -xe

COMMON_CFLAGS="-DnPROFILING -O3 -I ./include -DCL_TARGET_OPENCL_VERSION=300 -DCL_USE_DEPRECATED_OPENCL_1_2_APIS"
FILES="`find ./src -maxdepth 1 -type f -name "*.c"` ./src/platform_specific/render_linux_x11.c"
CC="gcc"
LIBS="-lm `pkg-config --static --libs OpenCL x11`"

if [ "`pkg-config --libs xext`" > /dev/null ]; then
    LIBS="$LIBS `pkg-config --static --libs xext`"
    COMMON_CFLAGS="$COMMON_CFLAGS -DUSE_XEXT"
fi

if test -f ./libatomistic.a; then
    rm ./libatomistic.a
fi

if [ "$1" = "install" ]; then
    if test -f $HOME/.local/lib/atomistic; then
        rm -r $HOME/.local/lib/atomistic
    fi
    mkdir --parents $HOME/.local/lib/atomistic
    mkdir --parents $HOME/.local/bin
    CFLAGS="$COMMON_CFLAGS"

    $CC $CFLAGS create_kernel.c -o create_kernel
    ./create_kernel ./kernel_files/tyche_i.c ./include/constants.h ./include/v3d.h ./include/grid_types.h ./kernel_files/random.h ./kernel_files/simulation_funcs.h ./include/colors.h ./src/v3d.c ./kernel_files/random.c ./kernel_files/simulation_funcs.c ./src/colors.c ./kernel_files/kernel.c
    rm ./create_kernel

    $CC -fPIC $CFLAGS -c $FILES $LIBS

    FILES_OBJ="`find -type f -name "*.o"`"
    ar cr libatomistic.a $FILES_OBJ

    rm *.o
    cp ./libatomistic.a $HOME/.local/lib/atomistic/
    cp -r ./include $HOME/.local/lib/atomistic/include
    cp ./run_atomistic $HOME/.local/bin/run_atomistic
else
    set -xe
    CFLAGS="$COMMON_CFLAGS -Wall -Wextra -pedantic -ggdb -g3 -Wno-overlength-strings -Wno-override-init"

    $CC $CFLAGS create_kernel.c -o create_kernel
    ./create_kernel ./kernel_files/tyche_i.c ./include/constants.h ./include/v3d.h ./include/grid_types.h ./kernel_files/random.h ./kernel_files/simulation_funcs.h ./include/colors.h ./src/v3d.c ./kernel_files/random.c ./kernel_files/simulation_funcs.c ./src/colors.c ./kernel_files/kernel.c
    rm ./create_kernel

    $CC $CFLAGS -c $FILES $LIBS

    FILES_OBJ="`find -type f -name "*.o"`"
    ar cr libatomistic.a $FILES_OBJ

    rm *.o

    $CC -L./ $CFLAGS main.c -o main -latomistic $LIBS
fi
