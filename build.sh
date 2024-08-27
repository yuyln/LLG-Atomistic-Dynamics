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

if test -f ./libatomistic3d.a; then
    rm ./libatomistic3d.a
fi

if [ "$1" = "install" ]; then
    if test -d $HOME/.local/lib/atomistic3d; then
        rm -r $HOME/.local/lib/atomistic3d
    fi
    mkdir --parents $HOME/.local/lib/atomistic3d
    mkdir --parents $HOME/.local/bin
    CFLAGS="$COMMON_CFLAGS"

    $CC $CFLAGS create_kernel.c -o create_kernel
    ./create_kernel ./kernel_files/tyche_i.c ./include/constants.h ./include/v3d.h ./include/grid_types.h ./kernel_files/random.h ./kernel_files/simulation_funcs.h ./include/colors.h ./src/v3d.c ./kernel_files/random.c ./kernel_files/simulation_funcs.c ./src/colors.c ./kernel_files/kernel.c
    rm ./create_kernel

    $CC -fPIC $CFLAGS -c $FILES $LIBS

    FILES_OBJ="`find -type f -name "*.o"`"
    ar cr libatomistic3d.a $FILES_OBJ

    rm *.o
    cp ./libatomistic3d.a $HOME/.local/lib/atomistic3d/
    cp -r ./include $HOME/.local/lib/atomistic3d/include
    cp ./run_atomistic $HOME/.local/bin/run_atomistic3d

    $CC -fPIC $CFLAGS -shared -o libatomistic3d.so $FILES $LIBS
    cp ./libatomistic3d.so $HOME/.local/lib/atomistic3d/
else
    set -xe
    CFLAGS="$COMMON_CFLAGS -Wall -Wextra -pedantic -ggdb -g3 -Wno-overlength-strings -Wno-override-init"

    $CC $CFLAGS create_kernel.c -o create_kernel
    ./create_kernel ./kernel_files/tyche_i.c ./include/constants.h ./include/v3d.h ./include/grid_types.h ./kernel_files/random.h ./kernel_files/simulation_funcs.h ./include/colors.h ./src/v3d.c ./kernel_files/random.c ./kernel_files/simulation_funcs.c ./src/colors.c ./kernel_files/kernel.c
    rm ./create_kernel

    $CC $CFLAGS -c $FILES $LIBS

    FILES_OBJ="`find -type f -name "*.o"`"
    ar cr libatomistic3d.a $FILES_OBJ

    rm *.o

    $CC -L./ $CFLAGS main.c -o main -l:libatomistic3d.a $LIBS
fi
