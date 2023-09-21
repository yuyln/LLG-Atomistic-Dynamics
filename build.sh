##!/bin/sh
#
#set -xe
#
#gcc -Wall -Wextra -O0 -pedantic -o prepare prepare.c
#./prepare
#
#gcc -DRK4  -I ./headers -O3 -Wall -Wextra -pedantic -ggdb -o main main.c -lm -lOpenCL -fopenmp
#./main

#gcc -DRK4 -L ./OpenCL/lib -I ./OpenCL/include -I ./headers -O3 -Wall -Wextra -pedantic -ggdb -o analyze analyze.c -lm -lOpenCL -fopenmp
#./analyze
#
#

#!/bin/sh
set -xe

CFLAGS="-DRK4 -Wall -Wextra -pedantic -O3 -ggdb -I ./headers"
LIBS="-lm `pkg-config --static --libs OpenCL` -fopenmp"

gcc $CFLAGS prepare.c -o prepare $LIBS
./prepare

gcc $CFLAGS main.c -o main $LIBS

gcc $CFLAGS analyze.c -o analyze $LIBS

