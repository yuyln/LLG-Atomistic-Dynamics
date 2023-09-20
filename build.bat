@REM gcc -DRK4 -L ./OpenCL/lib -I ./OpenCL/include -I ./headers -Wall -Wextra -O0 -ggdb -pedantic -o prepare prepare.c -lOpenCL -fopenmp
@REM .\prepare.exe
 
@REM gcc -DRK4 -L ./OpenCL/lib -I ./OpenCL/include -I ./headers -O3 -Wall -Wextra -pedantic -ggdb -o main main.c -lm -lOpenCL -fopenmp
@REM .\main.exe

@REM gcc -DRK4 -L ./OpenCL/lib -I ./OpenCL/include -I ./headers -O3 -Wall -Wextra -pedantic -ggdb -o analyze analyze.c -lm -lOpenCL -fopenmp
@REM .\analyze


@REM call vcvars64.bat

cl /O2 /D RK4 /D _USE_MATH_DEFINES /D WIN /MD /I./headers /I./OpenCL/include /o prepare prepare.c ./OpenCL/lib/OpenCL.lib user32.lib gdi32.lib shell32.lib /link /NODEFAULTLIB:library
.\prepare.exe

cl /O2 /D RK4 /D _USE_MATH_DEFINES /D WIN /MD /I./headers /I./OpenCL/include /o main main.c ./OpenCL/lib/OpenCL.lib user32.lib gdi32.lib shell32.lib /link /NODEFAULTLIB:library

@REM cl /O2 /D _USE_MATH_DEFINES /D WIN /MD /I./headers /I./OpenCL/include /o analyze analyze.c ./OpenCL/lib/OpenCL.lib user32.lib gdi32.lib shell32.lib /link /NODEFAULTLIB:library

del *.obj

