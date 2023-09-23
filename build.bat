@REM call vcvars64.bat

cl /O2 /D RK4 /D _USE_MATH_DEFINES /D WIN /MD /I./OpenCL/include /o prepare prepare.c ./OpenCL/lib/OpenCL.lib user32.lib gdi32.lib shell32.lib /openmp /link /NODEFAULTLIB:library
.\prepare.exe

cl /O2 /D RK4 /D _USE_MATH_DEFINES /D WIN /MD /I./OpenCL/include /o main main.c ./OpenCL/lib/OpenCL.lib user32.lib gdi32.lib shell32.lib /openmp /link /NODEFAULTLIB:library

cl /O2 /D _USE_MATH_DEFINES /D WIN /MD /I./OpenCL/include /o analyze analyze.c ./OpenCL/lib/OpenCL.lib user32.lib gdi32.lib shell32.lib /openmp /link /NODEFAULTLIB:library

del *.obj

