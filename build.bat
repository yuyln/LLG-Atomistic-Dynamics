@REM call vcvars64.bat

set CFLAGS=/I ./include /I ./OpenCL/include /D nPROFILING /W0 /Od /D CL_TARGET_OPENCL_VERSION=300 /D CL_USE_DEPRECATED_OPENCL_1_2_APIS /Zi

set FILES=.\src\*.c .\src\platform_specific\render_windows.c

set LIBS=/link .\OpenCL\lib\OpenCL.lib kernel32.lib user32.lib gdi32.lib

cl %CFLAGS% create_kernel.c /o create_kernel

create_kernel.exe .\kernel_files\tyche_i.c .\include\constants.h .\include\v3d.h .\include\grid_types.h .\kernel_files\random.h .\kernel_files\simulation_funcs.h .\include\colors.h .\src\v3d.c .\kernel_files\random.c .\kernel_files\simulation_funcs.c .\src\colors.c .\kernel_files\kernel.c

del .\create_kernel.exe

cl %CFLAGS% %FILES% /c %LIBS%
lib .\*.obj /OUT:libatomistic.lib

del .\*.obj

cl %CFLAGS% main_windows.c /o main %LIBS% .\libatomistic.lib
