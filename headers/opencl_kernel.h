#ifndef __OPEN_CL_KERNEL
#define __OPEN_CL_KERNEL
static const char kernel_data[] = "\
kernel void Add()\n\
{\n\
    int i = get_global_id(0);\n\
    a + b;\n\
}";
#endif