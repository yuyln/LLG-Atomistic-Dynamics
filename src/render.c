#include "render.h"

#include <X11/Xlib.h>
#define XK_LATIN1
#include <X11/keysymdef.h>
#include <X11/extensions/Xdbe.h>

struct simulation_window {
    unsigned int width;
    unsigned int height;
    RGBA32 *buffer;
    Display *display;
    Window window;
    XdbeBackBuffer back_buffer;
    XWindowAttributes wa;
    XImage *image;
    GC gc;
    Atom wm_delete_window;
};

simulation_window render_init(unsigned int width, unsigned int height) {
    simulation_window ret = {.width = width, .height = height};
    ret.buffer = calloc(width * height, sizeof(*ret.buffer));

    ret.display = XOpenDisplay(NULL);
    if (ret.display == NULL) {
        fprintf(stderr, "ERROR: could not open the default display\n");
        exit(1);
    }

    int major_version_return, minor_version_return;
    if(XdbeQueryExtension(ret.display, &major_version_return, &minor_version_return)) {
        printf("XDBE version %d.%d\n", major_version_return, minor_version_return);
    } else {
        fprintf(stderr, "XDBE is not supported!!!1\n");
        exit(1);
    }

    ret.window = XCreateSimpleWindow(
            ret.display,
            XDefaultRootWindow(ret.display),
            0, 0,
            width, height,
            0,
            0,
            0);

    ret.back_buffer = XdbeAllocateBackBufferName(ret.display, ret.window, 0);
    printf("back_buffer ID: %lu\n", ret.back_buffer);

    ret.wa = {0};
    XGetWindowAttributes(ret.display, ret.window, &ret.wa);

    ret.image = XCreateImage(ret.display,
            ret.wa.visual,
            ret.wa.depth,
            ZPixmap,
            0,
            (char*) ret.buffer,
            width,
            height,
            sizeof(RGBA32) * 8,
            width * sizeof(RGBA32));

    ret.gc = XCreateGC(ret.display, ret.window, 0, NULL);

    ret.wm_delete_window = XInternAtom(ret.display, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(ret.display, ret.window, &ret.wm_delete_window, 1);

    XSelectInput(ret.display, ret.window, KeyPressMask);

    XMapWindow(ret.display, ret.window);

    return ret;
}
