#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <string.h>
#define XK_LATIN1
#include <X11/keysymdef.h>
#include <X11/extensions/Xdbe.h>
#include <stdlib.h>
#include <stdio.h>

#include "render.h"

struct render_window {
    Display *display;
    Window window;
    XdbeBackBuffer back_buffer;
    XWindowAttributes wa;
    XImage *image;
    GC gc;
    Atom wm_delete_window;

    unsigned int width;
    unsigned int height;
    RGBA32 *buffer;
    bool should_close;

    window_input input;
};

render_window *window_init(unsigned int width, unsigned int height) {
    render_window ret = {.width = width, .height = height};
    ret.buffer = calloc(width * height, sizeof(*ret.buffer));

    ret.display = XOpenDisplay(NULL);
    if (ret.display == NULL) {
        fprintf(stderr, "[ FATAL ] Could not open the default display\n");
        exit(1);
    }

    int major_version_return, minor_version_return;
    if(XdbeQueryExtension(ret.display, &major_version_return, &minor_version_return)) {
        printf("[ INFO ] XDBE version %d.%d\n", major_version_return, minor_version_return);
    } else {
        fprintf(stderr, "[ FATAL ] XDBE is not supported\n");
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
    printf("[ INFO ] Back_buffer ID: %lu\n", ret.back_buffer);

    ret.wa = (XWindowAttributes){0};
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

    render_window *ret_ = calloc(1, sizeof(*ret_));
    memcpy(ret_, &ret, sizeof(*ret_));

    return ret_;
}

bool window_should_close(render_window *w) {
    return w->should_close;
}

//@TODO: Make resize possible
void window_poll(render_window *w) {
    memset(w->input.key_pressed, 0, sizeof(w->input.key_pressed));
    while (XPending(w->display) > 0) {
        XEvent event = (XEvent){0};
        XNextEvent(w->display, &event);
        switch (event.type) {
            case ClientMessage:
                if ((Atom) event.xclient.data.l[0] == w->wm_delete_window)
                    w->should_close = true;
                break;
            case KeyPress: {
                unsigned long code = XLookupKeysym(&event.xkey, 0);
                if (code < 255)
                    w->input.key_pressed[code] = true;
            }
                break;
            case ResizeRequest: 
                printf("Resize\n");
                break;
            default: {}
        }
    }
}

void window_close(render_window *w) {
    XFreeGC(w->display, w->gc);

    XDestroyImage(w->image);
    //free(w->buffer); //X frees this pointer

    XdbeDeallocateBackBufferName(w->display, w->back_buffer);

    XDestroyWindow(w->display, w->window);

    XCloseDisplay(w->display);
    free(w);
}

bool window_key_pressed(render_window *w, char k) {
    return w->input.key_pressed[(int)k];
}

void window_render(render_window *w) {
    XPutImage(w->display, w->back_buffer, w->gc, w->image, 0, 0, 0, 0, w->width, w->height);

    XdbeSwapInfo swap_info = (XdbeSwapInfo){.swap_window = w->window, .swap_action = 0};
    XdbeSwapBuffers(w->display, &swap_info, 1);
    //memset(w->buffer, 0, sizeof(*w->buffer) * w->width * w->height);
}

void window_draw_from_bytes(render_window *w, RGBA32 *bytes, int x0, int y0, int width, int height) {
    int b_w = width;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x0 >= (int)w->width) x0 = w->width - 1;
    if (y0 >= (int)w->height) y0 = w->height - 1;
    if (x0 + width >= (int)w->width) width = w->width - x0;
    if (y0 + height >= (int)w->height) width = w->width - x0;

    for (int y = y0; y < y0 + height; ++y)
        memmove(&w->buffer[y * w->width + x0], &bytes[(y - y0) * b_w], width * sizeof(*bytes));

    /*for (int y = y0; (y < y0 + height) && (y >= 0) && (y < (int)w->height); ++y)
        for (int x = x0; (x < x0 + width) && (x >= 0) && (x < (int)w->width); ++x)
            w->buffer[y * w->width + x] = bytes[(y - y0) * width + (x - x0)];*/
}

int window_width(render_window *window) {
    return window->width;
}

int window_height(render_window *window) {
    return window->height;
}
