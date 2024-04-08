#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

#include "render.h"

struct render_window {
    unsigned int width;
    unsigned int height;
    RGBA32 *buffer;
    bool should_close;
    window_input input;

    HMODULE instance;
    ATOM registered_class;
    HWND window_handle;
    MSG msg;

    BITMAPINFO bitmap_info;
};

static render_window w[1] = {0};

static void window_close(void) {
    free(w->buffer);
}

LRESULT windows_call_back(HWND window, UINT msg, WPARAM wparams, LPARAM lparams) {
    LRESULT ret = 0;
    switch(msg) {
        case WM_CLOSE: {
            DestroyWindow(w->window_handle);
            //ret = DefWindowProc(window, msg, wparams, lparams);
        } break;

        case WM_DESTROY: {
            w->should_close = true;
            window_close();
            PostQuitMessage(0);
            //ret = DefWindowProc(window, msg, wparams, lparams);
        } break;

        case WM_ACTIVATEAPP: {
            //printf("Hehe\n");
        } break;

        case WM_PAINT: {
            PAINTSTRUCT paint;
            //printf("called\n");
            HDC ctx = BeginPaint(w->window_handle, &paint);
            int x = paint.rcPaint.left;
            int y = paint.rcPaint.top;
            int width = paint.rcPaint.right - paint.rcPaint.left;
            int height = paint.rcPaint.bottom - paint.rcPaint.top;

            StretchDIBits(ctx, x, y, width, height, 0, 0, w->width, w->height, w->buffer, &w->bitmap_info, DIB_RGB_COLORS, SRCCOPY);

            EndPaint(w->window_handle, &paint);
        } break;

        case WM_SIZE: {
            ret = DefWindowProc(window, msg, wparams, lparams);
        } break;

        case WM_CHAR: {
            w->input.key_pressed[wparams] = true;
        } break;

        default: {
            ret = DefWindowProc(window, msg, wparams, lparams);
        } break;
    }
    return ret;
}


void window_init(const char *name, unsigned int width, unsigned int height) {
    w->width = width;
    w->height = height;
    w->buffer = calloc(width * height, sizeof(*w->buffer));

    w->instance = GetModuleHandle(NULL);

    WNDCLASS class = {0};
    class.style = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
    class.lpfnWndProc = windows_call_back;
    class.hInstance = w->instance;
    class.hCursor = LoadCursor(0, MAKEINTRESOURCE(32512));

    //HICON     hIcon;
    //LPCSTR    lpszMenuName;
    //
    class.lpszClassName = name;
    w->registered_class = RegisterClassA(&class);
    w->window_handle = CreateWindowA(class.lpszClassName, name, WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT, width, height, 0, 0, w->instance, 0);

    w->bitmap_info.bmiHeader.biSize = sizeof(w->bitmap_info.bmiHeader);
    w->bitmap_info.bmiHeader.biWidth = width;
    w->bitmap_info.bmiHeader.biHeight = -height;
    w->bitmap_info.bmiHeader.biPlanes = 1;
    w->bitmap_info.bmiHeader.biBitCount = sizeof(*w->buffer) * 8;
    w->bitmap_info.bmiHeader.biCompression = BI_RGB;
    w->should_close = false;
}

bool window_should_close(void) {
    return w->should_close;
}

void window_poll(void) {
    memset(w->input.key_pressed, 0, sizeof(w->input.key_pressed));

    if (PeekMessageA(&w->msg, 0, 0, 0, PM_REMOVE)) {
        TranslateMessage(&w->msg);
        DispatchMessage(&w->msg);
    }
}

bool window_key_pressed(char c) {
    return w->input.key_pressed[(int)c];
}

void window_render(void) {
    InvalidateRect(w->window_handle, NULL, FALSE);
    UpdateWindow(w->window_handle);
}

void window_draw_from_bytes(RGBA32 *bytes, int x0, int y0, int width, int height) {
    int b_w = width;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x0 >= (int)w->width) x0 = w->width - 1;
    if (y0 >= (int)w->height) y0 = w->height - 1;
    if (x0 + width >= (int)w->width) width = w->width - x0;
    if (y0 + height >= (int)w->height) height = w->height - y0;

    for (int y = y0; y < y0 + height; ++y)
        memmove(&w->buffer[y * w->width + x0], &bytes[(y - y0) * b_w], width * sizeof(*bytes));
}

int window_width(void) {
    return w->width;
}

int window_height(void) {
    return w->height;
}
