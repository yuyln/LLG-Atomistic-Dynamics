#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

#include "render.h"

struct render_window {
    HWND handle;
    unsigned int width;
    unsigned int height;
    RGBA32 *buffer;
    bool should_close;

    window_input input;
};



int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
    LPSTR lpCmdLine, int nCmdShow)
{
}

render_window *window_init(unsigned int width, unsigned int height) {
    const char g_szClassName[] = "myWindowClass";
    render_window *ret = calloc(sizeof(render_window), 1);

    WNDCLASSEX wc;
    HWND hwnd;

    //Step 1: Registering the Window Class
    wc.cbSize        = sizeof(WNDCLASSEX);
    wc.style         = 0;
    wc.lpfnWndProc   = 0;
    wc.cbClsExtra    = 0;
    wc.cbWndExtra    = 0;
    wc.hInstance     = 0;
    wc.hIcon         = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor       = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW+1);
    wc.lpszMenuName  = NULL;
    wc.lpszClassName = g_szClassName;
    wc.hIconSm       = LoadIcon(NULL, IDI_APPLICATION);

    if(!RegisterClassEx(&wc))
    {
        MessageBox(NULL, "Window Registration Failed!", "Error!",
            MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

    // Step 2: Creating the Window
    ret->handle = CreateWindowEx(
        WS_EX_CLIENTEDGE,
        g_szClassName,
        "The title of my window",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, width, height,
        NULL, NULL, 0, NULL);

    if (!ret->handle) {
        char buffer[1025] = {0};
        DWORD error = GetLastError();
        FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM,
                      NULL,
                      error, 
                      0,
                      buffer,
                      sizeof(buffer),
                      NULL);
        fprintf(stderr, "Could not create window %s\n", buffer);
        exit(1);
    }

    ret->width = width;
    ret->height = height;
    ret->buffer = calloc(width * height, sizeof(RGBA32));
    ret->should_close = false;
    return ret;
}

bool window_should_close(render_window *window) {

}

void window_poll(render_window *window) {

}

void window_close(render_window *window) {

}

bool window_key_pressed(render_window *window, char c) {

}

void window_render(render_window *window) {

}

void window_draw_from_bytes(render_window *window, RGBA32 *bytes, int x, int y, int width, int height) {

}

int window_width(render_window *window) {

}

int window_height(render_window *window) {

}

void window_resize(render_window *window) {

}
