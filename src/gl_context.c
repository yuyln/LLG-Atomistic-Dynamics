#include "gl_context.h"
#include <stdlib.h>
#include <stdio.h>

// https://github.com/tsoding/ded/blob/master/src/main.c
void MessageCallback(GLenum source,
                     GLenum type,
                     GLuint id,
                     GLenum severity,
                     GLsizei length,
                     const GLchar *message,
                     const void *userParam) {
    (void)source;
    (void)id;
    (void)length;
    (void)userParam;
    (void)type;
    (void)severity;
    (void)message;
#ifdef DEBUG
    fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
            (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
            type, severity, message);
#endif
}

void initGL(GLFWwindow **window, const char *name, unsigned int width, unsigned int height) {
    if (!glfwInit()) {
        fprintf(stderr, "Couldn't init GLFW\n");
        exit(1);
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    *window = glfwCreateWindow(width, height, name, NULL, NULL);

    if (!*window) {
        fprintf(stderr, "Couldn't create window\n");
        glfwTerminate();
        exit(1);
    }

    glfwMakeContextCurrent(*window);

    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Couldn't init GLEW\n");
        glfwTerminate();
        exit(1);
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);

    if (GLEW_ARB_debug_output) {
        glEnable(GL_DEBUG_OUTPUT);
        glDebugMessageCallback(MessageCallback, 0);
    }
    else
        fprintf(stderr, "WARNING! GLEW_ARB_debug_output is not available");
}
