#ifndef __PROFILER
#define __PROFILER
#if _POSIX_C_SOURCE < 199309L
#define _POSIX_C_SOURCE 199310L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

typedef struct Node
{
    char *name;
    size_t i;
    struct timespec T1, T2;
    double dt;
    struct Node *n;
} Node;

static Node *nodes;
static size_t i = 0;

size_t StartMeasure(const char *name);
double EndMeasure(const char *name);
void PrintAll(FILE *stream);

#ifdef PROFILER_IMPLEMENTATION
size_t StartMeasure(const char *name)
{
    struct timespec tempt;
    clock_gettime(CLOCK_REALTIME, &tempt);

    if (i == 0)
    {
        nodes = (Node*)malloc(sizeof(Node));
        nodes->i = 0;
        nodes->T1 = tempt;
        nodes->name = (char*)name;
        nodes->n = NULL;
        nodes->dt = 0.0;
        return i++;
    }


    Node *p = (Node*)malloc(sizeof(Node));
    p->name = (char*) name;
    p->i = i;
    p->n = NULL;
    p->dt = 0.0;
    p->T1 = tempt;

    Node *p1 = nodes;

    while (p1->n != NULL)
    {
        int r = strcmp(name, p1->name);
        if (r == 0)
        {
            p1->T1 = tempt;
            return p1->i;
        }
        p1 = p1->n;
    }
    p1->n = p;
    return i++;
}

double EndMeasure(const char *name)
{
    struct timespec tempt;
    clock_gettime(CLOCK_REALTIME, &tempt);
    Node *p = nodes;
    if (i == 0)
    {
        return -1.0;
    }
    while (p != NULL)
    {
        int r = strcmp(p->name, name);
        if (r != 0)
        {
            p = p->n;
            continue;
        }
        p->T2 = tempt;
        break;
    }
    int dts = p->T2.tv_sec - p->T1.tv_sec;
    int dtns = p->T2.tv_nsec - p->T1.tv_nsec;
    p->dt += (double)dts + (double)dtns / 1.0e9;
    return p->dt;
}

void PrintAll(FILE *stream)
{
    Node *p = nodes;
    while (p != NULL)
    {
        fprintf(stream, "[ %s ] -> [ %.9f ] seg\n", p->name, p->dt);
        p = p->n;
    }
}

#endif

#endif