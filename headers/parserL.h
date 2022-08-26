// https://github.com/yuyln/FileParser

#ifndef __PARSER
#define __PARSER
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <float.h>

// #ifdef _MSC_VER
// #define strtok_r strtok_s
// #endif

static const char seps[] = " \r\n:=\t";
char* parser_global_file_str = NULL;
char** parser_global_state = NULL;
size_t parser_global_n = 0;
size_t parser_global_file_size = 0;


void EndParse()
{
    free(parser_global_file_str);
    parser_global_file_str = NULL;

    free(parser_global_state);
    parser_global_state = NULL;
    
    parser_global_n = 0;
    parser_global_file_size = 0;
}

void StartParse(const char* file_path)
{
    if (parser_global_file_str || parser_global_state || parser_global_n || parser_global_file_str)
    {
        EndParse();
    }
    FILE *file = fopen(file_path, "rb");
    if (!file)
    {
        fprintf(stderr, "Couldn't open file %s: %s\n", file_path, strerror(errno));
        exit(1);
    }
    fseek(file, 0, SEEK_SET);
    fseek(file, 0, SEEK_END);

    parser_global_file_size = (size_t)ftell(file);
    parser_global_file_str = (char*)malloc(parser_global_file_size + 1);

    fseek(file, 0, SEEK_SET);
    fread(parser_global_file_str, 1, parser_global_file_size, file);
    parser_global_file_str[parser_global_file_size] = '\0';
    fclose(file);

    char* local_parser_file_str = strdup(parser_global_file_str);
    char* token = strtok(local_parser_file_str, seps);

    while (token)
    {
        ++parser_global_n;
        token = strtok(NULL, seps);
    }

    free(local_parser_file_str);

    parser_global_state = (char**)malloc(sizeof(char*) * parser_global_n);

    token = strtok(parser_global_file_str, seps);
    size_t i = 0;
    while (token)
    {
        parser_global_state[i++] = token;
        token = strtok(NULL, seps);
    }

}

long int FindIndexOfTag(const char* tag)
{
    for (size_t i = 0; i < parser_global_n; ++i)
        if (!strcmp(tag, parser_global_state[i]))
            return i;
            
    return -1;
}

double GetValueDouble(const char* tag)
{
    long int i_tag = FindIndexOfTag(tag);
    if (i_tag < 0)
    {
        fprintf(stderr, "Could not find tag %s\n", tag);
        exit(1);
    }
    return strtod(parser_global_state[i_tag + 1], NULL);
}

float GetValueFloat(const char* tag)
{
    long int i_tag = FindIndexOfTag(tag);
    if (i_tag < 0)
    {
        fprintf(stderr, "Could not find tag %s\n", tag);
        exit(1);
    }
    return strtof(parser_global_state[i_tag + 1], NULL);
}

long int GetValueInt(const char* tag, int base)
{
    long int i_tag = FindIndexOfTag(tag);
    if (i_tag < 0)
    {
        fprintf(stderr, "Could not find tag %s\n", tag);
        exit(1);
    }
    return strtol(parser_global_state[i_tag + 1], NULL, base);
}

unsigned long int GetValueUInt(const char* tag, int base)
{
    long int i_tag = FindIndexOfTag(tag);
    if (i_tag < 0)
    {
        fprintf(stderr, "Could not find tag %s\n", tag);
        exit(1);
    }
    return strtoul(parser_global_state[i_tag + 1], NULL, base);
}

unsigned long long int GetValueULLInt(const char* tag, int base)
{
    long int i_tag = FindIndexOfTag(tag);
    if (i_tag < 0)
    {
        fprintf(stderr, "Could not find tag %s\n", tag);
        exit(1);
    }
    return strtoull(parser_global_state[i_tag + 1], NULL, base);
}
#endif