#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <inttypes.h>

#include "string_view.h"
#include "logging.h"

uint64_t vfmt_get_size(const char *fmt, va_list args) {
    return vsnprintf(NULL, 0, fmt, args);
}

void str_cat_str(string *s, string s2) {
    uint64_t old_len = s->len;
    s->len += s2.len;
    char *old_ptr = s->str;
    s->str = realloc(s->str, s->len);
    if (!s->str) {
        logging_log(LOG_ERROR, "Could not realloc pointer of str \"%.*s\" to cat with \"%.*s\": %s", (int)old_len, old_ptr, (int)s2.len, s2.str, strerror(errno));
        s->str = old_ptr;
    } else
        memmove(&s->str[old_len], s2.str, s2.len);
}

void str_cat_cstr(string *s, const char *s2) {
    uint64_t old_len = s->len;
    uint64_t s2_len = strlen(s2);
    s->len += s2_len;
    char *old_ptr = s->str;
    s->str = realloc(s->str, s->len);
    if (!s->str) {
        logging_log(LOG_ERROR, "Could not realloc pointer of str \"%.*s\" to cat with \"%s\": %s", (int)old_len, old_ptr, s2, strerror(errno));
        s->str = old_ptr;
    } else
        memmove(&s->str[old_len], s2, s2_len);
}

void str_cat_fmt(string *s, const char *fmt, ...) {
    va_list arg_list;
    va_start(arg_list, fmt);
    uint64_t s2_len = vsnprintf(NULL, 0, fmt, arg_list) + 1;
    va_end(arg_list);

    char *tmp = calloc(s2_len, 1);
    if (!tmp) {
        logging_log(LOG_ERROR, "Could not alloc %"PRIu64" bytes for tmp: %s", s2_len, strerror(errno));
        goto err;
    }

    va_start(arg_list, fmt);
    vsnprintf(tmp, s2_len, fmt, arg_list);
    va_end(arg_list);

    s2_len = strlen(tmp);

    uint64_t old_len = s->len;
    s->len += s2_len;
    char *old_ptr = s->str;
    s->str = realloc(s->str, s->len);
    if (!s->str) {
        logging_log(LOG_ERROR, "Could not realloc pointer of str \"%.*s\" to cat with \"%s\": %s", (int)old_len, old_ptr, tmp, strerror(errno));
        s->str = old_ptr;
    } else
        memmove(&s->str[old_len], tmp, s2_len);
err: 
    free(tmp);
}

void str_cat_sv(string *s, string_view sv) {
    str_cat_str(s, (string){.str = (char*)sv.str, .len = sv.len});
}

void str_free(string *s) {
    free(s->str);
    memset(s, 0, sizeof(*s));
}

const char *str_as_cstr(string *s) {
    char *old_ptr = s->str;
    uint64_t old_len = s->len;

    if (s->len > 0)
        if (s->str[s->len - 1] == '\0')
            return s->str;

    if (!(s->str = realloc(s->str, s->len + 1))) {
        logging_log(LOG_ERROR, "Could not realloc pointer of str \"%.*s\" to include null terminator, returning null cstr: %s", (int)old_len, old_ptr, strerror(errno));
        return "\0";
    }

    s->str[old_len] = '\0';
    return s->str;
}

string str_from_cstr(const char *s) {
    string ret = {0};
    str_cat_cstr(&ret, s);
    return ret;
}

string str_from_fmt(const char *fmt, ...) {
    string ret = {0};
    va_list arg_list;
    va_start(arg_list, fmt);
    uint64_t tmp_len = vsnprintf(NULL, 0, fmt, arg_list) + 1;
    va_end(arg_list);

    char *tmp = calloc(tmp_len, 1);
    if (!tmp) {
        logging_log(LOG_ERROR, "Could not alloc %"PRIu64" bytes for tmp: %s", tmp_len, strerror(errno));
        goto err;
    }

    va_start(arg_list, fmt);
    vsnprintf(tmp, tmp_len, fmt, arg_list);
    va_end(arg_list);

    tmp_len = strlen(tmp);
    ret.len = tmp_len;
    ret.str = calloc(ret.len, 1);
    if (!ret.str) {
        logging_log(LOG_ERROR, "Could not alloc pointer for str [%"PRIu64" bytes], returning null string: %s", (int)ret.len, strerror(errno));
        memset(&ret, 0, sizeof(ret));
    }
    else
        memmove(ret.str, tmp, tmp_len);
err: 
    free(tmp);
    return ret;
}

const char *sv_as_cstr(string_view s) {
    return s.str;
}

string_view sv_from_cstr(const char *s) {
    return (string_view){.str = s, .len = strlen(s)};
}
