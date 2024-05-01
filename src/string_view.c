#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <inttypes.h>

#include "string_view.h"
#include "logging.h"
#include "allocator.h"

uint64_t vfmt_get_size(const char *fmt, va_list args) {
    return vsnprintf(NULL, 0, fmt, args);
}

void str_cat_str(string *s, string s2) {
    if (!s->can_manipulate) {
        logging_log(LOG_ERROR, "String argument passed to %s for manipulation can not be manipulated", __func__);
        return;
    }
    if (!s2.str) {
        logging_log(LOG_WARNING, "String argument passed to concatenate (s2) is NULL. Trying to concatenate %.*s.", (int)s->len, s->str);
        return;
    }

    uint64_t old_len = s->len;
    s->len += s2.len;
    s->str = mrealloc(s->str, s->len + 1);
    memmove(&s->str[old_len], s2.str, s2.len);
    s->str[s->len] = '\0';
}

void str_cat_cstr(string *s, const char *s2) {
    if (!s->can_manipulate) {
        logging_log(LOG_ERROR, "String argument passed to %s for manipulation can not be manipulated", __func__);
        return;
    }

    uint64_t old_len = s->len;
    uint64_t s2_len = strlen(s2);
    s->len += s2_len;
    s->str = mrealloc(s->str, s->len + 1);
    memmove(&s->str[old_len], s2, s2_len + 1);
}

void str_cat_fmt(string *s, const char *fmt, ...) {
    if (!s->can_manipulate) {
        logging_log(LOG_ERROR, "String argument passed to %s for manipulation can not be manipulated", __func__);
        return;
    }
    char *tmp = NULL;

    va_list arg_list;
    va_start(arg_list, fmt);
    if (!fmt) {
        logging_log(LOG_WARNING, "Format string provided is NULL");
        goto err;
    }

    uint64_t s2_len = vsnprintf(NULL, 0, fmt, arg_list) + 1;
    va_end(arg_list);

    tmp = mmalloc(s2_len);

    va_start(arg_list, fmt);
    vsnprintf(tmp, s2_len, fmt, arg_list);
    va_end(arg_list);

    s2_len = strlen(tmp);

    uint64_t old_len = s->len;
    s->len += s2_len;
    s->str = mrealloc(s->str, s->len + 1);
    memmove(&s->str[old_len], tmp, s2_len + 1);
err: 
    mfree(tmp);
}

void str_mfree(string *s) {
    if (s->can_manipulate) {
        mfree(s->str);
        memset(s, 0, sizeof(*s));
    }
}

const char *str_as_cstr(string *s) {
    return s->str;
}

string str_from_cstr(const char *s) {
    string ret = {.can_manipulate=true};
    str_cat_cstr(&ret, s);
    return ret;
}

string str_from_fmt(const char *fmt, ...) {
    string ret = {.can_manipulate=true};
    va_list arg_list;
    va_start(arg_list, fmt);
    if (!fmt) {
        logging_log(LOG_WARNING, "Format string provided is NULL");
        return (string){0};
    }
    uint64_t tmp_len = vsnprintf(NULL, 0, fmt, arg_list) + 1;
    va_end(arg_list);

    char *tmp = mmalloc(tmp_len);

    va_start(arg_list, fmt);
    vsnprintf(tmp, tmp_len, fmt, arg_list);
    va_end(arg_list);

    tmp_len = strlen(tmp);
    ret.len = tmp_len;
    ret.str = mmalloc(ret.len + 1);
    memmove(ret.str, tmp, tmp_len + 1);
    mfree(tmp);
    return ret;
}
