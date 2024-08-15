#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <inttypes.h>

#include "string_builder.h"
#include "logging.h"
#include "utils.h"
#include "allocator.h"

void sb_cat_sb(string_builder *s, string_builder s2) {
    for (uint64_t i = 0; i < s2.len; ++i)
        da_append(s, s2.items[i]);
}

void sb_cat_cstr(string_builder *s, const char *s2) {
    uint64_t s2_len = strlen(s2);
    for (uint64_t i = 0; i < s2_len; ++i)
        da_append(s, s2[i]);
}

void sb_cat_fmt(string_builder *s, const char *fmt, ...) {
    char *tmp = NULL;

    va_list arg_list;
    va_start(arg_list, fmt);
    if (!fmt) {
        logging_log(LOG_WARNING, "Format string_builder provided is NULL");
        goto err;
    }

    uint64_t s2_len = vsnprintf(NULL, 0, fmt, arg_list) + 1;
    va_end(arg_list);

    tmp = mmalloc(s2_len);

    va_start(arg_list, fmt);
    vsnprintf(tmp, s2_len, fmt, arg_list);
    va_end(arg_list);
    for (uint64_t i = 0; i < s2_len; ++i)
        da_append(s, tmp[i]);
err: 
    mfree(tmp);
}

void sb_free(string_builder *s) {
    mfree(s->items);
    memset(s, 0, sizeof(*s));
}

char *sb_as_cstr(string_builder *s) {
    if (s->items[s->len - 1] != '\0')
        da_append(s, '\0');
    return s->items;
}
