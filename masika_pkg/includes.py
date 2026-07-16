# -*- coding: utf-8 -*-
"""
includes.py — автоматическое подключение библиотек ("типа как в Rust":
пользователь не пишет #include вообще, если использует стандартные функции —
Masika сама подставляет нужные заголовки).

Работает в две стороны:
  1. по именам известных функций/макросов стандартной библиотеки, встреченным
     в исходных токенах (printf -> stdio.h, sqrt -> math.h, ...);
  2. по именам типов, которые сама Masika подставила при разрешении allme
     (bool -> stdbool.h, int32_t -> stdint.h, size_t -> stddef.h).

Некоторые заголовки требуют доп. флагов компоновки (math.h -> -lm,
pthread.h -> -pthread) — они возвращаются отдельно, чтобы драйвер добавил
их в командную строку gcc.
"""

import re
from typing import List, Set, Tuple
from .lexer import Token, T


# идентификатор функции/макроса -> заголовок
_FUNC_HEADERS = {
    # stdio.h
    **{name: "<stdio.h>" for name in (
        "printf", "fprintf", "sprintf", "snprintf", "vprintf", "vfprintf",
        "scanf", "fscanf", "sscanf", "puts", "fputs", "gets", "fgets",
        "fopen", "fclose", "fread", "fwrite", "fseek", "ftell", "rewind",
        "perror", "feof", "ferror", "getchar", "putchar", "fflush", "remove", "rename",
    )},
    # stdlib.h
    **{name: "<stdlib.h>" for name in (
        "malloc", "calloc", "realloc", "free", "exit", "abort", "atexit",
        "atoi", "atol", "atoll", "atof", "strtol", "strtoul", "strtod",
        "rand", "srand", "qsort", "bsearch", "system", "getenv",
        "abs", "labs", "llabs", "div", "ldiv",
    )},
    # string.h
    **{name: "<string.h>" for name in (
        "strlen", "strcpy", "strncpy", "strcat", "strncat", "strcmp", "strncmp",
        "strchr", "strrchr", "strstr", "strtok", "strdup", "strerror",
        "memcpy", "memmove", "memset", "memcmp",
    )},
    # math.h (+ -lm)
    **{name: "<math.h>" for name in (
        "sqrt", "sqrtf", "pow", "powf", "sin", "cos", "tan", "asin", "acos", "atan",
        "atan2", "floor", "ceil", "fabs", "log", "log2", "log10", "exp", "fmod",
        "round", "trunc", "hypot", "cbrt",
    )},
    # assert.h
    "assert": "<assert.h>",
    # time.h
    **{name: "<time.h>" for name in (
        "time", "clock", "difftime", "mktime", "localtime", "gmtime", "strftime", "asctime",
    )},
    # ctype.h
    **{name: "<ctype.h>" for name in (
        "isalpha", "isdigit", "isalnum", "isspace", "isupper", "islower",
        "toupper", "tolower", "ispunct",
    )},
    # pthread.h (+ -pthread)
    **{name: "<pthread.h>" for name in (
        "pthread_create", "pthread_join", "pthread_mutex_init", "pthread_mutex_lock",
        "pthread_mutex_unlock", "pthread_mutex_destroy", "pthread_cond_init",
        "pthread_cond_wait", "pthread_cond_signal", "pthread_cond_broadcast",
    )},
}

_MATH_HEADER = "<math.h>"
_PTHREAD_HEADER = "<pthread.h>"

# макро-константы -> заголовок
_MACRO_HEADERS = {
    "INT_MAX": "<limits.h>", "INT_MIN": "<limits.h>", "CHAR_BIT": "<limits.h>",
    "LONG_MAX": "<limits.h>", "UINT_MAX": "<limits.h>",
    "FLT_MAX": "<float.h>", "DBL_MAX": "<float.h>", "FLT_EPSILON": "<float.h>",
    "M_PI": "<math.h>", "M_E": "<math.h>",
    "EXIT_SUCCESS": "<stdlib.h>", "EXIT_FAILURE": "<stdlib.h>",
    "NULL": "<stddef.h>",
}

_STDINT_TYPES = ("int8_t", "int16_t", "int32_t", "int64_t",
                  "uint8_t", "uint16_t", "uint32_t", "uint64_t")


def _existing_includes(tokens: List[Token]) -> Set[str]:
    found = set()
    for t in tokens:
        if t.kind != T.PP:
            continue
        m = re.match(r'#\s*include\s*([<"][^">]+[>"])', t.text)
        if m:
            angled = m.group(1)
            if angled.startswith('"'):
                continue
            found.add(angled)
    return found


def compute_includes(tokens: List[Token], resolved_type_names: List[str]) -> Tuple[List[str], List[str]]:
    """Возвращает (список директив #include в порядке вывода, список
    дополнительных флагов компоновки типа -lm/-pthread)."""
    existing = _existing_includes(tokens)
    needed: Set[str] = set()
    link_flags: List[str] = []

    for t in tokens:
        if t.kind == T.IDENT:
            if t.text in _FUNC_HEADERS:
                needed.add(_FUNC_HEADERS[t.text])
            if t.text in _MACRO_HEADERS:
                needed.add(_MACRO_HEADERS[t.text])
        if t.kind == T.IDENT and t.text in ("true", "false"):
            needed.add("<stdbool.h>")

    for name in resolved_type_names:
        if "bool" in name:
            needed.add("<stdbool.h>")
        if any(st in name for st in _STDINT_TYPES):
            needed.add("<stdint.h>")
        if "size_t" in name:
            needed.add("<stddef.h>")

    if _MATH_HEADER in needed:
        link_flags.append("-lm")
    if _PTHREAD_HEADER in needed:
        link_flags.append("-pthread")

    to_add = [h for h in needed if h not in existing]
    # стабильный, предсказуемый порядок вывода
    order = ["<stdio.h>", "<stdlib.h>", "<string.h>", "<math.h>", "<stdbool.h>",
             "<stdint.h>", "<stddef.h>", "<assert.h>", "<time.h>", "<ctype.h>",
             "<limits.h>", "<float.h>", "<pthread.h>"]
    ordered = [h for h in order if h in to_add]
    ordered += sorted(h for h in to_add if h not in order)
    return ordered, sorted(set(link_flags))
