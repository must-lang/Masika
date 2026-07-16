# -*- coding: utf-8 -*-
"""
typesys.py — вывод типа для allme.

Идея: allme читает ПРАВУЮ часть инициализатора (а если инициализатора нет —
первое найденное впереди присваивание в той же области видимости), вычисляет
подходящий C-тип через маленький вычислитель выражений (eval_expr_type), а
затем "расширяет" этот тип, если дальше по области видимости переменной
присваивается что-то более широкое (напр. allme x = 5, а потом x = 100000 —
итоговый тип должен вместить оба). Расширение — только в сторону увеличения,
никогда в сторону сужения: если сомневаемся — берём тип побезопаснее/пошире,
а не поменьше. Это единственная сторона, в которую можно ошибаться безопасно.

Мы работаем на уровне ТОКЕНОВ, а не полноценного AST — для целей "подобрать
разумный тип" этого достаточно и не требует писать полный парсер C.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from .lexer import Token, T


# ---------------------------------------------------------------------------
# Типы
# ---------------------------------------------------------------------------

@dataclass
class CType:
    name: str                 # как называется в сгенерированном C, напр. "int32_t"
    family: str                # "int" | "float" | "bool" | "char" | "ptr" | "opaque" | "unknown"
    width: int = 0              # ширина в байтах — для сравнения при расширении
    unsigned: bool = False

    def __repr__(self):
        return f"<{self.name}>"


T_I8 = CType("int8_t", "int", 1, False)
T_U8 = CType("uint8_t", "int", 1, True)
T_I16 = CType("int16_t", "int", 2, False)
T_U16 = CType("uint16_t", "int", 2, True)
T_I32 = CType("int32_t", "int", 4, False)
T_U32 = CType("uint32_t", "int", 4, True)
T_I64 = CType("int64_t", "int", 8, False)
T_U64 = CType("uint64_t", "int", 8, True)
T_F32 = CType("float", "float", 4, False)
T_F64 = CType("double", "float", 8, False)
T_F80 = CType("long double", "float", 10, False)
T_BOOL = CType("bool", "bool", 1, False)
T_CHAR = CType("char", "char", 1, False)
T_STR = CType("const char *", "ptr", 8, False)
T_VOIDPTR = CType("void *", "ptr", 8, False)
T_SIZE = CType("size_t", "int", 8, True)
T_UNKNOWN = CType("int32_t", "unknown", 4, False)

_INT_BY_WIDTH = {
    (1, False): T_I8, (1, True): T_U8,
    (2, False): T_I16, (2, True): T_U16,
    (4, False): T_I32, (4, True): T_U32,
    (8, False): T_I64, (8, True): T_U64,
}


def pick_int_type(width: int, unsigned: bool) -> CType:
    for w in (1, 2, 4, 8):
        if w >= width:
            return _INT_BY_WIDTH[(w, unsigned)]
    return _INT_BY_WIDTH[(8, unsigned)]


def widen(a: Optional[CType], b: Optional[CType]) -> CType:
    """Возвращает тип, способный безопасно вместить оба a и b. Никогда не
    сужает — при неопределённости берёт более широкий/безопасный вариант."""
    if a is None:
        return b
    if b is None:
        return a
    if a.family == "unknown":
        return b
    if b.family == "unknown":
        return a
    if a.family == "opaque" or b.family == "opaque":
        return a if a.family == "opaque" else b
    if a.family == "ptr" or b.family == "ptr":
        if a.family == "ptr" and b.family == "ptr":
            return a if a.name == b.name else CType(a.name, "ptr", 8, False)
        return a if a.family == "ptr" else b
    if a.family == "bool" and b.family == "bool":
        return T_BOOL
    if a.family == "float" or b.family == "float":
        fa = a if a.family == "float" else None
        fb = b if b.family == "float" else None
        if fa and fb:
            return fa if fa.width >= fb.width else fb
        return fa or fb
    # оба — int/char/bool семейство
    wa = a.width or 4
    wb = b.width or 4
    return pick_int_type(max(wa, wb), a.unsigned or b.unsigned)


# ---------------------------------------------------------------------------
# Типы литералов
# ---------------------------------------------------------------------------

_INT_LIT_RE = re.compile(r'^(0[xX][0-9a-fA-F]+|0[bB][01]+|0[0-7]+|[0-9]+)([uUlL]*)$')


def type_for_int_literal(text: str) -> CType:
    m = _INT_LIT_RE.match(text)
    if not m:
        return T_I32
    digits, suffix = m.group(1), m.group(2)
    try:
        if digits[:2] in ("0x", "0X"):
            value = int(digits, 16)
        elif digits[:2] in ("0b", "0B"):
            value = int(digits[2:], 2)
        elif digits.startswith("0") and len(digits) > 1 and digits[1:].isdigit():
            value = int(digits, 8)
        else:
            value = int(digits)
    except ValueError:
        value = 0
    suf_low = suffix.lower()
    unsigned = "u" in suf_low
    long_count = suf_low.count("l")
    min_width = 8 if long_count >= 2 else (4 if long_count == 1 else 1)

    for width in (1, 2, 4, 8):
        if width < min_width:
            continue
        if unsigned:
            if 0 <= value < (1 << (width * 8)):
                return pick_int_type(width, True)
        else:
            lo, hi = -(1 << (width * 8 - 1)), (1 << (width * 8 - 1)) - 1
            if lo <= value <= hi:
                return pick_int_type(width, False)
    return pick_int_type(8, unsigned)


def type_for_float_literal(text: str) -> CType:
    if text and text[-1] in "fF":
        return T_F32
    if text and text[-1] in "lL":
        return T_F80
    return T_F64


# ---------------------------------------------------------------------------
# Таблица известных функций стандартной библиотеки (для вызовов в инициализаторе)
# ---------------------------------------------------------------------------

KNOWN_FUNCS: Dict[str, CType] = {
    "strlen": T_SIZE, "strcmp": T_I32, "strncmp": T_I32,
    "sqrt": T_F64, "pow": T_F64, "sin": T_F64, "cos": T_F64, "tan": T_F64,
    "floor": T_F64, "ceil": T_F64, "fabs": T_F64, "log": T_F64, "exp": T_F64,
    "sqrtf": T_F32, "powf": T_F32,
    "abs": T_I32, "labs": T_I64,
    "atoi": T_I32, "atol": T_I64, "atoll": T_I64, "atof": T_F64,
    "getchar": T_I32, "putchar": T_I32,
    "rand": T_I32, "time": T_I64,
    "fopen": CType("FILE *", "ptr", 8, False),
    "strdup": T_STR,
}

MALLOC_LIKE = {"malloc", "calloc", "realloc"}

# Базовые типовые ключевые слова C (используются и для распознавания cast'ов,
# и для сканирования обычных C-объявлений в typescan.py).
TYPE_KEYWORDS = {
    "int", "char", "float", "double", "long", "short", "unsigned", "signed",
    "void", "_Bool", "bool",
}
DECL_QUALIFIERS = {"const", "static", "volatile", "extern", "register", "inline", "restrict"}

_BUILTIN_TYPE_NAME_TO_CTYPE = {
    "int": T_I32, "signed int": T_I32, "unsigned int": T_U32, "unsigned": T_U32,
    "short": T_I16, "short int": T_I16, "unsigned short": T_U16,
    "long": T_I64, "long int": T_I64, "unsigned long": T_U64,
    "long long": T_I64, "unsigned long long": T_U64,
    "char": T_CHAR, "unsigned char": T_U8, "signed char": CType("int8_t", "int", 1, False),
    "float": T_F32, "double": T_F64, "long double": T_F80,
    "bool": T_BOOL, "_Bool": T_BOOL,
    "void": CType("void", "opaque", 0, False),
}


def normalize_builtin_type_text(words: List[str]) -> Optional[CType]:
    key = " ".join(w for w in words if w not in DECL_QUALIFIERS)
    key = re.sub(r"\s+", " ", key).strip()
    return _BUILTIN_TYPE_NAME_TO_CTYPE.get(key)


# ---------------------------------------------------------------------------
# Вычислитель типа выражения (работает на срезе токенов expr_tokens)
# ---------------------------------------------------------------------------

COMPARISON_OPS = {"<", ">", "<=", ">=", "==", "!="}
LOGICAL_OPS = {"&&", "||"}
ADDITIVE_OPS = {"+", "-", "|", "^"}
MULT_OPS = {"*", "/", "%", "<<", ">>", "&"}
ALL_SPLIT_OPS = ADDITIVE_OPS | MULT_OPS


class ExprEval:
    def __init__(self, lookup, warn):
        self.lookup = lookup   # Callable[[str], Optional[CType]]
        self.warn = warn       # Callable[[str], None]

    # --- служебные функции по срезу токенов (список Token, без NEWLINE) ---

    @staticmethod
    def _matching_close(toks: List[Token], open_idx: int, open_ch: str, close_ch: str) -> int:
        depth = 0
        for i in range(open_idx, len(toks)):
            if toks[i].text == open_ch:
                depth += 1
            elif toks[i].text == close_ch:
                depth -= 1
                if depth == 0:
                    return i
        return len(toks) - 1

    def eval(self, toks: List[Token]) -> CType:
        toks = [t for t in toks if t.kind != T.NEWLINE]
        if not toks:
            return T_UNKNOWN
        return self._eval_stripped(toks)

    def _eval_stripped(self, toks: List[Token]) -> CType:
        # снимаем обрамляющие круглые скобки, если они охватывают ВСЁ выражение
        while len(toks) >= 2 and toks[0].text == "(" and toks[-1].text == ")":
            close = self._matching_close(toks, 0, "(", ")")
            if close == len(toks) - 1:
                toks = toks[1:-1]
                if not toks:
                    return T_UNKNOWN
            else:
                break

        if not toks:
            return T_UNKNOWN

        # ведущий '!' -> результат bool
        if toks[0].kind == T.PUNCT and toks[0].text == "!":
            return T_BOOL

        # тернарный оператор cond ? a : b (по глубине 0)
        depth = 0
        qmark_idx = -1
        colon_idx = -1
        qdepth = 0
        for i, t in enumerate(toks):
            if t.kind == T.PUNCT and t.text in ("(", "[", "{"):
                depth += 1
            elif t.kind == T.PUNCT and t.text in (")", "]", "}"):
                depth -= 1
            elif depth == 0 and t.kind == T.PUNCT and t.text == "?":
                if qmark_idx == -1:
                    qmark_idx = i
                qdepth += 1
            elif depth == 0 and t.kind == T.PUNCT and t.text == ":" and qmark_idx != -1:
                qdepth -= 1
                if qdepth == 0:
                    colon_idx = i
                    break
        if qmark_idx != -1 and colon_idx != -1:
            then_t = self._eval_stripped(toks[qmark_idx + 1:colon_idx])
            else_t = self._eval_stripped(toks[colon_idx + 1:])
            return widen(then_t, else_t)

        # сравнения / логические операторы на глубине 0 -> bool
        depth = 0
        for t in toks:
            if t.kind == T.PUNCT and t.text in ("(", "[", "{"):
                depth += 1
            elif t.kind == T.PUNCT and t.text in (")", "]", "}"):
                depth -= 1
            elif depth == 0 and t.kind == T.PUNCT and (t.text in COMPARISON_OPS or t.text in LOGICAL_OPS):
                return T_BOOL

        # компаунд-литерал (Type){...} на весь срез
        if toks[0].text == "(" :
            close = self._matching_close(toks, 0, "(", ")")
            if close + 1 < len(toks) and toks[close + 1].text == "{" and toks[-1].text == "}":
                brace_close = self._matching_close(toks, close + 1, "{", "}")
                if brace_close == len(toks) - 1:
                    type_text = " ".join(x.text for x in toks[1:close])
                    return CType(type_text, "opaque", 0, False)

        # cast: (TYPE) rest, где TYPE — известные ключевые слова типа
        if toks[0].text == "(":
            close = self._matching_close(toks, 0, "(", ")")
            if close < len(toks) - 1:
                inner = toks[1:close]
                inner_words = [x.text for x in inner if x.kind in (T.KEYWORD, T.IDENT) or x.text == "*"]
                looks_like_type = inner and all(
                    (x.kind == T.KEYWORD and (x.text in TYPE_KEYWORDS or x.text in DECL_QUALIFIERS))
                    or x.text == "*"
                    for x in inner
                )
                if looks_like_type:
                    star_count = sum(1 for x in inner if x.text == "*")
                    base = normalize_builtin_type_text([x.text for x in inner if x.text != "*"])
                    if base is not None:
                        if star_count:
                            return CType(base.name + " " + "*" * star_count, "ptr", 8, False)
                        return base
                    return T_UNKNOWN

        # разбиваем по операторам сложения/умножения на глубине 0 и расширяем
        depth = 0
        split_positions = []
        for i, t in enumerate(toks):
            if t.kind == T.PUNCT and t.text in ("(", "[", "{"):
                depth += 1
            elif t.kind == T.PUNCT and t.text in (")", "]", "}"):
                depth -= 1
            elif depth == 0 and t.kind == T.PUNCT and t.text in ALL_SPLIT_OPS:
                # отличаем унарный +/-/& от бинарного: унарный, если это первый
                # токен среза или предыдущий (по глубине 0) токен — тоже оператор/'('/','/
                # ключевое слово return
                if i == 0:
                    continue
                prev = toks[i - 1]
                if prev.kind == T.PUNCT and (prev.text in ALL_SPLIT_OPS or prev.text in ("(", ",", "=") or prev.text in COMPARISON_OPS):
                    continue
                if prev.kind == T.KEYWORD and prev.text == "return":
                    continue
                split_positions.append(i)

        if split_positions:
            atoms = []
            start = 0
            for p in split_positions:
                atoms.append(toks[start:p])
                start = p + 1
            atoms.append(toks[start:])
            result = None
            for atom in atoms:
                if not atom:
                    continue
                result = widen(result, self._eval_atom(atom))
            return result if result is not None else T_UNKNOWN

        return self._eval_atom(toks)

    def _eval_atom(self, toks: List[Token]) -> CType:
        toks = [t for t in toks if t.kind != T.NEWLINE]
        if not toks:
            return T_UNKNOWN

        # снимаем ведущие унарные операторы
        i = 0
        deref_count = 0
        addr_count = 0
        force_bool = False
        while i < len(toks) and toks[i].kind == T.PUNCT and toks[i].text in ("-", "+", "~", "*", "&", "!"):
            if toks[i].text == "*":
                deref_count += 1
            elif toks[i].text == "&":
                addr_count += 1
            elif toks[i].text == "!":
                force_bool = True
            i += 1
        rest = toks[i:]
        if force_bool:
            return T_BOOL
        if not rest:
            return T_UNKNOWN

        base = self._eval_primary(rest)
        if deref_count and base.family == "ptr":
            name = base.name
            for _ in range(deref_count):
                if name.endswith("*"):
                    name = name[:-1].rstrip()
            base = CType(name if name else "int", "int" if name in ("", ) else base.family, base.width, base.unsigned) \
                if False else base  # упрощение: снятие разыменования не критично для ширины
        if addr_count:
            base = CType(base.name + " " + "*" * addr_count, "ptr", 8, False)
        return base

    def _eval_primary(self, toks: List[Token]) -> CType:
        t0 = toks[0]

        if len(toks) == 1:
            if t0.kind == T.INT:
                return type_for_int_literal(t0.text)
            if t0.kind == T.FLOAT:
                return type_for_float_literal(t0.text)
            if t0.kind == T.CHAR:
                return T_CHAR
            if t0.kind == T.STRING:
                return T_STR
            if t0.kind == T.IDENT:
                if t0.text in ("true", "false"):
                    return T_BOOL
                if t0.text == "NULL":
                    return T_VOIDPTR
                found = self.lookup(t0.text)
                if found is not None:
                    return found
                self.warn(f"не удалось определить тип идентификатора '{t0.text}', считаю неизвестным")
                return T_UNKNOWN
            if t0.kind == T.KEYWORD and t0.text in ("true", "false"):
                return T_BOOL

        # sizeof(...) или sizeof x
        if t0.kind == T.KEYWORD and t0.text == "sizeof":
            return T_SIZE

        # вызов функции: IDENT ( ... )
        if t0.kind == T.IDENT and len(toks) >= 2 and toks[1].text == "(":
            close = self._matching_close(toks, 1, "(", ")")
            args_tokens = toks[2:close]
            if t0.text in MALLOC_LIKE:
                for j, tk in enumerate(args_tokens):
                    if tk.kind == T.KEYWORD and tk.text == "sizeof" and j + 1 < len(args_tokens) and args_tokens[j + 1].text == "(":
                        c2 = self._matching_close(args_tokens, j + 1, "(", ")")
                        type_words = [x.text for x in args_tokens[j + 2:c2] if x.text != "*"]
                        stars = sum(1 for x in args_tokens[j + 2:c2] if x.text == "*")
                        base = normalize_builtin_type_text(type_words)
                        base_name = base.name if base else (" ".join(type_words) or "void")
                        return CType(base_name + " " + "*" * (stars + 1), "ptr", 8, False)
                return T_VOIDPTR
            if t0.text in KNOWN_FUNCS:
                return KNOWN_FUNCS[t0.text]
            found = self.lookup(t0.text + "()")
            if found is not None:
                return found
            self.warn(f"неизвестная функция '{t0.text}' в инициализаторе allme — тип результата не определён")
            return T_UNKNOWN

        # общие скобки (уже не cast и не compound literal — сюда попадаем редко)
        if t0.text == "(":
            close = self._matching_close(toks, 0, "(", ")")
            if close == len(toks) - 1:
                return self._eval_stripped(toks[1:close])

        return T_UNKNOWN
