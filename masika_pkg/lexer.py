# -*- coding: utf-8 -*-
"""
lexer.py — лексер языка Must.

Задача этого модуля — ТОЛЬКО токенизация. Никакой семантики (allme,
авто-';', авто-include) здесь нет — она живёт в asi.py / typesys.py /
includes.py и работает уже поверх готового списка токенов.

Ключевая архитектурная идея всего Masika: мы НЕ строим полноценное дерево
разбора C и не пересобираем текст из токенов. Мы находим в оригинальном
исходнике места (byte-offset'ы), которые нужно поправить (вставить ';',
заменить 'allme' на реальный тип, добавить #include в начало), и делаем
точечные правки прямо в исходном тексте. Это значит, что все пробелы,
комментарии и форматирование пользователя остаются как есть — 100%
lossless кроме тех мест, которые мы осознанно меняем.

Поэтому каждый токен несёт .start/.end — byte offset в исходной строке.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class T(Enum):
    IDENT = auto()
    KEYWORD = auto()
    INT = auto()
    FLOAT = auto()
    CHAR = auto()
    STRING = auto()
    PUNCT = auto()
    NEWLINE = auto()   # значимый физический перевод строки вне строк/комментариев
    PP = auto()        # директива препроцессора целиком (#include, #define, ...)
    EOF = auto()


# Ключевые слова C11/C17 + пара мягких удобств Must (true/false/bool/NULL
# считаем идентификаторами для C, но добавляем как "мягкие" константы через
# автоинклюд <stdbool.h>/<stddef.h> — см. includes.py). Здесь просто список
# слов, которые нельзя использовать как имя переменной под allme.
C_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if",
    "inline", "int", "long", "register", "restrict", "return", "short",
    "signed", "sizeof", "static", "struct", "switch", "typedef", "union",
    "unsigned", "void", "volatile", "while",
    "_Bool", "_Complex", "_Imaginary", "_Alignas", "_Alignof", "_Atomic",
    "_Generic", "_Noreturn", "_Static_assert", "_Thread_local",
}

# Собственное ключевое слово Must.
MUST_KEYWORD = "allme"

# Пунктуаторы, длинные варианты проверяются раньше коротких.
PUNCT_3 = ("<<=", ">>=", "...")
PUNCT_2 = ("->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||",
           "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=")
PUNCT_1 = set("+-*/%&|^~!<>=(){}[];:,.?#")


@dataclass
class Token:
    kind: T
    text: str
    line: int
    col: int
    start: int
    end: int
    # Структурные пометки, проставляются позже в asi.py (изначально False).
    is_control_paren_close: bool = False   # ')' закрывающая if/while/for/switch(...)
    is_value_brace: bool = False           # эта '{' или '}' — compound literal/initializer, не блок
    paren_preceded_by_ident: bool = False  # эта ')' — перед открывающей '(' стоял идентификатор

    def __repr__(self):
        return f"Token({self.kind.name}, {self.text!r}, {self.line}:{self.col})"


class LexError(Exception):
    def __init__(self, msg, line, col):
        super().__init__(f"{line}:{col}: {msg}")
        self.line = line
        self.col = col


def lex(src: str) -> List[Token]:
    """Токенизирует исходный текст Must/C. Возвращает плоский список Token,
    включая значимые NEWLINE (только те, что вне строк/символов/комментариев)
    и PP-токены для целых строк препроцессора."""
    tokens: List[Token] = []
    i = 0
    n = len(src)
    line = 1
    col = 1

    def advance(k=1):
        nonlocal i, line, col
        for _ in range(k):
            if i < n and src[i] == "\n":
                line += 1
                col = 1
            else:
                col += 1
            i += 1

    while i < n:
        c = src[i]

        # --- горизонтальные пробелы ---
        if c in " \t\r":
            advance()
            continue

        # --- перевод строки: значимый токен ---
        if c == "\n":
            l, cl, start = line, col, i
            advance()
            tokens.append(Token(T.NEWLINE, "\n", l, cl, start, i))
            continue

        # --- построчный комментарий // ... ---
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                advance()
            continue

        # --- блочный комментарий /* ... */ (переводы строк внутри не значимы) ---
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            l, cl = line, col
            advance(2)
            closed = False
            while i < n:
                if src[i] == "*" and i + 1 < n and src[i + 1] == "/":
                    advance(2)
                    closed = True
                    break
                advance()
            if not closed:
                raise LexError("незакрытый блочный комментарий /* ... */", l, cl)
            continue

        # --- директива препроцессора (учитываем продолжение строки через '\') ---
        if c == "#":
            l, cl, start = line, col, i
            while i < n:
                if src[i] == "\\" and i + 1 < n and src[i + 1] == "\n":
                    advance(2)
                    continue
                if src[i] == "\n":
                    break
                advance()
            tokens.append(Token(T.PP, src[start:i], l, cl, start, i))
            continue

        # --- строковый литерал ---
        if c == '"':
            l, cl, start = line, col, i
            advance()
            while i < n and src[i] != '"':
                if src[i] == "\\" and i + 1 < n:
                    advance(2)
                elif src[i] == "\n":
                    raise LexError("незакрытый строковый литерал", l, cl)
                else:
                    advance()
            if i >= n:
                raise LexError("незакрытый строковый литерал", l, cl)
            advance()
            tokens.append(Token(T.STRING, src[start:i], l, cl, start, i))
            continue

        # --- символьный литерал ---
        if c == "'":
            l, cl, start = line, col, i
            advance()
            while i < n and src[i] != "'":
                if src[i] == "\\" and i + 1 < n:
                    advance(2)
                elif src[i] == "\n":
                    raise LexError("незакрытый символьный литерал", l, cl)
                else:
                    advance()
            if i >= n:
                raise LexError("незакрытый символьный литерал", l, cl)
            advance()
            tokens.append(Token(T.CHAR, src[start:i], l, cl, start, i))
            continue

        # --- числа: int/float, dec/hex/oct/bin, суффиксы u/U/l/L/f/F ---
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            l, cl, start = line, col, i
            is_float = False
            if c == "0" and i + 1 < n and src[i + 1] in "xX":
                advance(2)
                while i < n and src[i] in "0123456789abcdefABCDEF":
                    advance()
                if i < n and src[i] == "." :
                    is_float = True
                    advance()
                    while i < n and src[i] in "0123456789abcdefABCDEF":
                        advance()
                if i < n and src[i] in "pP":  # hex-float экспонента
                    is_float = True
                    advance()
                    if i < n and src[i] in "+-":
                        advance()
                    while i < n and src[i].isdigit():
                        advance()
            elif c == "0" and i + 1 < n and src[i + 1] in "bB":
                advance(2)
                while i < n and src[i] in "01":
                    advance()
            else:
                while i < n and src[i].isdigit():
                    advance()
                if i < n and src[i] == "." and not (i + 1 < n and src[i + 1] == "."):
                    is_float = True
                    advance()
                    while i < n and src[i].isdigit():
                        advance()
                if i < n and src[i] in "eE":
                    is_float = True
                    advance()
                    if i < n and src[i] in "+-":
                        advance()
                    while i < n and src[i].isdigit():
                        advance()
            while i < n and src[i] in "uUlLfF":
                if src[i] in "fF":
                    is_float = True
                advance()
            tokens.append(Token(T.FLOAT if is_float else T.INT, src[start:i], l, cl, start, i))
            continue

        # --- идентификаторы / ключевые слова ---
        if c.isalpha() or c == "_":
            l, cl, start = line, col, i
            while i < n and (src[i].isalnum() or src[i] == "_"):
                advance()
            text = src[start:i]
            kind = T.KEYWORD if (text in C_KEYWORDS or text == MUST_KEYWORD) else T.IDENT
            tokens.append(Token(kind, text, l, cl, start, i))
            continue

        # --- пунктуация: сначала длинные варианты ---
        matched = None
        for p in PUNCT_3:
            if src.startswith(p, i):
                matched = p
                break
        if not matched:
            for p in PUNCT_2:
                if src.startswith(p, i):
                    matched = p
                    break
        if not matched and c in PUNCT_1:
            matched = c

        if matched:
            l, cl, start = line, col, i
            advance(len(matched))
            tokens.append(Token(T.PUNCT, matched, l, cl, start, i))
            continue

        raise LexError(f"неожиданный символ {c!r}", line, col)

    tokens.append(Token(T.EOF, "", line, col, i, i))
    return tokens


def real_tokens(tokens: List[Token]):
    """Итератор по токенам без NEWLINE — удобно для поиска 'предыдущего/
    следующего осмысленного токена'."""
    for t in tokens:
        if t.kind != T.NEWLINE:
            yield t
