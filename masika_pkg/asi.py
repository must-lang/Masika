# -*- coding: utf-8 -*-
"""
asi.py — автоматическая расстановка ';' (Automatic Semicolon Insertion).

Правило по духу как в Go: смотрим на ПОСЛЕДНИЙ токен перед переводом строки.
Если это токен, которым обычно ЗАКАНЧИВАЕТСЯ выражение/оператор (идентификатор,
число, строка, символ, ')', ']', '++', '--', 'return'/'break'/'continue'
без выражения после, или ')' закрывающая compound literal '}') — значит
строка скорее всего закончена, и если явного ';' нет, мы его вставляем.

Как и в Go, следствие этого правила: чтобы ПРОДОЛЖИТЬ выражение на
следующей строке, оператор должен остаться в КОНЦЕ текущей строки, а не
в начале следующей (напр. многострочный вызов функции без trailing comma
перед закрывающей ')' работать не будет — как и в Go, добавляй запятую
после последнего аргумента, если переносишь ')' на отдельную строку).

Два важных структурных исключения, которых нет в Go, но которые нужны
в C-подобном языке:

  1. ')' закрывающая заголовок if/while/for/switch(...) — НИКОГДА не
     считается концом выражения (иначе 'if (x)\n{' превратилось бы в
     пустой if: 'if (x);\n{'). Это же правило распространяется на ')'
     заголовка функции, если сразу после неё (возможно на следующей
     строке) идёт '{' — то есть это определение функции, а не прототип.

  2. '}' закрывающая БЛОК (тело функции/if/while/for/switch/do, любой
     голый блок { ... }) — НЕ требует ';'. А вот '}' закрывающая
     compound literal или initializer-list ( '= { ... }' ) — требует,
     если это конец statement'а.

Это не претендует на 100% формальную строгость (у ASI в любом языке —
Go, JS — есть пограничные случаи). Явно поставленный пользователем ';'
всегда работает и отключает эвристику для этого места — это "аварийный
выход" на случай редких неоднозначностей.
"""

from dataclasses import dataclass
from typing import List, Set
from .lexer import Token, T


TRIGGER_KEYWORDS = {"return", "break", "continue"}
TRIGGER_PUNCT = {")", "]", "++", "--"}
CONTROL_KEYWORDS = {"if", "while", "for", "switch"}


def annotate_structure(tokens: List[Token]) -> None:
    """Проставляет is_control_paren_close и is_value_brace_close прямо на
    токенах (мутирует список). Требует отдельного прохода, т.к. нужен
    полный контекст (стек скобок), которого нет при чистой токенизации."""

    paren_ctrl_stack: List[bool] = []      # для каждой открытой '(' — была ли она control-header
    paren_ident_stack: List[bool] = []     # для каждой открытой '(' — стоял ли перед ней идентификатор
    # (это отличает 'main(' / 'foo(' — вызов или объявление функции — от
    #  '(Point){...}' compound literal, где перед '(' идентификатора нет:
    #  там обычно '=', ',', 'return' и т.п.)
    brace_value_stack: List[bool] = []     # для каждой открытой '{' — является ли она "value brace"
    brace_is_do_body: List[bool] = []      # параллельный стек: это '{' — тело do { ... } while?
    pending_control = False                # только что видели if/while/for/switch
    pending_is_do = False                  # только что видели 'do'
    just_closed_do_body = False            # только что закрылась '}' тела do (или был однострочный do)
    single_stmt_do_pending = False         # do без блока '{}' — ждём 'while' от него

    def prev_real_idx(idx: int) -> int:
        j = idx - 1
        while j >= 0 and tokens[j].kind == T.NEWLINE:
            j -= 1
        return j

    for idx, tok in enumerate(tokens):
        if tok.kind == T.NEWLINE or tok.kind == T.PP:
            continue

        if tok.kind == T.KEYWORD and tok.text == "do":
            pending_is_do = True
            # однострочное тело (без '{') определяем по следующему реальному токену
            j = idx + 1
            while j < len(tokens) and tokens[j].kind == T.NEWLINE:
                j += 1
            if j < len(tokens) and not (tokens[j].kind == T.PUNCT and tokens[j].text == "{"):
                single_stmt_do_pending = True
            continue

        if tok.kind == T.KEYWORD and tok.text == "while":
            if just_closed_do_body or single_stmt_do_pending:
                # это 'while' от do-while — его '(' обычная, не control,
                # чтобы закрывающая ')' осталась триггером для ';'
                just_closed_do_body = False
                single_stmt_do_pending = False
            else:
                pending_control = True
            continue

        if tok.kind == T.KEYWORD and tok.text in CONTROL_KEYWORDS:
            pending_control = True
            continue

        if tok.kind == T.PUNCT and tok.text == "(":
            paren_ctrl_stack.append(pending_control)
            j = prev_real_idx(idx)
            preceded_by_ident = j >= 0 and tokens[j].kind == T.IDENT
            paren_ident_stack.append(preceded_by_ident)
            pending_control = False
            continue

        if tok.kind == T.PUNCT and tok.text == ")":
            is_ctrl = paren_ctrl_stack.pop() if paren_ctrl_stack else False
            preceded_by_ident = paren_ident_stack.pop() if paren_ident_stack else False
            tok.is_control_paren_close = is_ctrl
            tok.paren_preceded_by_ident = preceded_by_ident
            pending_control = False
            continue

        if tok.kind == T.PUNCT and tok.text == "{":
            j = prev_real_idx(idx)
            is_value = False
            if j >= 0:
                prev = tokens[j]
                if prev.kind == T.PUNCT and prev.text in ("=", ",", "{"):
                    is_value = True
                elif prev.kind == T.PUNCT and prev.text == ")" and not prev.is_control_paren_close \
                        and not prev.paren_preceded_by_ident:
                    # ')' не control-заголовка и не вызова/объявления функции
                    # (перед '(' не было идентификатора) => это compound
                    # literal вида '(Type){...}', а не тело функции.
                    is_value = True
            brace_value_stack.append(is_value)
            brace_is_do_body.append(pending_is_do)
            tok.is_value_brace = is_value
            pending_is_do = False
            continue

        if tok.kind == T.PUNCT and tok.text == "}":
            is_value = brace_value_stack.pop() if brace_value_stack else False
            was_do_body = brace_is_do_body.pop() if brace_is_do_body else False
            tok.is_value_brace = is_value
            if was_do_body:
                just_closed_do_body = True
            pending_control = False
            continue

        pending_is_do = False
        pending_control = False


def _is_trigger(tok: Token) -> bool:
    if tok.kind in (T.IDENT, T.INT, T.FLOAT, T.CHAR, T.STRING):
        return True
    if tok.kind == T.KEYWORD and tok.text in TRIGGER_KEYWORDS:
        return True
    if tok.kind == T.PUNCT:
        if tok.text in TRIGGER_PUNCT:
            # ')' control-заголовка — никогда не триггер
            if tok.text == ")" and tok.is_control_paren_close:
                return False
            return True
        if tok.text == "}":
            return tok.is_value_brace
    return False


def decide_semicolons(tokens: List[Token]) -> Set[int]:
    """Возвращает множество byte-offset'ов, В КОТОРЫХ нужно вставить ';'
    (по одному на "пропущенный" разделитель). Не трогает места, где ';'
    уже стоит явно.

    Важное отличие от Go: внутри незакрытых '(' / '[' / value-'{' (то есть
    посреди списка аргументов вызова, индексации или initializer-list) ';'
    НИКОГДА не вставляется, независимо от того, что за токен перед
    переводом строки. В Go многострочный вызов можно было бы "спасти"
    trailing comma перед закрывающей ')' — но в C trailing comma в списке
    АРГУМЕНТОВ ВЫЗОВА не валиден (в отличие от initializer-list вида
    '{1, 2, 3,}', где он допустим). Поэтому вместо конвенции про запятую
    Masika просто не трогает то, что явно ещё не закрыто — это работает
    без исключений и не требует от пользователя никаких трюков."""

    insert_at: Set[int] = set()
    n = len(tokens)

    # Глубина вложенности expression-контекста в каждой позиции: растёт на
    # '(' / '[' / value-'{', падает на соответствующую закрывающую. Блочная
    # '{' (тело функции/if/while/...) НЕ считается — внутри блока операторы
    # идут последовательно и каждый по-прежнему нуждается в своём ';'.
    depth_before: List[int] = [0] * (n + 1)
    depth = 0
    for idx, t in enumerate(tokens):
        depth_before[idx] = depth
        if t.kind == T.PUNCT and t.text in ("(", "["):
            depth += 1
        elif t.kind == T.PUNCT and t.text in (")", "]"):
            depth = max(0, depth - 1)
        elif t.kind == T.PUNCT and t.text == "{" and t.is_value_brace:
            depth += 1
        elif t.kind == T.PUNCT and t.text == "}" and t.is_value_brace:
            depth = max(0, depth - 1)
    depth_before[n] = depth

    for idx, tok in enumerate(tokens):
        if tok.kind != T.NEWLINE:
            continue

        if depth_before[idx] > 0:
            continue  # внутри незакрытых (), [] или value-{} — не трогаем

        # находим предыдущий реальный токен
        j = idx - 1
        while j >= 0 and tokens[j].kind == T.NEWLINE:
            j -= 1
        if j < 0:
            continue
        prev = tokens[j]

        if prev.kind == T.PUNCT and prev.text == ";":
            continue  # уже явно поставлено
        if prev.kind == T.PP:
            continue  # директива препроцессора сама себе разделитель

        if not _is_trigger(prev):
            continue

        # смотрим следующий реальный токен (может быть за несколько NEWLINE)
        k = idx + 1
        while k < n and tokens[k].kind in (T.NEWLINE,):
            k += 1
        nxt = tokens[k] if k < n else None

        if nxt is not None and nxt.kind == T.PUNCT and nxt.text == "{":
            # Allman-style: 'if (x)\n{' / 'int main()\n{' / голый блок,
            # который относится к предыдущей строке — не разделяем.
            continue
        if nxt is not None and nxt.kind == T.KEYWORD and nxt.text in ("else",):
            continue
        if nxt is not None and nxt.kind == T.KEYWORD and nxt.text == "while" and prev.kind == T.PUNCT and prev.text == "}":
            continue  # do { ... } while(...)

        insert_at.add(prev.end)

    return insert_at
