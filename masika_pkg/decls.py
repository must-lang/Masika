# -*- coding: utf-8 -*-
"""
decls.py — обход дерева токенов, разрешение 'allme' и учёт обычных
C-объявлений (чтобы 'allme y = x + 1' знало тип x, даже если x объявлен
как обычный 'int x').

Главная функция — resolve_declarations(tokens, semicolon_offsets) -> (edits, warnings).

Порядок в пайплайне: lex -> asi.annotate_structure -> asi.decide_semicolons
-> decls.resolve_declarations (этот модуль, использует результат ASI, чтобы
корректно находить конец инициализатора, даже если пользователь не поставил
явный ';') -> includes -> применение edits к исходному тексту.
"""

from typing import List, Dict, Optional, Set, Tuple
from .lexer import Token, T
from .typesys import (
    CType, ExprEval, widen, T_UNKNOWN, T_I32, T_I64, T_U64, T_BOOL, T_STR,
    T_VOIDPTR, TYPE_KEYWORDS, DECL_QUALIFIERS, normalize_builtin_type_text,
)


# ---------------------------------------------------------------------------
# Эвристика по имени переменной — используется, только если ни инициализатор,
# ни дальнейшие присваивания не дали никакого сигнала о типе.
# ---------------------------------------------------------------------------

_PREFIX_HINTS = [
    (("is_", "has_", "can_", "should_", "was_", "did_"), T_BOOL),
    (("p_", "ptr_"), T_VOIDPTR),
]
_SUFFIX_HINTS = [
    (("_flag", "_ok", "_done", "_ready", "_valid"), T_BOOL),
    (("_count", "_len", "_length", "_size", "_idx", "_index", "_num", "_qty"), CType("size_t", "int", 8, True)),
    (("_ptr", "_p"), T_VOIDPTR),
    (("_str", "_name", "_msg", "_text", "_label"), T_STR),
]


def _name_heuristic(name: str) -> Optional[CType]:
    low = name.lower()
    for prefixes, ctype in _PREFIX_HINTS:
        if low.startswith(prefixes):
            return ctype
    for suffixes, ctype in _SUFFIX_HINTS:
        if low.endswith(suffixes):
            return ctype
    return None


# ---------------------------------------------------------------------------
# Служебный обход срезов токенов
# ---------------------------------------------------------------------------

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


def _skip_nl(tokens: List[Token], idx: int) -> int:
    while idx < len(tokens) and tokens[idx].kind == T.NEWLINE:
        idx += 1
    return idx


def _split_top_level_commas(toks: List[Token]) -> List[List[Token]]:
    """Делит срез токенов на части по запятым глубины 0 (для списков
    инициализаторов и деклараторов)."""
    parts = []
    depth = 0
    start = 0
    real = [t for t in toks if t.kind != T.NEWLINE]
    for i, t in enumerate(real):
        if t.kind == T.PUNCT and t.text in ("(", "[", "{"):
            depth += 1
        elif t.kind == T.PUNCT and t.text in (")", "]", "}"):
            depth -= 1
        elif depth == 0 and t.kind == T.PUNCT and t.text == ",":
            parts.append(real[start:i])
            start = i + 1
    parts.append(real[start:])
    return parts


def _find_stmt_end_from(tokens: List[Token], start_idx: int, semicolon_offsets: Set[int]) -> Tuple[int, int]:
    """От start_idx ищет конец текущего выражения/statement'а на глубине 0.
    Возвращает (last_content_idx, semi_idx). semi_idx == -1, если разделитель
    виртуальный (ASI), а не явный токен ';'."""
    depth = 0
    i = start_idx
    n = len(tokens)
    last_real = start_idx - 1
    while i < n:
        t = tokens[i]
        if t.kind == T.NEWLINE or t.kind == T.PP:
            i += 1
            continue
        if t.kind == T.PUNCT and t.text in ("(", "[", "{"):
            depth += 1
        elif t.kind == T.PUNCT and t.text in (")", "]", "}"):
            depth -= 1
        if depth <= 0:
            if t.kind == T.PUNCT and t.text == ";":
                return last_real, i
            if t.end in semicolon_offsets:
                return i, -1
        last_real = i
        i += 1
    return last_real, -1


def _find_enclosing_scope_end(tokens: List[Token], from_idx: int) -> int:
    """Находит индекс токена '}' (блочного, не value-brace), закрывающего
    БЛИЖАЙШИЙ охватывающий блок, начиная поиск от from_idx. Если охватывающего
    блока нет (глобальная область), возвращает len(tokens)."""
    depth = 0
    i = from_idx
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if t.kind == T.PUNCT and t.text == "{" and not t.is_value_brace:
            depth += 1
        elif t.kind == T.PUNCT and t.text == "}" and not t.is_value_brace:
            if depth == 0:
                return i
            depth -= 1
        i += 1
    return n


def _scan_forward_widen(tokens: List[Token], name: str, from_idx: int, scope_end: int,
                         semicolon_offsets: Set[int], evaluator: ExprEval, current: CType) -> CType:
    """Ищет 'name = expr ;' (простое переприсваивание, НЕ объявление) от
    from_idx до scope_end (включая вложенные блоки) и расширяет current.
    Сознательно не отслеживает тень одноимённой переменной во вложенном
    блоке — в редких случаях так можно взять чуть более широкий тип, чем
    строго необходимо, но никогда не более узкий (это безопасная сторона
    ошибки).

    Использует "тихий" вычислитель (без warn): это забегание ВПЕРЁД по
    токенам, до которых основной цикл resolve_declarations ещё не дошёл,
    поэтому идентификаторы, объявленные чуть ниже (например счётчик 'k' в
    заголовке for, который встретится в теле цикла раньше, чем основной
    цикл до него дойдёт), здесь могут быть ещё не в scope_stack. Для самого
    widen() это безвредно (widen с unknown — no-op), а вот предупреждать
    пользователя об этом не нужно — это ложный сигнал, не его ошибка.

    ВАЖНО про безопасность: помимо явного 'name = expr' и 'name += expr',
    здесь же учитываются СРАВНЕНИЯ вида 'name < expr' / 'expr <= name' и т.п.
    Это критично для типичного паттерна счётчика цикла:

        allme i = 0
        for (i = 0; i < n; i++) { ... }

    Если бы мы смотрели только на присваивания, 'i' получил бы int8_t (из
    начального '0') и никогда не узнал бы, что n может доходить до 100000 —
    в реальном C это переполнение/неопределённое поведение на ровном месте,
    ради 'оптимального типа'. Поэтому: с чем переменную сравнивают, с тем
    она и должна быть способна сравниться — это безопасное расширение."""
    silent_evaluator = ExprEval(evaluator.lookup, lambda msg: None)
    i = from_idx
    n = min(scope_end, len(tokens))
    while i < n:
        t = tokens[i]

        if t.kind == T.IDENT and t.text == name:
            j = _skip_nl(tokens, i + 1)
            if j < n and tokens[j].kind == T.PUNCT and tokens[j].text in _ASSIGN_LIKE_OPS:
                # name = expr  ИЛИ  name += expr  (и другие составные) —
                # в обоих случаях итоговое значение должно уместиться в name
                expr_start = j + 1
                last_idx, semi_idx = _find_stmt_end_from(tokens, expr_start, semicolon_offsets)
                if last_idx >= expr_start:
                    t2 = silent_evaluator.eval(tokens[expr_start:last_idx + 1])
                    current = widen(current, t2)
                i = (semi_idx + 1) if semi_idx != -1 else (last_idx + 1)
                continue
            if j < n and tokens[j].kind == T.PUNCT and tokens[j].text in _COMPARISON_OPS:
                # name < expr  — расширяем name до диапазона, с которым сравниваем
                rhs_end = _operand_end(tokens, j + 1, n)
                if rhs_end > j + 1:
                    t2 = silent_evaluator.eval(tokens[j + 1:rhs_end])
                    current = widen(current, t2)

        elif t.kind == T.PUNCT and t.text in _COMPARISON_OPS:
            j = _skip_nl(tokens, i + 1)
            if j < n and tokens[j].kind == T.IDENT and tokens[j].text == name:
                # expr < name — та же логика, но переменная справа
                lhs_start = _operand_start(tokens, i, from_idx)
                if lhs_start < i:
                    t2 = silent_evaluator.eval(tokens[lhs_start:i])
                    current = widen(current, t2)

        i += 1
    return current


_ASSIGN_LIKE_OPS = {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}
_COMPARISON_OPS = {"<", ">", "<=", ">=", "==", "!="}
_OPERAND_BOUNDARY = {";", ",", "&&", "||", "?", ":"}


def _operand_end(tokens: List[Token], start: int, limit: int) -> int:
    """От start ищет конец 'операнда' сравнения (может включать арифметику:
    'i < n - 1') — до ближайшего сравнения/логического/';'/',' на глубине 0."""
    depth = 0
    i = start
    while i < limit:
        t = tokens[i]
        if t.kind == T.NEWLINE:
            i += 1
            continue
        if t.kind == T.PUNCT and t.text in ("(", "[", "{"):
            depth += 1
        elif t.kind == T.PUNCT and t.text in (")", "]", "}"):
            if depth == 0:
                break
            depth -= 1
        elif depth == 0 and t.kind == T.PUNCT and (t.text in _COMPARISON_OPS or t.text in _OPERAND_BOUNDARY):
            break
        i += 1
    return i


def _operand_start(tokens: List[Token], end_excl: int, floor: int) -> int:
    """Симметрично _operand_end, но назад: от end_excl (не включая) до начала
    операнда, не выходя за floor (начало текущего окна сканирования)."""
    depth = 0
    i = end_excl - 1
    result = end_excl
    while i >= floor:
        t = tokens[i]
        if t.kind == T.NEWLINE:
            i -= 1
            continue
        if t.kind == T.PUNCT and t.text in (")", "]", "}"):
            depth += 1
        elif t.kind == T.PUNCT and t.text in ("(", "[", "{"):
            if depth == 0:
                break
            depth -= 1
        elif depth == 0 and t.kind == T.PUNCT and (t.text in _COMPARISON_OPS or t.text in _OPERAND_BOUNDARY):
            break
        result = i
        i -= 1
    return result


def _looks_like_plain_decl_start(tokens: List[Token], i: int) -> bool:
    t = tokens[i]
    if t.kind != T.KEYWORD:
        return False
    if t.text in TYPE_KEYWORDS or t.text in DECL_QUALIFIERS or t.text in ("struct", "union", "enum"):
        return True
    return False


def _consume_type_words(tokens: List[Token], i: int) -> Tuple[List[str], int, bool]:
    """С позиции i читает подряд идущие ключевые слова типа/квалификаторы
    (+ 'struct/union/enum Tag'), возвращает (список слов, новый индекс,
    was_struct_like). Останавливается перед первым '*' или идентификатором-
    декларатором."""
    words = []
    was_struct = False
    n = len(tokens)
    while i < n:
        i = _skip_nl(tokens, i)
        if i >= n:
            break
        t = tokens[i]
        if t.kind == T.KEYWORD and (t.text in TYPE_KEYWORDS or t.text in DECL_QUALIFIERS):
            words.append(t.text)
            i += 1
            continue
        if t.kind == T.KEYWORD and t.text in ("struct", "union", "enum"):
            was_struct = True
            words.append(t.text)
            i += 1
            i = _skip_nl(tokens, i)
            if i < n and tokens[i].kind == T.IDENT:
                words.append(tokens[i].text)
                i += 1
            break
        break
    return words, i, was_struct


def _handle_plain_decl(tokens: List[Token], i: int, semicolon_offsets: Set[int],
                        scope_stack: List[Dict[str, CType]]) -> Optional[int]:
    words, k, was_struct = _consume_type_words(tokens, i)
    if not words:
        return None
    n = len(tokens)
    k = _skip_nl(tokens, k)
    star_count = 0
    while k < n and tokens[k].kind == T.PUNCT and tokens[k].text == "*":
        star_count += 1
        k = _skip_nl(tokens, k + 1)

    if k >= n or tokens[k].kind != T.IDENT:
        # 'struct Point {' (определение типа) или что-то, что мы не
        # распознаём как декларатор переменной — не наш случай.
        return None
    nxt = _skip_nl(tokens, k + 1)
    if nxt >= n or not (tokens[nxt].kind == T.PUNCT and tokens[nxt].text in ("=", ";", ",", "[", "(")):
        return None
    if nxt < n and tokens[nxt].kind == T.PUNCT and tokens[nxt].text == "(":
        # это объявление/определение ФУНКЦИИ ('int foo(...)'), а не переменной —
        # обычные declaration-правила тут не при чем, просто выходим из режима
        # декларации и даём основному циклу идти дальше как обычно.
        return None

    base = normalize_builtin_type_text(words) if not was_struct else CType(" ".join(words), "opaque", 0, False)
    if base is None:
        base = CType(" ".join(words), "opaque", 0, False)
    if star_count:
        ctype = CType(base.name + " " + "*" * star_count, "ptr", 8, False)
    else:
        ctype = base

    # список деклараторов через запятую: name [= expr] (, name2 [= expr2])*
    pos = k
    while True:
        if pos >= n or tokens[pos].kind != T.IDENT:
            break
        decl_name = tokens[pos].text
        scope_stack[-1][decl_name] = ctype
        after = _skip_nl(tokens, pos + 1)
        if after < n and tokens[after].kind == T.PUNCT and tokens[after].text == "[":
            close_br = _matching_close(tokens, after, "[", "]")
            after = _skip_nl(tokens, close_br + 1)
        if after < n and tokens[after].kind == T.PUNCT and tokens[after].text == "=":
            last_idx, semi_idx = _find_stmt_end_from(tokens, after + 1, semicolon_offsets)
            # ищем ',' или конец на глубине 0 внутри найденного диапазона —
            # но _find_stmt_end_from не останавливается на ',', так что для
            # мульти-деклараторов с инициализатором просто переходим к ';'/ASI
            # и завершаем (в реальном коде 'int a = 1, b = 2;' тоже встречается,
            # но реже; базовый случай 'int a, b, c;' обрабатываем полностью).
            end_pos = (semi_idx + 1) if semi_idx != -1 else (last_idx + 1)
            return end_pos
        if after < n and tokens[after].kind == T.PUNCT and tokens[after].text == ",":
            pos = _skip_nl(tokens, after + 1)
            continue
        if after < n and tokens[after].kind == T.PUNCT and tokens[after].text == ";":
            return after + 1
        if after < n and after in semicolon_offsets:
            return after + 1
        last_idx, semi_idx = _find_stmt_end_from(tokens, after, semicolon_offsets)
        return (semi_idx + 1) if semi_idx != -1 else (last_idx + 1)
    return pos + 1


def _handle_allme_decl(tokens: List[Token], i: int, semicolon_offsets: Set[int],
                        scope_stack: List[Dict[str, CType]], evaluator: ExprEval,
                        warn) -> Tuple[int, List[Tuple[int, int, str]]]:
    n = len(tokens)
    allme_tok = tokens[i]
    j = _skip_nl(tokens, i + 1)
    if j >= n or tokens[j].kind != T.IDENT:
        return i + 1, []
    name_tok = tokens[j]
    name = name_tok.text
    k = _skip_nl(tokens, j + 1)

    is_array = False
    bracket_open_idx = -1
    array_len_known = None
    if k < n and tokens[k].kind == T.PUNCT and tokens[k].text == "[":
        is_array = True
        bracket_open_idx = k
        close_br = _matching_close(tokens, k, "[", "]")
        inner = [t for t in tokens[k + 1:close_br] if t.kind != T.NEWLINE]
        if inner:
            array_len_known = "".join(t.text for t in inner)
        k = _skip_nl(tokens, close_br + 1)

    has_init = k < n and tokens[k].kind == T.PUNCT and tokens[k].text == "="
    result_type = None
    stmt_end_idx = None
    elem_count = None

    if has_init:
        init_start = _skip_nl(tokens, k + 1)
        if init_start < n and tokens[init_start].kind == T.PUNCT and tokens[init_start].text == "{":
            close_brace = _matching_close(tokens, init_start, "{", "}")
            parts = _split_top_level_commas(tokens[init_start + 1:close_brace])
            parts = [p for p in parts if p]
            elem_count = len(parts)
            for part in parts:
                result_type = widen(result_type, evaluator.eval(part))
            last_idx, semi_idx = _find_stmt_end_from(tokens, close_brace, semicolon_offsets)
        else:
            last_idx, semi_idx = _find_stmt_end_from(tokens, init_start, semicolon_offsets)
            result_type = evaluator.eval(tokens[init_start:last_idx + 1])
        stmt_end_idx = (semi_idx + 1) if semi_idx != -1 else (last_idx + 1)
    else:
        last_idx, semi_idx = _find_stmt_end_from(tokens, k, semicolon_offsets)
        stmt_end_idx = (semi_idx + 1) if semi_idx != -1 else max(last_idx + 1, k)

    if not is_array:
        scope_stack[-1][name] = result_type if result_type is not None else T_UNKNOWN
        scope_end = _find_enclosing_scope_end(tokens, i)
        result_type = _scan_forward_widen(tokens, name, stmt_end_idx, scope_end,
                                           semicolon_offsets, evaluator, result_type)

    if result_type is None or result_type.family == "unknown":
        heuristic = _name_heuristic(name)
        if heuristic is not None:
            result_type = heuristic
        else:
            warn(f"тип для 'allme {name}' не удалось определить — использую int32_t по умолчанию "
                 f"(строка {allme_tok.line})")
            result_type = T_I32

    scope_stack[-1][name] = result_type

    edits = [(allme_tok.start, allme_tok.end, result_type.name)]
    if is_array and bracket_open_idx != -1 and array_len_known is None and elem_count is not None:
        insert_at = tokens[bracket_open_idx].end
        edits.append((insert_at, insert_at, str(elem_count)))

    return stmt_end_idx, edits


def resolve_declarations(tokens: List[Token], semicolon_offsets: Set[int]):
    """Главная точка входа. Возвращает (edits, warnings)."""
    edits: List[Tuple[int, int, str]] = []
    warnings: List[str] = []
    scope_stack: List[Dict[str, CType]] = [dict()]
    struct_body_stack: List[bool] = []
    n = len(tokens)
    i = 0

    def lookup(nm: str) -> Optional[CType]:
        for scope in reversed(scope_stack):
            if nm in scope:
                return scope[nm]
        return None

    def do_warn(msg: str):
        warnings.append(msg)

    evaluator = ExprEval(lookup, do_warn)

    while i < n:
        t = tokens[i]

        if t.kind == T.NEWLINE or t.kind == T.PP:
            i += 1
            continue

        if t.kind == T.PUNCT and t.text == "{":
            is_struct_body = False
            j = i - 1
            while j >= 0 and tokens[j].kind == T.NEWLINE:
                j -= 1
            if j >= 0:
                if tokens[j].kind == T.IDENT:
                    j2 = j - 1
                    while j2 >= 0 and tokens[j2].kind == T.NEWLINE:
                        j2 -= 1
                    if j2 >= 0 and tokens[j2].kind == T.KEYWORD and tokens[j2].text in ("struct", "union", "enum"):
                        is_struct_body = True
                elif tokens[j].kind == T.KEYWORD and tokens[j].text in ("struct", "union", "enum"):
                    is_struct_body = True
            if not t.is_value_brace:
                scope_stack.append(dict())
            struct_body_stack.append(is_struct_body)
            i += 1
            continue

        if t.kind == T.PUNCT and t.text == "}":
            if not t.is_value_brace and len(scope_stack) > 1:
                scope_stack.pop()
            if struct_body_stack:
                struct_body_stack.pop()
            i += 1
            continue

        in_struct_body = bool(struct_body_stack) and struct_body_stack[-1]

        if t.kind == T.KEYWORD and t.text == "allme" and not in_struct_body:
            new_i, decl_edits = _handle_allme_decl(tokens, i, semicolon_offsets, scope_stack, evaluator, do_warn)
            edits.extend(decl_edits)
            i = max(new_i, i + 1)
            continue

        if not in_struct_body and _looks_like_plain_decl_start(tokens, i):
            new_i = _handle_plain_decl(tokens, i, semicolon_offsets, scope_stack)
            if new_i is not None:
                i = max(new_i, i + 1)
                continue

        i += 1

    return edits, warnings
