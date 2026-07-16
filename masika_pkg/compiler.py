# -*- coding: utf-8 -*-
"""
compiler.py — оркестрация транспиляции Must -> C.

Пайплайн:
    исходный текст .mu
        -> lex()                         (lexer.py)
        -> annotate_structure()          (asi.py)      — размечает скобки
        -> decide_semicolons()           (asi.py)      — где вставить ';'
        -> resolve_declarations()        (decls.py)    — allme -> реальный тип
        -> compute_includes()            (includes.py) — какие #include нужны
        -> применение всех правок к ИСХОДНОМУ тексту (хирургически, без
           пересборки токенов — поэтому все пробелы/комментарии/форматирование
           пользователя остаются как есть)
        -> итоговый C-текст с '#line 1 "файл.mu"', чтобы ошибки/предупреждения
           gcc указывали на строки .mu, а не на сгенерированный .c
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from .lexer import lex, LexError
from .asi import annotate_structure, decide_semicolons
from .decls import resolve_declarations
from .includes import compute_includes


@dataclass
class TranspileResult:
    c_source: str
    link_flags: List[str]
    warnings: List[str]
    ok: bool = True
    error: str = ""


HEADER_BANNER = "/* сгенерировано Masika — не редактировать вручную, редактируйте .mu */\n"


def transpile(src: str, original_filename: str) -> TranspileResult:
    try:
        tokens = lex(src)
    except LexError as e:
        return TranspileResult(c_source="", link_flags=[], warnings=[], ok=False,
                                error=f"{original_filename}:{e.line}:{e.col}: ошибка лексера: {e}")

    annotate_structure(tokens)
    semicolon_offsets = decide_semicolons(tokens)

    try:
        decl_edits, decl_warnings = resolve_declarations(tokens, semicolon_offsets)
    except Exception as e:  # noqa: BLE001 — не даём внутренней ошибке уронить весь процесс без диагноза
        return TranspileResult(c_source="", link_flags=[], warnings=[], ok=False,
                                error=f"{original_filename}: внутренняя ошибка разрешения allme: {e!r}")

    resolved_type_names = [repl for (_, _, repl) in decl_edits]
    headers, link_flags = compute_includes(tokens, resolved_type_names)

    all_edits: List[Tuple[int, int, str]] = list(decl_edits)
    all_edits += [(off, off, ";") for off in semicolon_offsets]

    body = _apply_edits(src, all_edits)

    header_lines = [HEADER_BANNER]
    for h in headers:
        header_lines.append(f"#include {h}\n")
    header_lines.append("\n")
    header_lines.append(f'#line 1 "{original_filename}"\n')
    c_source = "".join(header_lines) + body

    return TranspileResult(c_source=c_source, link_flags=link_flags, warnings=decl_warnings, ok=True)


def _apply_edits(src: str, edits: List[Tuple[int, int, str]]) -> str:
    """Применяет список (start, end, replacement) к исходному тексту.
    Правки не пересекаются по построению (ASI вставляет только в 'зазорах'
    сразу после токена, allme-правки — ровно в границах самого токена
    'allme' или сразу после '[' для массивов), поэтому достаточно применить
    их справа налево по смещению, чтобы более ранние смещения не съехали."""
    out = src
    for start, end, repl in sorted(edits, key=lambda e: (-e[0], -e[1])):
        out = out[:start] + repl + out[end:]
    return out
