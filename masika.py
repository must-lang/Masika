#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
masika.py — точка входа компилятора Masika для языка Must.

    masika файл.mu                 собрать в бинарник рядом с исходником
    masika файл.mu -o вывод        собрать с явным именем бинарника
    masika файл.mu --emit-c        показать сгенерированный C и выйти
    masika файл.mu --no-optimize   собрать без агрессивных флагов (для отладки)
    masika файл.mu --no-analyze    не запускать анализ архитектуры после сборки
    masika файл.mu -v              подробный вывод (в т.ч. точная команда gcc)

Masika принимает только файлы с расширением '.mu' — это явный маркер, что
файл написан на Must, а не на обычном C. Файл с любым другим расширением
(или без него) отклоняется с понятной ошибкой, ещё до попытки разбора.

Всё, что не распознано как флаг Masika (напр. '-lm', '-pthread'), передаётся
дальше в gcc как есть — совсем как при обычном 'gcc файл.c -какой-то-флаг'.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from masika_pkg.compiler import transpile          # noqa: E402
from masika_pkg.gccrun import compile_c, analyze_architecture, format_arch_summary  # noqa: E402

__version__ = "0.1.0"
REQUIRED_EXT = ".mu"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="masika",
        description="Masika — компилятор языка Must (C + allme + авто-';' + авто-include + максимальная оптимизация)",
    )
    p.add_argument("source", help="исходный файл .mu")
    p.add_argument("-o", "--output", help="имя выходного бинарника (по умолчанию — имя файла без .mu)")
    p.add_argument("--emit-c", action="store_true",
                   help="только показать сгенерированный C и выйти, не компилировать")
    p.add_argument("--save-c", metavar="PATH",
                   help="сохранить сгенерированный C по указанному пути (по умолчанию — рядом с исходником, *.masika.c)")
    p.add_argument("--no-optimize", action="store_true",
                   help="собрать без агрессивных флагов оптимизации (быстрая сборка для отладки)")
    p.add_argument("--no-analyze", action="store_true",
                   help="не запускать анализ архитектуры (векторизация/инлайн) после сборки")
    p.add_argument("-v", "--verbose", action="store_true", help="подробный вывод, включая точную команду gcc")
    p.add_argument("--version", action="version", version=f"masika {__version__}")
    return p


def main(argv=None) -> int:
    parser = build_arg_parser()
    args, extra_gcc_args = parser.parse_known_args(argv)

    if not os.path.isfile(args.source):
        print(f"masika: файл не найден: {args.source}", file=sys.stderr)
        return 1

    root, ext = os.path.splitext(args.source)
    if ext.lower() != REQUIRED_EXT:
        shown = ext if ext else "(расширения нет)"
        print(
            f"masika: неверное расширение файла: '{shown}' — "
            f"Masika компилирует только файлы с расширением '{REQUIRED_EXT}'",
            file=sys.stderr,
        )
        return 1

    with open(args.source, encoding="utf-8") as f:
        src = f.read()

    result = transpile(src, args.source)
    if not result.ok:
        print(f"masika: {result.error}", file=sys.stderr)
        return 1

    for w in result.warnings:
        print(f"masika: предупреждение: {w}", file=sys.stderr)

    default_c_path = root + ".masika.c"
    c_path = args.save_c or default_c_path
    with open(c_path, "w", encoding="utf-8") as f:
        f.write(result.c_source)

    if args.emit_c:
        print(result.c_source)
        return 0

    if args.verbose:
        print(f"masika: сгенерированный C сохранён в {c_path}")
        if result.link_flags:
            print(f"masika: автоматически добавлены флаги линковки: {' '.join(result.link_flags)}")

    output = args.output or root

    build = compile_c(c_path, output, result.link_flags, extra_gcc_args, optimize=not args.no_optimize)

    if args.verbose:
        print("masika: команда gcc:", " ".join(build.cmd))

    if build.stderr:
        sys.stderr.write(build.stderr)
        if not build.stderr.endswith("\n"):
            sys.stderr.write("\n")

    if not build.ok:
        print(f"masika: сборка не удалась (код возврата gcc: {build.returncode})", file=sys.stderr)
        return build.returncode or 1

    print(f"masika: готово -> {output}")

    if not args.no_analyze and not args.no_optimize:
        insight = analyze_architecture(c_path, result.link_flags, extra_gcc_args)
        print("\n--- анализ архитектуры (векторизация/инлайн, gcc -fopt-info) ---")
        print(format_arch_summary(insight))

    return 0


if __name__ == "__main__":
    sys.exit(main())
