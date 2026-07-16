# -*- coding: utf-8 -*-
"""
gccrun.py — запуск настоящего (немодифицированного) gcc как внешнего
инструмента: gcc используется ЧЁРНЫМ ЯЩИКОМ через CLI, ровно как его
использовал бы Makefile — никаких патчей в исходники GCC. Это и проще,
и не тянет за собой GPLv3 для Masika (не модифицируем и не линкуемся
с GCC, просто вызываем бинарник).

Здесь же — "анализ архитектуры": Masika не пишет свой статический анализатор
с нуля, а прогоняет gcc c -fopt-info-vec/-Winline и разбирает его же
диагностику (GCC и так умеет объяснять, что не смог векторизовать/заинлайнить).
"""

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional


# Флаг максимальной оптимизации — ровно тот набор, что задал пользователь.
OPTIMIZATION_FLAGS = [
"-Ofast",
"-march=native",
"-flto=auto", 
"-fuse-linker-plugin",
"-fomit-frame-pointer",
"-fno-plt",
"-fprofile-correction",
"-fgraphite-identity",
"-floop-nest-optimize",
"-fipa-pta", 
"-fipa-cp-clone", 
"-fipa-bit-cp",
"-fsched-pressure", 
"-pipe", 
"-s",
]

# Для анализа архитектуры (-fopt-info-vec/-Winline) нужен ОТДЕЛЬНЫЙ набор
# флагов без '-flto=auto': при LTO gcc откладывает векторизацию и большую
# часть инлайнинга на этап финальной линковки, и обычный проход 'gcc -c'
# в таком режиме не печатает вообще никакой -fopt-info диагностики — не
# потому что оптимизировать нечего, а потому что этот этап ещё не наступил.
# Итоговая сборка (compile_final) по-прежнему использует -flto=auto как
# указал пользователь; здесь мы жертвуем точным соответствием финальным
# флагам ради того, чтобы анализ вообще что-то показывал.
ANALYSIS_FLAGS = [f for f in OPTIMIZATION_FLAGS if f != "-flto=auto"]


@dataclass
class BuildResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    binary_path: Optional[str] = None
    cmd: List[str] = field(default_factory=list)


def find_gcc() -> str:
    return os.environ.get("MASIKA_GCC", "gcc")


def compile_c(c_source_path: str, output_path: str, extra_link_flags: List[str],
              extra_user_flags: List[str], optimize: bool = True) -> BuildResult:
    gcc = find_gcc()
    cmd = [gcc, c_source_path, "-o", output_path]
    if optimize:
        cmd += OPTIMIZATION_FLAGS
    cmd += extra_link_flags
    cmd += extra_user_flags

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        return BuildResult(ok=False, returncode=127, stdout="",
                            stderr=f"не найден компилятор '{gcc}'. Убедитесь, что gcc установлен и доступен в PATH.",
                            cmd=cmd)
    except subprocess.TimeoutExpired:
        return BuildResult(ok=False, returncode=-1, stdout="", stderr="gcc не ответил за 300 секунд (timeout)", cmd=cmd)

    ok = proc.returncode == 0
    return BuildResult(ok=ok, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
                        binary_path=output_path if ok else None, cmd=cmd)


# ---------------------------------------------------------------------------
# "Анализ архитектуры": прогоняем gcc ещё раз (компиляция в объектный файл,
# без записи финального бинарника) с флагами диагностики оптимизатора и
# разбираем её же вывод в компактную сводку.
# ---------------------------------------------------------------------------

_VEC_OPTIMIZED_RE = re.compile(r'^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s*(?:optimized|note):\s*(?P<msg>.*loop vectorized.*)$', re.IGNORECASE)
_VEC_MISSED_RE = re.compile(r'^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s*missed:\s*(?P<msg>.*)$')
_INLINE_RE = re.compile(r'^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s*note:\s*(?P<msg>.*inlin\w*.*)$', re.IGNORECASE)
_NOISE_MISSED_RE = re.compile(r'clobbers memory', re.IGNORECASE)


@dataclass
class ArchInsight:
    vectorized_count: int
    missed_vec: List[str]
    inline_notes: List[str]
    raw_stderr: str


def analyze_architecture(c_source_path: str, extra_link_flags: List[str],
                          extra_user_flags: List[str], max_items: int = 6) -> ArchInsight:
    gcc = find_gcc()
    with tempfile.TemporaryDirectory() as td:
        obj_path = os.path.join(td, "arch_probe.o")
        cmd = [gcc, "-c", c_source_path, "-o", obj_path] + ANALYSIS_FLAGS + [
            "-fopt-info-vec-optimized", "-fopt-info-vec-missed", "-Winline",
        ] + extra_link_flags + extra_user_flags
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return ArchInsight(0, [], [], "")

    lines = proc.stderr.splitlines()
    vec_ok = 0
    missed = []
    inl = []
    for ln in lines:
        if _VEC_OPTIMIZED_RE.match(ln):
            vec_ok += 1
        elif "missed" in ln.lower() and ":" in ln:
            m = _VEC_MISSED_RE.match(ln)
            if m:
                msg = m.group("msg").strip()
                if _NOISE_MISSED_RE.search(msg):
                    continue  # шум уровня "clobbers memory" на обычных вызовах, не о циклах
                missed.append(f"{m.group('file')}:{m.group('line')}: {msg}")
        elif _INLINE_RE.match(ln):
            m = _INLINE_RE.match(ln)
            inl.append(f"{m.group('file')}:{m.group('line')}: {m.group('msg').strip()}")

    return ArchInsight(vectorized_count=vec_ok, missed_vec=missed[:max_items],
                        inline_notes=inl[:max_items], raw_stderr=proc.stderr)


def format_arch_summary(insight: ArchInsight) -> str:
    parts = []
    parts.append(f"Векторизовано циклов: {insight.vectorized_count}")
    if insight.missed_vec:
        parts.append("Пропущенные возможности векторизации:")
        for m in insight.missed_vec:
            parts.append(f"  - {m}")
    if insight.inline_notes:
        parts.append("Заметки об инлайне:")
        for m in insight.inline_notes:
            parts.append(f"  - {m}")
    if not insight.missed_vec and not insight.inline_notes and insight.vectorized_count == 0:
        parts.append("(gcc не сообщил ни о векторизации, ни о пропущенных возможностях — "
                      "вероятно, в коде нет явных числовых циклов для анализа)")
    return "\n".join(parts)
