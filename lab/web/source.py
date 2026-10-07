"""Pull the source of a function or class out of a file *without importing it*.

Uses ``ast`` so it works for files that need torch (harness/, recover_job.py)
even where torch isn't installed. A reference looks like
``"lab/train.py:KD.__call__"`` (path relative to the repo root).
"""

from __future__ import annotations

import ast
import os
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@lru_cache(maxsize=None)
def _parse(path: str):
    with open(os.path.join(ROOT, path)) as f:
        text = f.read()
    return text.splitlines(), ast.parse(text)


def get_source(ref: str) -> dict:
    path, _, qual = ref.partition(":")
    lines, tree = _parse(path)
    node = tree
    for part in qual.split("."):
        found = None
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                    and child.name == part:
                found = child
                break
        if found is None:
            raise KeyError(f"{ref}: no {part!r} in {path}")
        node = found
    start = min([d.lineno for d in node.decorator_list] + [node.lineno])
    src = "\n".join(lines[start - 1:node.end_lineno])
    return dict(ref=ref, file=path, name=qual, line=start, end_line=node.end_lineno,
                source=src, doc=ast.get_docstring(node) or "")
