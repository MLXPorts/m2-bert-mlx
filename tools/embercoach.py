#!/usr/bin/env python
"""
EmberCoach — numerics-aware teaching linter for MLX / PyTorch FFT & tensor math.

What it does
- Detects uses of torch.fft / mlx.fft (rfft/irfft) and teaches correct normalization
  (exactly one 1/n), and TIME vs FREQ combine consistency.
- Warns against NumPy FFT in compute paths (float64 promotion, CPU hop).
- Flags Python-scalar arithmetic on tensors and explains how to fix (use lib ops,
  typed device scalars).
- Points out common precision pitfalls: .item(), .numpy(), device/dtype hops.

Usage
  python tools/embercoach.py <file-or-dir> [more files/dirs]

Exit code is always 0 (teaching mode). Prints actionable guidance with file:line.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Finding:
    path: pathlib.Path
    line: int
    kind: str  # 'tip' | 'warn' | 'error'
    code: str  # short code, e.g. FFT-NORM-001
    msg: str


class Coach(ast.NodeVisitor):
    def __init__(self, src: str, path: pathlib.Path) -> None:
        self.src = src
        self.path = path
        self.alias_torch: Optional[str] = None
        self.alias_mx: Optional[str] = None
        self.alias_np: Optional[str] = None
        self.findings: List[Finding] = []
        self.stack: List[ast.AST] = []

    # --- helpers ---
    def _add(self, node: ast.AST, kind: str, code: str, msg: str) -> None:
        self.findings.append(Finding(self.path, getattr(node, 'lineno', 1), kind, code, msg))

    def _is_name(self, node: ast.AST, name: str) -> bool:
        return isinstance(node, ast.Name) and node.id == name

    def _is_attr_chain(self, node: ast.AST, *parts: str) -> bool:
        cur = node
        for p in reversed(parts):
            if isinstance(cur, ast.Attribute) and cur.attr == p:
                cur = cur.value
            elif isinstance(cur, ast.Name) and cur.id == p and p == parts[0]:
                return True
            else:
                return False
        return isinstance(cur, ast.Name)

    def _matches_fft(self, node: ast.Call, which: str) -> bool:
        f = node.func
        # torch.fft.rfft / irfft, mlx.fft.rfft / irfft
        if isinstance(f, ast.Attribute):
            # torch.fft.rfft
            if (isinstance(f.value, ast.Attribute)
                and isinstance(f.value.value, ast.Name)
                and f.attr == which
                and ((self.alias_torch and f.value.value.id == self.alias_torch and f.value.attr == 'fft')
                     or (self.alias_mx and f.value.value.id == self.alias_mx and f.value.attr == 'fft'))):
                return True
        return False

    def _get_kw(self, node: ast.Call, name: str) -> Optional[ast.AST]:
        for kw in node.keywords:
            if kw.arg == name:
                return kw.value
        return None

    # --- visitors ---
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == 'torch':
                self.alias_torch = alias.asname or 'torch'
            elif alias.name in ('mlx.core', 'mlx'):  # tolerate both
                self.alias_mx = alias.asname or ('mx' if alias.name == 'mlx.core' else 'mlx')
            elif alias.name == 'numpy':
                self.alias_np = alias.asname or 'np'
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ''
        if mod.startswith('torch'):
            self.alias_torch = 'torch'
        elif mod.startswith('mlx'):
            self.alias_mx = 'mx'
        elif mod == 'numpy':
            self.alias_np = 'np'
        self.generic_visit(node)

    def visit(self, node: ast.AST):
        self.stack.append(node)
        super().visit(node)
        self.stack.pop()

    def _inside_indexing(self) -> bool:
        # Consider it indexing if any ancestor is a Subscript (x[...]) and
        # current node lies within that subtree (approximate via presence in stack).
        return any(isinstance(n, ast.Subscript) for n in self.stack)

    def _backend_hint(self) -> str:
        if self.alias_mx and not self.alias_torch:
            return "Use mx.array(..., dtype=mx.float32) (or device-bound scalar) and mx.add/mx.multiply/mx.divide."
        if self.alias_torch and not self.alias_mx:
            return "Use torch.tensor(..., dtype=torch.float32, device=<device>) and torch.add/mul/div."
        return "Use backend tensor scalars (torch.tensor or mx.array) and backend math ops (add/mul/div)."

    def visit_Call(self, node: ast.Call) -> None:
        # 1) FFT coach
        if self._matches_fft(node, 'rfft'):
            lib = 'torch' if self.alias_torch else ('mlx' if self.alias_mx else 'lib')
            self._add(
                node, 'tip', 'FFT-NORM-001',
                f"{lib}.fft.rfft detected. Ensure exactly one 1/n across rfft/irfft. "
                f"If you later call irfft(norm='forward'), divide the spectrum by n here; "
                f"else keep rfft unscaled and let irfft apply 1/n."
            )
        if self._matches_fft(node, 'irfft'):
            norm = self._get_kw(node, 'norm')
            if isinstance(norm, ast.Constant) and isinstance(norm.value, str):
                if norm.value == 'forward':
                    self._add(
                        node, 'tip', 'FFT-NORM-002',
                        "irfft(norm='forward') means no scaling on the inverse. Make sure your rfft path "
                        "applies 1/n to the spectrum exactly once (e.g., divide rfft(k) by n)."
                    )
                elif norm.value in ('backward', None):
                    self._add(
                        node, 'tip', 'FFT-NORM-003',
                        "irfft with default/backward applies 1/n on inverse. Do NOT pre-divide the spectrum or you will double-scale."
                    )
        # 2) NumPy FFT in compute paths
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            base = node.func.value.id
            attr = node.func.attr
            if self.alias_np and base == self.alias_np and attr in ('fft', 'rfft', 'irfft', 'fftn', 'irfftn'):
                self._add(
                    node, 'warn', 'FFT-DTYPE-001',
                    "numpy.fft promotes float32→float64 and runs on CPU. Avoid in compute paths; use torch.fft or mlx.fft to stay float32 on device."
                )
            # Passing Python numerics into backend math ops → strict error
            if base in ((self.alias_torch or ''), (self.alias_mx or '')) and attr in (
                'add','subtract','multiply','divide','power','pow','tanh','sigmoid','gelu','erf','exp','log','maximum','minimum',
                'sin','cos','sqrt','rsqrt','relu','silu','softmax','matmul','einsum','pad','roll','stack','concatenate','clip','where'
            ):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)) and not self._inside_indexing():
                        self._add(
                            node, 'error', 'OPS-CALL-SCALAR-STRICT',
                            f"Python numeric literal passed to {base}.{attr}. {self._backend_hint()}"
                        )
        # 3) .item() / .numpy() / .cpu()
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'item':
                self._add(node, 'warn', 'NUM-ITEM-001', 
                          ".item() pulls a scalar to Python, forcing a round+host hop; avoid in compute graphs.")
            elif node.func.attr == 'numpy':
                self._add(node, 'warn', 'NUM-NP-002', 
                          ".numpy() moves data to CPU and float64 land; avoid mid-graph.")
            elif node.func.attr in ('cpu', 'to'):
                # Heuristic: device hop mid-graph
                self._add(node, 'tip', 'DEV-HOP-001', 
                          "Device hop detected. Extra copies add rounding; keep tensors on a single device through FFT/matmul.")
        # 4) float()/int() wraps (Python scalar casts)
        if isinstance(node.func, ast.Name) and node.func.id in ('float', 'int'):
            if self.alias_torch:
                self._add(
                    node, 'error', 'NUM-CAST-STRICT',
                    "Python cast float()/int() on a tensor breaks the graph, moves to CPU, and rounds to Python scalar. "
                    "Keep values as torch tensors; if you need a constant, use torch.tensor(..., dtype=torch.float32, device=<device>)."
                )
            elif self.alias_mx:
                self._add(
                    node, 'error', 'NUM-CAST-STRICT',
                    "Python cast float()/int() on an MLX array breaks lazy execution and rounds on host. "
                    "Prefer mx.array(..., dtype=mx.float32) for constants and keep values as tensors in math ops."
                )
            else:
                self._add(
                    node, 'error', 'NUM-CAST-STRICT',
                    "Avoid float()/int() on tensors in compute paths; keep backend tensors and use dtype conversions on device."
                )
        # 5) Dunder conversions (__array__, __float__, __int__, __index__)
        if isinstance(node.func, ast.Attribute) and node.func.attr in ('__array__', '__float__', '__int__', '__index__'):
            self._add(
                node, 'warn', 'NUM-DUNDER-001',
                f"Dunder conversion {node.func.attr}() triggers a host/dtype conversion; avoid in hot paths. Use backend tensor ops instead."
            )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        # Python-scalar arithmetic on tensors (teaching version)
        # Heuristic: literal numeric on one side, Name/Call/Attribute on the other side.
        lit = isinstance(node.left, ast.Constant) and isinstance(node.left.value, (int, float)) or \
              isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float))
        if lit:
            if not self._inside_indexing():
                self._add(
                    node, 'error', 'OPS-SCALAR-STRICT',
                    f"Python numeric in tensor math. {self._backend_hint()}"
                )
        self.generic_visit(node)

    def _report_assign_const(self, value: ast.AST, node: ast.AST) -> None:
        if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
            # Strict: error unless inside indexing (assignment never inside slice normally)
            hint = self._backend_hint()
            # MLX specific nasty hint for floats
            if self.alias_mx and isinstance(value.value, float):
                hint = "Use mx.array(0.1, dtype=mx.float32) (or your float) instead of bare Python float."
            self._add(node, 'error', 'ASSIGN-SCALAR-STRICT', f"Bare Python scalar assignment detected. {hint}")
        elif isinstance(value, (ast.Tuple, ast.List)):
            # If any element is numeric constant, flag (we can tune later)
            for elt in value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, (int, float)):
                    self._add(node, 'error', 'ASSIGN-SCALAR-STRICT', f"Bare Python numeric in assignment literal. {self._backend_hint()}")
                    break

    def visit_Assign(self, node: ast.Assign) -> None:
        self._report_assign_const(node.value, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._report_assign_const(node.value, node)
        self.generic_visit(node)


def scan_path(p: pathlib.Path) -> List[Finding]:
    out: List[Finding] = []
    files: List[pathlib.Path] = []
    if p.is_dir():
        files = [x for x in p.rglob('*.py')]
    elif p.suffix == '.py':
        files = [p]
    for f in files:
        try:
            src = f.read_text(encoding='utf-8')
        except Exception:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        c = Coach(src, f)
        c.visit(tree)
        out.extend(c.findings)
    return out


def main():
    ap = argparse.ArgumentParser(description='EmberCoach numerics-aware teaching linter')
    ap.add_argument('paths', nargs='+', help='Files or directories to scan')
    args = ap.parse_args()

    total = 0
    errors = 0
    for p in args.paths:
        for f in scan_path(pathlib.Path(p)):
            total += 1
            if f.kind == 'error':
                errors += 1
            print(f"{f.path}:{f.line}: [{f.kind} {f.code}] {f.msg}")
    if total == 0:
        print('EmberCoach: no teaching tips — looking good!')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
