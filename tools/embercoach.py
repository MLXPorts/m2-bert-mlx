#!/usr/bin/env python
"""
EmberCoach — Teaching linter for numerical precision in GPU computing

WHY THIS EXISTS:
Running 300M ops/sec for 24 hours = 25.9 trillion operations. Each float32 operation
introduces ~6e-8 relative error. Even "tiny" differences compound. This tool teaches
you where precision breaks and how to fix it.

WHAT IT TEACHES:
- FFT normalization (exactly one 1/n across rfft/irfft pairs)
- Python scalar hygiene (why float()/int() break graphs and add rounding)
- Device/dtype consistency (avoid CPU hops that add extra rounding)
- When you need extended-precision kernels
- Backend tensor operations (torch.add vs +, mx.multiply vs *)

DOCUMENTATION:
- Deep dive: docs/NUMERICAL_PRECISION_GUIDE.md
- Findings: docs/NUMERIC_STABILITY_TORCH_vs_MLX.md

Usage:
  python tools/embercoach.py <file-or-dir> [more files/dirs]

Exit code 1 if errors found (strict enforcement of precision rules).
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Finding:
    path: pathlib.Path
    line: int
    kind: str  # 'teach' | 'warn' | 'error'
    code: str  # e.g. FFT-NORM-001
    msg: str
    why: str  # The deeper explanation


# Teaching messages with WHY context
TEACHINGS = {
    'FFT-NORM-001': {
        'title': 'FFT Normalization Contract',
        'why': """
The convolution theorem requires EXACTLY ONE 1/N factor across forward/inverse:
- PyTorch norm='forward': forward divides by N, inverse doesn't
- PyTorch norm='backward' (default): forward doesn't, inverse divides by N
- MLX default: forward doesn't, inverse divides by N

Applying 1/N twice (or zero times) gives order-one amplitude errors.
Over billions of ops, even small normalization mistakes compound.

See: docs/NUMERICAL_PRECISION_GUIDE.md § "FFT Normalization"
""",
        'fix_torch': """
If using irfft(norm='forward'):
    k_f = torch.fft.rfft(k, n=2*L) / (2*L)  # divide spectrum once
    y = torch.fft.irfft(u_f * k_f, n=2*L, norm='forward')  # no inverse scaling

If using irfft() default:
    k_f = torch.fft.rfft(k, n=2*L)  # don't divide
    y = torch.fft.irfft(u_f * k_f, n=2*L)  # inverse applies 1/N
""",
        'fix_mlx': """
MLX default (inverse scales by 1/N):
    k_f = mx.fft.rfft(k, n=2*L)  # don't divide
    y = mx.fft.irfft(u_f * k_f, n=2*L)  # inverse applies 1/N

Never mix conventions or you get double/missing scaling!
"""
    },

    'PYTHON-SCALAR': {
        'title': 'Python Scalars Break Precision',
        'why': """
When you write `y = x * 0.5` where x is a tensor:
1. Python 0.5 is float64
2. Framework either:
   - Promotes x to float64 (expensive, wrong precision)
   - Demotes 0.5 to float32 (adds a rounding step)
   - Breaks lazy graph and computes immediately
3. Result: TWO roundings instead of ONE

Over 25 trillion ops (300M/sec × 24hr), these extra roundings accumulate.
Also breaks Metal buffer links in MLX (destroys gradient tracking).

See: docs/NUMERICAL_PRECISION_GUIDE.md § "Python Scalars in Tensor Math"
""",
        'fix_torch': """
Create device-bound scalars:
    half = torch.tensor(0.5, dtype=torch.float32, device=x.device)
    y = torch.mul(x, half)  # or x * half if you must use operator

Use backend ops:
    torch.add(), torch.multiply(), torch.divide()
    (not Python +, *, /)
""",
        'fix_mlx': """
Create typed MLX scalars:
    half = mx.array(0.5, dtype=mx.float32)
    y = mx.multiply(x, half)

Use backend ops:
    mx.add(), mx.multiply(), mx.divide(), mx.power()
    (not Python +, *, /, **)
"""
    },

    'ITEM-NUMPY': {
        'title': 'Graph Breaks Destroy Precision',
        'why': """
When you call .item(), .numpy(), float(), or int() on a tensor:
1. Forces GPU→CPU copy
2. Evaluates lazy graph immediately (loses fusion opportunities)
3. Destroys Metal buffer link (MLX can't track gradients)
4. Adds host rounding: tensor → Python scalar → back to tensor
5. Cannot be fused with subsequent ops

Each round-trip adds noise. Over billions of ops, this compounds to visible drift.

See: docs/NUMERICAL_PRECISION_GUIDE.md § ".item(), .numpy(), float(), int() Conversions"
""",
        'fix_torch': """
Keep values as tensors:
    # Bad:
    if similarity.item() > threshold:

    # Good:
    threshold_t = torch.tensor(threshold, dtype=torch.float32, device=similarity.device)
    if torch.greater(similarity, threshold_t):

Store tensors in metadata, not Python scalars.
""",
        'fix_mlx': """
Keep values as MLX arrays:
    # Bad:
    if float(similarity) > threshold:

    # Good:
    threshold_a = mx.array(threshold, dtype=mx.float32)
    if mx.greater(similarity, threshold_a):

MLX is lazy — breaking that loses the entire optimization pipeline.
"""
    },

    'NUMPY-FFT': {
        'title': 'NumPy FFT Silently Promotes to float64',
        'why': """
numpy.fft.rfft/irfft ALWAYS promotes float32 → float64 and runs on CPU.
Then rounds back to float32 when you convert to GPU tensor.

This means:
1. Wrong precision path (float64 when you want float32)
2. CPU hop (destroys GPU pipeline)
3. Extra rounding on the way back
4. Impossible to fuse with GPU ops

NumPy is fine for final comparisons (tests), but NEVER in compute paths.

See: docs/NUMERIC_STABILITY_TORCH_vs_MLX.md § "NumPy Promotion"
""",
        'fix': """
Replace numpy.fft with backend FFT:
    # Bad:
    y_np = np.fft.rfft(x_np)

    # Good (PyTorch):
    y = torch.fft.rfft(x)

    # Good (MLX):
    y = mx.fft.rfft(x)

NumPy acceptable ONLY for final test comparisons (not compute).
"""
    },

    'DEVICE-HOP': {
        'title': 'Device Hops Add Extra Rounding',
        'why': """
When tensors live on different devices (CPU/GPU) or you call .cpu()/.to() mid-graph:
1. Framework copies data between devices
2. May change memory layout (stride/padding)
3. Each copy can introduce new rounding
4. Loses kernel fusion opportunities
5. Creates implicit synchronization points

Keep ALL tensors on ONE device through hot computation paths.

See: docs/NUMERICAL_PRECISION_GUIDE.md § "Device Hops and Hidden Copies"
""",
        'fix': """
Create all constants on target device from the start:
    # PyTorch:
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    x = torch.tensor(data, dtype=torch.float32, device=device)
    bias = torch.tensor(0.1, dtype=torch.float32, device=device)

    # MLX (defaults to GPU):
    x = mx.array(data, dtype=mx.float32)
    bias = mx.array(0.1, dtype=mx.float32)

Avoid .cpu()/.to() unless absolutely necessary (e.g., final logging).
"""
    },

    'EXTENDED-PRECISION': {
        'title': 'When You Need Extended Precision Kernels',
        'why': """
Standard float32 pipeline rounds at EVERY operation:
    Input → FFT (round) → Multiply (round) → IFFT (round) → Bias (round) → Output

Four rounding points compound errors. For long-running workloads (billions of ops),
use extended precision (double-double) to round ONCE at the end:
    Input → upcast to dd → FFT (dd) → Multiply (dd) → IFFT (dd) → round ONCE → Output

Apple M-series GPUs don't have native float64, but we can emulate ~32 decimal digits
using two float32 values (hi, lo) with error-free transforms.

See: docs/NUMERICAL_PRECISION_GUIDE.md § "The Solution: Extended Precision"
""",
        'when': """
Consider extended precision when:
- FFT-based convolutions (frequency-domain multiply most sensitive)
- Depthwise convolutions (small kernels, many iterations)
- Long reductions/accumulations
- Running >10^9 operations on same state
- Need bit-reproducibility across runs

Available via HyperProfile flags:
    ep_freqmul=true    # dd complex multiply (cheap, high leverage)
    ep_depthwise=true  # dd depthwise conv (cheap)
    ep_fft=true        # dd butterfly (heavier, for extreme stability)
""",
        'kernels': """
Precision kernels available in:
    experimental/metal_bitexact/ComplexMul.metal  (strict FP32, will extend to dd)
    experimental/metal_bitexact/Depthwise3.metal  (deterministic 3-tap)

To use:
    1. Set HyperProfile flag (e.g., ep_freqmul=true in profiles/torch_like.json)
    2. Extended-precision kernels replace standard ops automatically
    3. No code changes needed beyond profile selection
"""
    },

    'MLX-METAL-KERNEL': {
        'title': 'MLX Metal Kernel Best Practices',
        'why': """
MLX's mx.fast.metal_kernel() provides direct Metal GPU access, but has specific requirements:

1. JIT generates function signatures - provide BODY only (no signature)
2. Grid/threadgroup must use MLX scalars in tuples (not Python ints)
3. Kernel source must be inline Python string (can't load .metal files directly)
4. Cache compiled kernels globally (compilation is expensive)
5. Complex64 support is limited on Metal GPU (many ops CPU-only)

Common mistakes break lazy execution and destroy buffer tracking.

See: .claude/CLAUDE.md § "MLX Metal Kernel Development Guide"
""",
        'correct_pattern': """
# CORRECT Metal kernel pattern:

_KERNEL = None
_HEADER = '''#include <metal_stdlib>
using namespace metal;
'''

_SOURCE = r'''
    // Kernel BODY only - no function signature
    uint tid = thread_position_in_grid.x;
    if (tid >= n) return;
    out[tid] = inp[tid] * 2.0f;
'''

def my_operation(x):
    global _KERNEL

    batch, length = x.shape

    # Compile once on first use
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="my_kernel",
            input_names=["params", "x"],
            output_names=["y"],
            header=_HEADER,
            source=_SOURCE
        )

    # Params buffer (all dims as mx.array)
    params = mx.array([batch, length], dtype=mx.uint32)

    # Grid/threadgroup with MLX scalars in tuples
    total_mx = mx.array(batch * length, dtype=mx.uint32)
    tpg = mx.array(256, dtype=mx.uint32)
    one = mx.array(1, dtype=mx.uint32)
    num_groups = mx.divide(
        mx.add(total_mx, mx.subtract(tpg, one)),
        tpg
    )
    grid = (num_groups, one, one)
    threadgroup = (tpg, one, one)

    # Dispatch
    (y,) = _KERNEL(
        inputs=[params, x],
        output_shapes=[(batch, length)],
        output_dtypes=[mx.float32],
        grid=grid,
        threadgroup=threadgroup
    )
    return y
""",
        'common_mistakes': """
# WRONG - Loading .metal file with full signature:
source = open("kernel.metal").read()  # Has function signature - will fail

# WRONG - Python scalars in grid:
grid = ((total + 255) // 256, 1, 1)  # Python ints break buffer links

# WRONG - Not caching compilation:
kernel = mx.fast.metal_kernel(...)  # Compiles EVERY call - very slow

# WRONG - Trying complex64 ops (many are CPU-only):
# Just use mx.multiply for complex - custom kernels won't help
"""
    }
}


class Coach(ast.NodeVisitor):
    def __init__(self, src: str, path: pathlib.Path) -> None:
        self.src = src
        self.path = path
        self.alias_torch: Optional[str] = None
        self.alias_mx: Optional[str] = None
        self.alias_np: Optional[str] = None
        self.findings: List[Finding] = []
        self.stack: List[ast.AST] = []

        # Track usage patterns to give deeper advice
        self.has_fft_rfft = False
        self.has_fft_irfft = False
        self.has_complex_multiply = False
        self.python_scalar_count = 0

    def _add(self, node: ast.AST, kind: str, code: str, msg: str, why: str = '') -> None:
        self.findings.append(Finding(self.path, getattr(node, 'lineno', 1), kind, code, msg, why))

    def _is_name(self, node: ast.AST, name: str) -> bool:
        return isinstance(node, ast.Name) and node.id == name

    def _matches_fft(self, node: ast.Call, which: str) -> bool:
        f = node.func
        if isinstance(f, ast.Attribute):
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

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == 'torch':
                self.alias_torch = alias.asname or 'torch'
            elif alias.name in ('mlx.core', 'mlx'):
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
        return any(isinstance(n, ast.Subscript) for n in self.stack)

    def _backend_hint(self) -> str:
        if self.alias_mx and not self.alias_torch:
            return TEACHINGS['PYTHON-SCALAR']['fix_mlx']
        if self.alias_torch and not self.alias_mx:
            return TEACHINGS['PYTHON-SCALAR']['fix_torch']
        return "Use backend tensor scalars (torch.tensor or mx.array) and backend math ops."

    def visit_Call(self, node: ast.Call) -> None:
        # Track FFT usage for final summary
        if self._matches_fft(node, 'rfft'):
            self.has_fft_rfft = True
            lib = 'torch' if self.alias_torch else ('mlx' if self.alias_mx else 'lib')
            fix = TEACHINGS['FFT-NORM-001']['fix_torch'] if self.alias_torch else TEACHINGS['FFT-NORM-001']['fix_mlx']
            self._add(
                node, 'teach', 'FFT-NORM-001',
                f"{lib}.fft.rfft detected - normalization check required",
                TEACHINGS['FFT-NORM-001']['why'] + '\n' + fix
            )

        if self._matches_fft(node, 'irfft'):
            self.has_fft_irfft = True
            norm = self._get_kw(node, 'norm')
            norm_val = None
            if isinstance(norm, ast.Constant) and isinstance(norm.value, str):
                norm_val = norm.value

            msg = "irfft normalization: "
            if norm_val == 'forward':
                msg += "norm='forward' means inverse does NOT scale. Ensure rfft path divides spectrum by n."
            elif norm_val in ('backward', None):
                msg += "default/backward means inverse scales by 1/n. Do NOT pre-divide spectrum."
            else:
                msg += "verify normalization contract (exactly one 1/n across pair)."

            self._add(node, 'teach', 'FFT-NORM-002', msg, TEACHINGS['FFT-NORM-001']['why'])

        # MLX Metal kernel detection
        if (isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)):
            base = node.func.value.value.id
            mid = node.func.value.attr
            attr = node.func.attr
            if self.alias_mx and base == self.alias_mx and mid == 'fast' and attr == 'metal_kernel':
                # Provide teaching on correct Metal kernel pattern
                self._add(
                    node, 'teach', 'MLX-METAL-KERNEL-001',
                    "mx.fast.metal_kernel detected - ensure correct usage pattern",
                    TEACHINGS['MLX-METAL-KERNEL']['why'] + '\n' +
                    TEACHINGS['MLX-METAL-KERNEL']['correct_pattern'] + '\n' +
                    TEACHINGS['MLX-METAL-KERNEL']['common_mistakes']
                )

        # NumPy FFT
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            base = node.func.value.id
            attr = node.func.attr
            if self.alias_np and base == self.alias_np and attr in ('fft', 'rfft', 'irfft', 'fftn', 'irfftn'):
                self._add(
                    node, 'error', 'NUMPY-FFT-001',
                    "numpy.fft promotes float32->float64 and runs on CPU",
                    TEACHINGS['NUMPY-FFT']['why'] + '\n' + TEACHINGS['NUMPY-FFT']['fix']
                )

            # Python numerics passed to backend ops
            if base in ((self.alias_torch or ''), (self.alias_mx or '')) and attr in (
                'add','subtract','multiply','divide','power','pow','matmul','einsum'
            ):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)) and not self._inside_indexing():
                        self.python_scalar_count += 1
                        self._add(
                            node, 'error', 'PYTHON-SCALAR-001',
                            f"Python numeric literal passed to {base}.{attr}",
                            TEACHINGS['PYTHON-SCALAR']['why'] + '\n' + self._backend_hint()
                        )

        # .item() / .numpy() / float()/int()
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ('item', 'numpy'):
                self._add(
                    node, 'error', 'GRAPH-BREAK-001',
                    f".{node.func.attr}() breaks lazy graph, forces GPU->CPU copy",
                    TEACHINGS['ITEM-NUMPY']['why'] + '\n' +
                    (TEACHINGS['ITEM-NUMPY']['fix_torch'] if self.alias_torch else TEACHINGS['ITEM-NUMPY']['fix_mlx'])
                )
            elif node.func.attr in ('cpu', 'to'):
                self._add(
                    node, 'warn', 'DEVICE-HOP-001',
                    f"Device hop (.{node.func.attr}) adds extra rounding",
                    TEACHINGS['DEVICE-HOP']['why'] + '\n' + TEACHINGS['DEVICE-HOP']['fix']
                )

        # float()/int() casts
        if isinstance(node.func, ast.Name) and node.func.id in ('float', 'int'):
            self._add(
                node, 'error', 'CAST-001',
                f"Python {node.func.id}() cast breaks graph and adds host rounding",
                TEACHINGS['ITEM-NUMPY']['why'] + '\n' +
                (TEACHINGS['ITEM-NUMPY']['fix_torch'] if self.alias_torch else TEACHINGS['ITEM-NUMPY']['fix_mlx'])
            )

        # SCALAR-DTYPE-REQUIRED: mx.array(number) must specify dtype explicitly
        if (self.alias_mx and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == self.alias_mx and node.func.attr == 'array'):
            has_dtype = any((isinstance(kw, ast.keyword) and kw.arg == 'dtype') for kw in (node.keywords or []))
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, (int, float)) and not has_dtype:
                self._add(
                    node, 'error', 'SCALAR-DTYPE-REQUIRED',
                    'mx.array used with Python numeric without dtype=... (use mx.float32/mx.uint32 etc.)',
                    'Always create typed MLX scalars: mx.array(0.5, dtype=mx.float32), mx.array(1, dtype=mx.uint32).'
                )

        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        lit = (isinstance(node.left, ast.Constant) and isinstance(node.left.value, (int, float))) or \
              (isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)))
        if lit and not self._inside_indexing():
            self.python_scalar_count += 1
            self._add(
                node, 'error', 'PYTHON-SCALAR-002',
                "Python numeric in tensor expression (use backend ops)",
                TEACHINGS['PYTHON-SCALAR']['why'] + '\n' + self._backend_hint()
            )
        self.generic_visit(node)

    def _report_assign_const(self, value: ast.AST, node: ast.AST) -> None:
        if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
            hint = self._backend_hint()
            if self.alias_mx and isinstance(value.value, float):
                hint = "Use mx.array(0.1, dtype=mx.float32) instead of bare float."
            self._add(
                node, 'error', 'ASSIGN-SCALAR-001',
                "Bare Python scalar in assignment",
                TEACHINGS['PYTHON-SCALAR']['why'] + '\n' + hint
            )

    def visit_Assign(self, node: ast.Assign) -> None:
        self._report_assign_const(node.value, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._report_assign_const(node.value, node)
        self.generic_visit(node)


def scan_path(p: pathlib.Path) -> Tuple[List[Finding], Dict[str, int], bool]:
    """Returns (findings, usage_stats, path_exists)"""
    findings: List[Finding] = []
    files: List[pathlib.Path] = []
    stats = {'files': 0, 'has_fft': 0, 'python_scalars': 0}

    if not p.exists():
        return findings, stats, False

    if p.is_dir():
        files = [x for x in p.rglob('*.py')]
    elif p.suffix == '.py':
        files = [p]

    # Increase recursion limit for deeply nested AST structures
    # (e.g., long chains of binary operations like a + b + c + d + ...)
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(10000, old_limit))

    try:
        for f in files:
            try:
                src = f.read_text(encoding='utf-8')
            except Exception:
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue

            stats['files'] += 1
            c = Coach(src, f)
            c.visit(tree)
            findings.extend(c.findings)

            if c.has_fft_rfft or c.has_fft_irfft:
                stats['has_fft'] += 1
            stats['python_scalars'] += c.python_scalar_count
    finally:
        # Restore original recursion limit
        sys.setrecursionlimit(old_limit)

    return findings, stats, True


def print_summary(stats: Dict[str, int], findings: List[Finding]):
    """Print educational summary based on what was found"""
    print("\n" + "="*80)
    print("EMBERCOACH SUMMARY")
    print("="*80)
    print(f"Scanned {stats['files']} files")
    print(f"Found {len(findings)} teaching moments")

    errors = sum(1 for f in findings if f.kind == 'error')
    warns = sum(1 for f in findings if f.kind == 'warn')
    teaches = sum(1 for f in findings if f.kind == 'teach')

    print(f"  ERRORS: {errors} (strict precision violations)")
    print(f"  WARNINGS: {warns} (drift risks)")
    print(f"  TEACHING: {teaches} (good practices)")

    if stats['has_fft'] > 0:
        print(f"\nFFT USAGE: Detected in {stats['has_fft']} files")
        print("   Consider extended-precision kernels for long-running workloads:")
        print("   " + TEACHINGS['EXTENDED-PRECISION']['when'])

    if stats['python_scalars'] > 0:
        print(f"\nPYTHON SCALARS: {stats['python_scalars']} usage patterns detected")
        print("   At 300M ops/sec × 24hr, these extra roundings compound significantly.")
        print("   Review: docs/NUMERICAL_PRECISION_GUIDE.md § 'Python Scalars'")

    print("\nDOCUMENTATION:")
    print("   docs/NUMERICAL_PRECISION_GUIDE.md  (comprehensive tutorial)")
    print("   docs/NUMERIC_STABILITY_TORCH_vs_MLX.md  (findings summary)")
    print("="*80 + "\n")


def main():
    ap = argparse.ArgumentParser(
        description='EmberCoach: Teaching linter for GPU numerical precision',
        epilog='Teaches WHY precision matters and HOW to fix issues. See docs/ for details.'
    )
    ap.add_argument('paths', nargs='+', help='Files or directories to scan')
    ap.add_argument('--verbose', '-v', action='store_true', help='Show full WHY context for each finding')
    args = ap.parse_args()

    all_findings = []
    combined_stats = {'files': 0, 'has_fft': 0, 'python_scalars': 0}
    invalid_paths = []

    for p in args.paths:
        path_obj = pathlib.Path(p)
        findings, stats, exists = scan_path(path_obj)
        if not exists:
            invalid_paths.append(p)
            continue
        all_findings.extend(findings)
        for k in combined_stats:
            combined_stats[k] += stats[k]

    # Report invalid paths
    if invalid_paths:
        print("❌ Error: The following paths do not exist:")
        for p in invalid_paths:
            print(f"   {p}")
        print()
        if not all_findings and combined_stats['files'] == 0:
            raise SystemExit(1)

    # Print findings
    for f in all_findings:
        kind_label = f.kind.upper()
        print(f"{f.path}:{f.line}: [{kind_label} {f.code}] {f.msg}")
        if args.verbose and f.why:
            print(f"   WHY: {f.why}")
            print()

    # Summary
    if all_findings:
        print_summary(combined_stats, all_findings)
    elif combined_stats['files'] > 0:
        print("✅ EmberCoach: No precision issues found — excellent!")

    # Exit code
    errors = sum(1 for f in all_findings if f.kind == 'error')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
