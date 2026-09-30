#!/usr/bin/env python3
"""Static device audit: catch 'indexing a CPU tensor with a GPU index' before submit.

WHY THIS EXISTS
---------------
Job 581629 died at the 32-second mark with
    RuntimeError: indices should be either on cpu or on the same device as the
    indexed tensor (cpu)
on `tgt_all_t[obj]`, where `tgt_all_t` is a CPU-resident constant bank and `obj`
had been moved to the GPU by the batch loop.

The submit script's CPU smoke test had run that EXACT code path successfully one
minute earlier.  It could not have failed: on a CPU-only node there is no second
device, so the whole class of bug is invisible to it.  Adding more CPU smoke does
not help.  What helps is checking the invariant directly, which is what this does.

THE RULE
--------
A tensor created by `torch.from_numpy(...)` lives on the CPU.  If it is later
indexed, the index must also be on the CPU, so the subscript must go through
`.cpu()` / `.numpy()` / a numpy array, or the whole expression must be moved to the
device first.  Anything else is flagged.

Escape hatch for a genuine false positive: put `# device-audit: ok` on the line.

Exit code 0 = clean, 1 = findings.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# creators on the torch namespace that yield a CPU tensor unless a device is given
CPU_CREATORS = {"from_numpy", "tensor", "zeros", "ones", "empty", "full", "arange",
                "eye", "randn", "rand", "randint", "linspace", "as_tensor"}

# dtype casts that DO NOT move a tensor between devices.  Getting this wrong is a
# silent hole in the audit: `torch.from_numpy(x).float()` is still CPU-resident.
NON_MOVING_ATTRS = {"float", "double", "half", "long", "int", "short", "byte",
                    "bool", "bfloat16", "contiguous", "clone", "detach", "requires_grad_"}

# substrings that prove the index is CPU-side (or not a tensor index at all)
CPU_PROOF = (".cpu(", ".numpy(", "np.", "numpy", "tolist()", "reshape(", "arange(")
MARKER = "device-audit: ok"


def _is_cpu_bank(node: ast.AST) -> bool:
    """True if this expression yields a CPU-resident tensor.

    Walks THROUGH non-moving method calls, because
        torch.from_numpy(x).float()
    is still a CPU tensor -- the cast does not move it.  Without the walk that
    chain is not recognised as a bank at all, and the audit silently misses the
    device bug sitting right behind it.
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in NON_MOVING_ATTRS:
            return _is_cpu_bank(node.func.value)
        return (node.func.attr in CPU_CREATORS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and not any(k.arg == "device" for k in node.keywords))
    return False


def _to_moves_device(call: ast.Call) -> bool:
    """True only for `.to(<device>)` / `.to(<"cuda…">)` / `.to(torch.device(...))`.

    `.to(torch.float32)` is a dtype cast and leaves the tensor on the CPU, so it
    must NOT count as a move -- otherwise the audit would clear the very bug it
    was written to catch.

    `.to(<tensor>.device)` counts: it is the idiomatic way to graft a freshly built
    module onto a device-resident one, which is the fix the graft rule points at.
    """
    if not call.args:
        return False
    a = call.args[0]
    if isinstance(a, ast.Constant) and isinstance(a.value, str):
        return True
    if isinstance(a, ast.Call) and isinstance(a.func, ast.Attribute):
        return a.func.attr == "device"
    if isinstance(a, ast.Attribute):
        return a.attr == "device"            # e.g. `.to(base.weight.device)`
    return isinstance(a, ast.Name)          # a device variable


def _moved_to_device(node: ast.AST) -> bool:
    """Walk the method chain; `.cuda()` or a device-taking `.to(...)` moves it."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        attr = node.func.attr
        if attr == "cuda":
            return True
        if attr == "to":
            return _to_moves_device(node)
        if attr in NON_MOVING_ATTRS:
            return _moved_to_device(node.func.value)
        return False
    return False


def _graft_findings(path: Path, src: str, lines: list[str], tree: ast.AST) -> list[str]:
    """Catch 'build a module on the CPU, then graft it onto a device-resident model'.

    WHY THIS RULE EXISTS
    --------------------
    Job 581657 died in its fifth arm with
        RuntimeError: Expected all tensors to be on the same device, but got mat2 is
        on cpu, different from other tensors on cuda:0
    in `LoRALinear.forward`, because that class created its parameters with
    `nn.Parameter(torch.empty(...))` (CPU) and `inject_lora` then attached it to an
    encoder that had ALREADY been moved to the GPU with `setattr`.  Unlike
    `Proj(...).to(dev)`, a `setattr` graft has no chained `.to(...)`, so nothing ever
    moves those parameters, and no CPU smoke test can see it: on a CPU-only node the
    grafted module is on the same device as everything else and the code runs fine.

    The first device bug (581629) was `tensor[gpu_index]` on a CPU bank.  This is a
    different mechanism with the same signature -- invisible to CPU smoke, fatal on
    GPU -- so it needs its own rule rather than a variation of the first.

    SCOPE: only `setattr(target, name, Ctor(...))` where the third argument is a call
    to something whose name is capitalised -- i.e. a class construction.  Two things
    are deliberately NOT flagged:
      * `self.x = Ctor(...)` inside `__init__`, because there the whole module is still
        on the CPU and the caller's `Class(...).to(dev)` moves everything together;
      * lowercase callables such as `setattr(cfg, k, getattr(args, k))` (a namespace
        copy in cfmsf_route_probe.py), which is not a module graft at all.
    Both exclusions matter: this audit is run as a preflight inside the long GPU jobs,
    so a false positive does not merely annoy -- it aborts a running experiment.  A
    rule that flags legitimate code would get switched off, and then it protects
    nothing.

    Escape hatch: `# device-audit: ok` on the line.
    """
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "setattr"):
            continue
        if len(node.args) != 3:
            continue
        new_val = node.args[2]
        if not isinstance(new_val, ast.Call):
            continue
        if _moved_to_device(new_val):
            continue        # `setattr(m, n, Ctor(...).to(dev))` has been fixed
        ctor = new_val.func
        ctor_name = (ctor.id if isinstance(ctor, ast.Name)
                     else ctor.attr if isinstance(ctor, ast.Attribute) else "")
        if not ctor_name[:1].isupper():
            continue        # not a class construction; not a graft
        ln = node.lineno
        seg = ast.get_source_segment(src, new_val) or "?"
        line_text = lines[ln - 1] if 0 <= ln - 1 < len(lines) else ""
        if MARKER in line_text:
            continue
        findings.append(
            f"{path}:{ln}: `setattr(...)` grafts `{seg}` onto an existing module. "
            f"Any parameters it creates land on the CPU while the target is on the "
            f"GPU -- inherit device/dtype from the wrapped tensor inside the class, "
            f"or move the new module explicitly (`... .to(<tensor>.device)`)")
    return findings


def audit(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src)
    cpu_banks: set[str] = set()
    exempt: set[str] = set()
    findings: list[str] = []

    for node in ast.walk(tree):
        # gather CPU-resident banks (both plain assign and annotated assign)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if value is not None and _is_cpu_bank(value) and not _moved_to_device(value):
                for t in targets:
                    if isinstance(t, ast.Name):
                        cpu_banks.add(t.id)
                        # An exemption marker written anywhere on the DECLARING line
                        # applies to that bank.  It has to be keyed to the bank, not
                        # to the subscript line: the device error is raised at the
                        # index site, but the intent is usually documented where the
                        # bank is declared, and requiring the marker at every use
                        # site would just get copy-pasted around.
                        if MARKER in lines[t.lineno - 1]:
                            exempt.add(t.id)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not (isinstance(node.value, ast.Name) and node.value.id in cpu_banks):
            continue
        if node.value.id in exempt:
            continue
        idx = node.slice
        idx_src = ast.get_source_segment(src, idx) or ""
        # a literal or a CPU-proved index is fine
        if isinstance(idx, ast.Constant):
            continue
        if any(p in idx_src for p in CPU_PROOF):
            continue
        ln = node.lineno
        line_text = lines[ln - 1] if 0 <= ln - 1 < len(lines) else ""
        if MARKER in line_text or MARKER in idx_src:
            continue
        findings.append(
            f"{path}:{ln}: CPU-resident bank `{node.value.id}` indexed by a "
            f"device-dependent expression `{idx_src}` -- needs .cpu() on the index "
            f"or .to(device) on the tensor")
    findings += _graft_findings(path, src, lines, tree)
    return findings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--selftest", action="store_true",
                    help="also prove the audit catches the exact 581629 bug")
    args = ap.parse_args()

    bad: list[str] = []
    for f in args.files:
        p = Path(f)
        if not p.is_file():
            print(f"[device-audit] MISSING {p}")
            bad.append(f"{p} missing")
            continue
        found = audit(p)
        print(f"[device-audit] {p}: {'CLEAN' if not found else f'{len(found)} finding(s)'}")
        bad += found

    if args.selftest:
        import tempfile
        # Case 1: the EXACT bug from job 581629, plus two decoys that a naive audit
        # would mishandle: a dtype cast (.float()) must NOT count as a device move,
        # and a .cpu() index must be accepted.
        cases = {
            "581629 bug (device index into CPU bank)": (
                "import torch\n"
                "tgt = torch.from_numpy(numpy_bank)      # CPU\n"
                "for b in loader:\n"
                "    obj = b[4].to(dev)\n"
                "    y = tgt[obj]\n", True),
            "dtype cast is not a move": (
                "import torch\n"
                "tgt = torch.from_numpy(numpy_bank).float()\n"
                "for b in loader:\n"
                "    obj = b[4].to(dev)\n"
                "    y = tgt[obj]\n", True),
            "fixed form is accepted": (
                "import torch\n"
                "tgt = torch.from_numpy(numpy_bank)\n"
                "for b in loader:\n"
                "    obj = b[4].to(dev)\n"
                "    y = tgt[obj.cpu()].to(dev)\n", False),
            "bank moved to device is accepted": (
                "import torch\n"
                "tgt = torch.from_numpy(numpy_bank).to(dev)\n"
                "for b in loader:\n"
                "    obj = b[4].to(dev)\n"
                "    y = tgt[obj]\n", False),
            "explicit marker is accepted": (
                "import torch\n"
                "tgt = torch.from_numpy(numpy_bank)  # device-audit: ok (numpy index)\n"
                "y = tgt[rows_np]\n", False),
            # --- the 581657 bug: CPU params grafted onto a device-resident model ---
            "581657 bug (setattr grafts a CPU-built module)": (
                "import torch\n"
                "class Ada(torch.nn.Module):\n"
                "    def __init__(self, base):\n"
                "        super().__init__()\n"
                "        self.A = torch.nn.Parameter(torch.empty(8, base.in_features))\n"
                "def inject(m):\n"
                "    setattr(m, 'lin', Ada(m.lin))\n", True),
            "graft fixed with an explicit move is accepted": (
                "import torch\n"
                "def inject(m):\n"
                "    setattr(m, 'lin', Ada(m.lin).to(m.lin.weight.device))\n", False),
            "plain constructor inside __init__ is not a graft": (
                "import torch\n"
                "class M(torch.nn.Module):\n"
                "    def __init__(self, d):\n"
                "        super().__init__()\n"
                "        self.lin = torch.nn.Linear(d, d)\n"
                "    def forward(self, x):\n"
                "        return self.lin(x)\n", False),
        }
        with tempfile.TemporaryDirectory() as d:
            for label, (code, should_flag) in cases.items():
                f = Path(d) / "case.py"
                f.write_text(code, encoding="utf-8")
                got = len(audit(f))
                ok = (got > 0) if should_flag else (got == 0)
                print(f"[device-audit][selftest] {'PASS' if ok else 'FAIL'}  {label}: "
                      f"{got} finding(s), expected {'>=1' if should_flag else '0'}")
                if not ok:
                    bad.append(f"[selftest] {label}: {got} findings, "
                               f"expected {'>0' if should_flag else '0'}")

    for b in bad:
        print(f"  [FAIL] {b}")
    if bad:
        print("[device-audit] FAILED")
        sys.exit(1)
    print("[device-audit] OK")


if __name__ == "__main__":
    main()
