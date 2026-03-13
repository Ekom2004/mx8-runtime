import ast
import functools
import inspect
import os
import textwrap
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_coordinator = _pkg_dir / "mx8d-coordinator"
if not _coordinator.is_file():
    _coordinator = _pkg_dir / "mx8d-coordinator.exe"
if _coordinator.is_file():
    os.environ.setdefault("MX8_COORDINATOR_BIN", str(_coordinator))

from .mx8 import *  # noqa: F401,F403


def _flatten_add(expr):
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        return _flatten_add(expr.left) + _flatten_add(expr.right)
    return [expr]


def _segments_from_return_expr(expr, param_name):
    if isinstance(expr, ast.Constant) and expr.value is None:
        return None

    terms = _flatten_add(expr)
    segments = []
    for term in terms:
        if isinstance(term, ast.Name) and term.id == param_name:
            segments.append(("input", b""))
            continue
        if isinstance(term, ast.Constant) and isinstance(term.value, (bytes, bytearray)):
            segments.append(("literal", bytes(term.value)))
            continue
        raise ValueError(
            "@mx8.transform only supports bytes expressions composed of the input sample and bytes literals"
        )
    if not segments:
        raise ValueError("@mx8.transform return expression cannot be empty")
    return segments


def _compile_segments_to_wat(segments):
    if segments is None:
        body = "i64.const -1"
    else:
        instr = [
            "(local $cursor i32)",
            "local.get $out_ptr",
            "local.set $cursor",
        ]
        for kind, payload in segments:
            if kind == "input":
                instr.extend(
                    [
                        "local.get $cursor",
                        "local.get $in_ptr",
                        "local.get $in_len",
                        "memory.copy",
                        "local.get $cursor",
                        "local.get $in_len",
                        "i32.add",
                        "local.set $cursor",
                    ]
                )
                continue
            for b in payload:
                instr.extend(
                    [
                        "local.get $cursor",
                        f"i32.const {int(b)}",
                        "i32.store8",
                        "local.get $cursor",
                        "i32.const 1",
                        "i32.add",
                        "local.set $cursor",
                    ]
                )

        instr.extend(
            [
                "local.get $cursor",
                "local.get $out_ptr",
                "i32.sub",
                "i64.extend_i32_u",
            ]
        )
        body = "\n      ".join(instr)

    return f"""(module
  (memory (export \"memory\") 1)
  (func (export \"transform\") (param $in_ptr i32) (param $in_len i32) (param $out_ptr i32) (result i64)
      {body}
  )
)
"""


def _build_compile_hook(user_fn):
    try:
        source = textwrap.dedent(inspect.getsource(user_fn))
    except (OSError, TypeError) as exc:
        raise ValueError(
            "@mx8.transform could not read function source; define the transform in a module file"
        ) from exc
    module = ast.parse(source)
    fn_def = None
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == user_fn.__name__:
            fn_def = node
            break
    if fn_def is None:
        raise ValueError("@mx8.transform could not locate function AST")

    if fn_def.decorator_list and len(fn_def.decorator_list) > 1:
        raise ValueError("@mx8.transform function should only use @mx8.transform")

    args = fn_def.args
    if args.vararg is not None or args.kwarg is not None or args.kwonlyargs:
        raise ValueError("@mx8.transform only supports a single positional parameter")
    if len(args.args) != 1:
        raise ValueError("@mx8.transform function must accept exactly one argument")

    if len(fn_def.body) != 1 or not isinstance(fn_def.body[0], ast.Return):
        raise ValueError("@mx8.transform body must be a single return statement")

    param_name = args.args[0].arg
    segments = _segments_from_return_expr(fn_def.body[0].value, param_name)
    wat = _compile_segments_to_wat(segments)

    def _mx8_compile():
        return wat

    return _mx8_compile


def transform(fn=None):
    def _decorate(user_fn):
        if not callable(user_fn):
            raise TypeError("@mx8.transform expects a callable")

        code = getattr(user_fn, "__code__", None)
        if code is None:
            raise TypeError("@mx8.transform requires a Python function")

        blocked_names = {
            "open",
            "exec",
            "eval",
            "compile",
            "random",
            "time",
            "datetime",
            "socket",
            "subprocess",
            "os",
            "sys",
            "requests",
            "httpx",
            "urllib",
        }
        used_names = set(getattr(code, "co_names", ()))
        hits = sorted(used_names.intersection(blocked_names))
        if hits:
            raise ValueError(
                "@mx8.transform deterministic sandbox violation: disallowed symbols: "
                + ", ".join(hits)
            )

        compile_hook = _build_compile_hook(user_fn)

        @functools.wraps(user_fn)
        def _wrapped(sample):
            return user_fn(sample)

        _wrapped.__mx8_transform__ = True
        _wrapped.__mx8_compile__ = compile_hook
        _wrapped.__mx8_transform_name__ = getattr(user_fn, "__name__", "anonymous")
        return _wrapped

    if fn is None:
        return _decorate
    return _decorate(fn)


__doc__ = mx8.__doc__
if hasattr(mx8, "__all__"):
    __all__ = list(mx8.__all__)
else:
    __all__ = []
if "transform" not in __all__:
    __all__.append("transform")
