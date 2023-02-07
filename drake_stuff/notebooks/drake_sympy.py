"""
Adapted from Matthew Halm's stuff:
https://github.com/DAIRLab/drake-pytorch/blob/3c7e33d58f1ad26008bd89f3e0fe1951b5175d3b/drake_pytorch/symbolic.py#L212
"""

import functools

import numpy as np
import sympy
from sympy import sympify

import pydrake.symbolic as sym


def _fastpow(a, b):
    if float(b) == 2.0:
        return a * a
    else:
        return a ** b


_DRAKE_TO_SYMPY_FUNCS = {
    sym.ExpressionKind.Add: sympy.Add,
    sym.ExpressionKind.Mul: sympy.Mul,
    sym.ExpressionKind.Div: lambda x, y: x / y,
    sym.ExpressionKind.Log: sympy.log,
    sym.ExpressionKind.Abs: sympy.Abs,
    sym.ExpressionKind.Exp: sympy.exp,
    sym.ExpressionKind.Pow: _fastpow,
    sym.ExpressionKind.Sin: sympy.sin,
    sym.ExpressionKind.Cos: sympy.cos,
    sym.ExpressionKind.Tan: sympy.tan,
    sym.ExpressionKind.Asin: sympy.asin,
    sym.ExpressionKind.Acos: sympy.acos,
    sym.ExpressionKind.Atan: sympy.atan,
    sym.ExpressionKind.Atan2: sympy.atan2,
    sym.ExpressionKind.Sinh: sympy.sinh,
    sym.ExpressionKind.Cosh: sympy.cosh,
    sym.ExpressionKind.Tanh: sympy.tanh,
    sym.ExpressionKind.Min: sympy.Min,
    sym.ExpressionKind.Max: sympy.Max,
    sym.ExpressionKind.Ceil: sympy.ceiling,
    sym.ExpressionKind.Floor: sympy.floor
}

_SYMPIFY_STR_SIMPLIFY = {1.0: "1", -1.0: "-1"}


def _sympy_constant_cast(x):
    x = float(x)
    str_value = _SYMPIFY_STR_SIMPLIFY.get(x)
    if str_value is None:
        str_value = str(x)
    return sympify(str_value)


def make_drake_to_sympy(drake_vars):
    return {
        hash(drake_var): sympy.Symbol(drake_var.get_name())
        for drake_var in drake_vars
    }


def drake_to_sympy(expr, to_sympy):
    """
    Convert a drake symbolic expression to sympy

    @param expr The drake expression to convert
    @param vardict Dictionary which corresponds drake variables to sympy Symbols
    @return The sympy expression
    """
    # If it's a float, just return the expression
    if isinstance(expr, float):
        return _sympy_constant_cast(expr)
    # switch based on the expression kind
    kind = expr.get_kind()
    _, expr_args = expr.Unapply()
    if kind == sym.ExpressionKind.Constant:
        arg, = expr_args
        return _sympy_constant_cast(arg)
    elif kind == sym.ExpressionKind.Var:
        var, = expr_args
        return to_sympy[hash(var)]
    else:
        sympy_func = _DRAKE_TO_SYMPY_FUNCS[kind]
        # expression combines arguments / is not leaf node
        # first, sympify constituents
        recurse = lambda arg: drake_to_sympy(arg, to_sympy)
        sympy_args = [recurse(arg) for arg in expr_args]
        return sympy_func(*sympy_args)


def drake_to_sympy_matrix(A, to_sympy):
    func = np.vectorize(
        functools.partial(drake_to_sympy, to_sympy=to_sympy)
    )
    return sympy.Matrix(func(A))


def pretty_trig(expr, syms, *, simplify=True):
    if simplify:
        expr = sympy.trigsimp(expr)
    subs = dict()
    for sym in syms:
        name = str(sym)
        c_r = sympy.cos(sym)
        c_r_short = sympy.Symbol(f"c_{name}")
        s_r = sympy.sin(sym)
        s_r_short = sympy.Symbol(f"s_{name}")
        subs[c_r] = c_r_short
        subs[s_r] = s_r_short
    expr = expr.subs(subs)
    return expr


def drake_sym_replace(expr, old, new):

    def recurse(sub):
        return drake_sym_replace(sub, old, new)

    if isinstance(expr, np.ndarray):
        recurse = np.vectorize(recurse)
        return recurse(expr)

    if not isinstance(expr, sym.Expression):
        return expr
    if expr.EqualTo(old):
        return new
    ctor, old_args = expr.Unapply()
    new_args = [recurse(x) for x in old_args]
    return ctor(*new_args)
