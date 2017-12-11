#!/usr/bin/env python

def cexec(stmt, locals_ = None, globals_ = None):
    if locals_ is None:
        locals_ = {}
    if globals_ is None:
        globals_ = {}
    eval_locals = dict(stmt=stmt, locals_=locals_, impl=_impl)
    eval("impl(stmt, locals_)", eval_locals, globals_)

def _impl(stmt, locals_):
    # Record old things.
    locals().update(locals_)
    _old_vars = None
    _new_vars = None
    _old_vars = locals().keys()
    # Execute with context.
    exec stmt
    # Figure out new things.
    _new_vars = set(locals().keys()) - set(_old_vars)
    for var in _new_vars:
        locals_[var] = locals()[var]

cexec("print(1); x = 2; print(x)", locals())
print(2 * x)
