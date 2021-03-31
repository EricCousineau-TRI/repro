import pprint as pp
from textwrap import indent

def stuff():
    # TODO(eric.cousineau): This causes training to break on pl==1.2.0, but not
    # 1.2.6.
    #   File ".../pytorch_lightning/core/optimizer.py", line 100, in _to_lightning_optimizer
    #     optimizer = trainer.lightning_optimizers[opt_idx]
    # KeyError: 0
    pprint_trainer_args(trainer)


def pprint_trainer_args(trainer):
    ignore = {"progress_bar_dict"}
    out = {}
    for attr in dir(trainer):
        if attr.startswith("_") or attr in ignore:
            continue
        try:
            v = getattr(trainer, attr)
        except AttributeError as e:
            raise RuntimeError((e, attr))
        if is_primitive(v):
            out[attr] = v
    print(pformat(out))


def is_primitive(v):
    if isinstance(v, (bool, int, float, str)):
        return True
    elif isinstance(v, (list, tuple)):
        for vi in v:
            if not is_primitive(vi):
                return False
        return True
    elif isinstance(v, dict):
        for ki, vi in v.items():
            if not is_primitive(ki) or not is_primitive(vi):
                return False
        return True
    else:
        return False
    assert False



def pformat(obj, incr="  "):
    """
    Pretty formatting for values with more vertical whitespace, less hanging
    indents.
    """
    def sub_pformat(obj):
        txt = pformat(obj, incr=incr)
        return indent(txt, incr)
    # Try short version.
    short_len = 60
    maybe_short = pp.pformat(obj)
    if "\n" not in maybe_short and len(maybe_short) <= short_len:
        return maybe_short

    if isinstance(obj, list):
        out = f"[\n"
        for obj_i in obj:
            out += sub_pformat(obj_i) + ",\n"
        out += f"]"
        return out
    elif isinstance(obj, dict):
        out = f"{{\n"
        for k_i, obj_i in obj.items():
            txt = sub_pformat(obj_i)
            out += f"{incr}{repr(k_i)}: {txt.strip()},\n"
        out += f"}}"
        return out
    else:
        return indent(pp.pformat(obj), incr)
