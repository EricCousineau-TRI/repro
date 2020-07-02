class AttrDict(dict):
    """Access / mutate dictionary entries as attributes."""
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    @staticmethod
    def create_recursive(other):
        assert isinstance(other, dict), type(other)
        out = AttrDict()
        for k, v in other.items():
            if isinstance(v, dict):
                v = AttrDict.create_recursive(v)
            out[k] = v
        return out

    def asdict(self):
        out = dict()
        for k, v in self.items():
            if isinstance(v, dict) and type(v) != dict:
                v = dict(v)
            out[k] = v
        return out


def recursive_dict_update(a, b):
    """
    Rescursively merge entries from b into a, where b will values in a for any
    keys. Does not copy values (e.g. aliasing can still be an issue).

    Will raise an error if field types do not match for shared keys (e.g. a[k]
    is a dict, but b[k] is not).
    """
    assert isinstance(a, dict) and isinstance(b, dict), (a, b)
    for k, b_v in b.items():
        if k in a:
            a_v = a[k]
            # Do not allow heterogeneous type updates.
            assert type(a_v) == type(b_v), (k, a_v, b_v)
            if isinstance(b_v, dict):
                a[k] = recursive_dict_update(a_v, b_v)
            else:
                a[k] = b_v
        else:
            a[k] = b_v
    return a


def normalize_wandb_sweep_config(config, update=recursive_dict_update):
    """
    Change config dict from (yaml form):

        top
          mid:
            bottom: 1  # Default value.
        top.mid.bottom: 0.25  # From wandb sweep controller.
        # top.mid: 0.35  # Should cause error.
        # top.mid: {bottom: 0.45}  # Should cause error.

    to (yaml form):

        top:
          mid:
            bottom: 0.25

    Nested fields are of the form {"a.b.c": value}, which should then map to
    the structure {"a": {"b": {"c": value}}}.

    Example usage:

        config = AttrDict(
            my_param=1,
            nested=AttrDict(
                sub_param=2,
            ),
        )
        wandb.init(config=config)
        config = normalize_wandb_sweep_config(dict(wandb.config.user_items()))
        config = AttrDict.create_recursive(config)

    Workaround for: https://github.com/wandb/client/issues/982
    """
    # Strictly parse dictionaries only.
    assert type(config) == dict
    out = dict()
    nested = dict()
    for k, v in config.items():
        assert isinstance(k, str), k
        k_list = k.split(".")
        if len(k_list) > 1:
            assert not isinstance(v, dict), (
                f"Cannot have nested field be a dict: {(k, v)}")
            nested_n = nested
            for k_i in k_list[:-1]:
                if k_i not in nested_n:
                    nested_n[k_i] = dict()
                nested_n = nested_n[k_i]
                assert isinstance(nested_n, dict), (
                    "Intermediate field must be dict")
            k_n = k_list[-1]
            assert k_n not in nested_n, (
                "Cannot mix middle level nesting")
            nested_n[k_n] = v
        else:
            out[k] = v
    return update(out, nested)
