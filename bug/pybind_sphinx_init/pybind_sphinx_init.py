import os
# Optionally add workaround.
INIT_WORKAROUND = os.environ["_INIT_WORKAROUND"] == "1"


def autodoc_skip_member(app, what, name, obj, skip, options):
    """Skips undesirable members."""
    # The qualname for pybind may end up being something like
    # `PyCapsule.__init__`.
    if "__init__" in name:
        return False
    return None


def setup(app):
    """Installs Drake-specific extensions and patches.
    """
    # Skip specific members.
    if INIT_WORKAROUND:
        app.connect('autodoc-skip-member', autodoc_skip_member)
    return dict(parallel_read_safe=True)
