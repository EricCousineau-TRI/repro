"""
Tools to extract deps recursively and maintain a (custom) graph.
"""

ExtractDepsInfo = provider(
    fields = [
        # All dependencies, including label itself.
        "dep_labels",
        # Map from this label to all of its dependencies (including itself).
        # TODO(eric.cousineau): Better way to transitively aggregate dicts?
        "dep_label_map",
    ],
)

# TODO(eric.cousineau): If degree of connectivity is needed, just need to add
# a field like `direct_deps`, then some basic tools.

# TODO(eric.cousineau): Would be nice to supply this as external option. Eh.
_REMAP_LABELS = {
    # Bazel seems to resolve this alias. Un-resolve it.
    Label("@drake//tools/install/libdrake:drake_shared_library"): (
        Label("@drake//:drake_shared_library")
    ),
}

def remap_label_for_anzu(label):
    return _REMAP_LABELS.get(label, label)

def _extract_deps_aspect_impl(target, ctx):
    deps = getattr(ctx.rule.attr, "deps", [])

    dep_labels = []
    dep_label_map = {}
    for dep in deps:
        info = dep[ExtractDepsInfo]
        dep_labels.append(info.dep_labels)
        dep_label_map.update(info.dep_label_map)

    target_label = remap_label_for_anzu(target.label)
    dep_labels = depset([target_label], transitive = dep_labels)
    dep_label_map[target_label] = dep_labels
    return [
        ExtractDepsInfo(
            dep_labels = dep_labels,
            dep_label_map = dep_label_map,
        ),
    ]

"""
Provides both transitive dep_labels and dep_label_map, from label to its own
deps.
"""

extract_deps_aspect = aspect(
    implementation = _extract_deps_aspect_impl,
    attr_aspects = ["deps"],
)

def aggregate_extract_deps(infos):
    """
    Aggregate for consumers that may query against multiple targets.
    """
    dep_labels = []
    dep_label_map = {}
    for info in infos:
        dep_labels.append(info.dep_labels)
        dep_label_map.update(info.dep_label_map)
    dep_labels = depset(transitive = dep_labels)
    return ExtractDepsInfo(
        dep_labels = dep_labels,
        dep_label_map = dep_label_map,
    )

def expand_label_deps(labels, dep_label_map):
    """
    Given a set of labels, return their deps from dep_lable_map.
    Returns expanded deps, as well as any missing targets.

    Note: We return missing because some labels may not have sufficient
    information based on what target a rule() or aspect() has access to.
    """
    dep_labels_transitive = []
    missing = []
    for label in labels.to_list():
        dep_labels = dep_label_map.get(label, None)
        if dep_labels == None:
            missing.append(label)
        else:
            dep_labels_transitive.append(dep_labels)
    dep_labels = depset(transitive = dep_labels_transitive)
    missing = depset(missing)
    return dep_labels, missing
