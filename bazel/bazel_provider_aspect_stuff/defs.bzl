MyAspectInfo = provider(fields = ["label_to_target"])

def _my_aspect_impl(target, ctx):
    deps = ctx.rule.attr.deps

    label_to_target = {}
    for dep in deps:
        info = dep[MyAspectInfo]
        label_to_target.update(info.label_to_target)
        # this *does* have the info we want.
        label_to_target[dep.label] = dep

    # this does not yet have MyAspectInfo
    # label_to_target[target.label] = target
    return [
        MyAspectInfo(label_to_target = label_to_target)
    ]

my_aspect = aspect(
    implementation = _my_aspect_impl,
    attr_aspects = ["deps"],
)

OtherAspectInfo = provider(fields = ["meh"])

def _other_aspect_impl(target, ctx):
    return [
        OtherAspectInfo(meh = None)
    ]

other_aspect = aspect(
    implementation = _other_aspect_impl,
)

def demand(good, msg = ""):
    if not good:
        fail(msg)

def _my_rule_impl(ctx):
    lib_a_label = Label("//:lib_a")
    lib_b_label = Label("//:lib_b")
    # see if we can reach into `:lib_a` aspects via `:lib_b`.
    (dep,) = ctx.attr.deps
    # demand(lib_b.label == lib_b_label)
    label_to_target = dep[MyAspectInfo].label_to_target
    lib_a_via_map = label_to_target[lib_a_label]
    lib_b_via_map = label_to_target[lib_b_label]
    # print(lib_b)
    print(lib_b_via_map)
    print(lib_a_via_map)

    # check_providers(lib_b)
    # Fails here, specifically on MyAspectInfo.
    check_providers(lib_b_via_map)
    check_providers(lib_a_via_map)

def check_providers(target):
    target[CcInfo]
    target[MyAspectInfo]
    # This doesn't work :/
    # Would need to couple using `aspect(required_aspect_providers)`, which
    # makes things less ideal.
    target[OtherAspectInfo]

_my_rule = rule(
    implementation = _my_rule_impl,
    attrs = {
        "deps": attr.label_list(
            providers = [CcInfo],
            aspects = [my_aspect, other_aspect],
        ),
    },
)

def my_rule(name, deps):
    shim_name = "_" + name + "_shim"
    native.cc_library(
        name = shim_name,
        deps = deps,
    )
    _my_rule(
        name = name,
        deps = [":" + shim_name],
    )
