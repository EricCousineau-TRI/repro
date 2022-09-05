MyAspectInfo = provider(fields = ["dep_label_to_info"])

def _my_aspect_impl(target, ctx):
    deps = ctx.rule.attr.deps

    dep_label_to_info = {}
    for dep in deps:
        info = dep[MyAspectInfo]
        dep_label_to_info[dep.label] = info
        dep_label_to_info.update(info.dep_label_to_info)
    return [
        MyAspectInfo(dep_label_to_info = dep_label_to_info)
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
    (lib_b,) = ctx.attr.deps
    demand(lib_b.label == lib_b_label)
    info = lib_b[MyAspectInfo]
    label_to_info = dict(info.dep_label_to_info)
    label_to_info[lib_b.label] = info

    lib_a_via_map = label_to_info[lib_a_label]
    lib_b_via_map = label_to_info[lib_b_label]
    # print(lib_b)
    print(lib_b_via_map)
    print(lib_a_via_map)

    # check_providers(lib_b)
    # # Fails here, specifically on MyAspectInfo.
    # check_providers(lib_b_via_map)
    # check_providers(lib_a_via_map)

def check_providers(target):
    target[CcInfo]
    target[MyAspectInfo]
    # This doesn't work :/
    # Would need to couple using `aspect(required_aspect_providers)`, which
    # makes things less ideal.
    # target[OtherAspectInfo]

my_rule = rule(
    implementation = _my_rule_impl,
    attrs = {
        "deps": attr.label_list(
            providers = [CcInfo],
            aspects = [my_aspect, other_aspect],
        ),
    },
)
