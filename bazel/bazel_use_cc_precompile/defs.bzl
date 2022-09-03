def _impl(ctx):
    for dep in ctx.attr.deps:
        print(dep)

cc_scraper = rule(
    implementation = _impl,
    attrs = [
        "deps": attr.label_list(),
    ],
)
