def do_stuff():
    items = ["b", "a", "b", "b", "a"]
    # will dedup yield ["b", "a"] or ["a", "b"] ?
    print(depset(items).to_list())
    print(depset(items[::-1]).to_list())
