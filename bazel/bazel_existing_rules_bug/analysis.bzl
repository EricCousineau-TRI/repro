def print_rule_deps():
    for rule in native.existing_rules().values():
        name = rule["name"]
        deps = rule["deps"]
        print("{}: {}".format(name, deps))
