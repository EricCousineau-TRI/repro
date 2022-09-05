def debug_stuff():
    for name, rule in native.existing_rules().items():
        print(name)
        print(rule["tags"])
