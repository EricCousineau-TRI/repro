
def cmake_replace_cmd(reps):
    seds = []
    for (key, value) in reps.items():
        seds.append("sed s#@{}@#{}#g".format(key, value))
    supersed = " | ".join(seds)
    return "( {} ) < $< > $@".format(supersed)
