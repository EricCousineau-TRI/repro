import copy_ctor as m

def check():
    c = m.Custom(1)
    print("---")
    c2 = m.Custom(c)

def main():
    check()
