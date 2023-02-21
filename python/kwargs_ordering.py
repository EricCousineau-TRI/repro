def my_func(a, b=2, *, c, d=4):
    print(a, b, c, d)


my_func(1, c=3)
my_func(10, 20, c=30, d=40)

"""
Output:

1 2 3 4
10 20 30 40
"""
