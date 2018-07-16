expected = open('input.txt').read().strip()
actual = open('output.txt').read().strip()

assert expected == actual, (
    "{} != {}".format(expected, actual))
