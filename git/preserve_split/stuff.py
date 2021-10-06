#!/usr/bin/env python3

"""
Learn https://devblogs.microsoft.com/oldnewthing/20190919-00/?p=102904 by example.
"""

import shell_defs as m


def write_to(filename, text, *, mode="w"):
    with open(filename, mode) as f:
        f.write(text)


def main():
    repo = "example_repo"
    author_1 = "Alice <alice>"
    author_2 = "Bob <bob>"
    author_3 = "Greg <greg>"
    file = "foods.txt"
    file_new = "foods_new.txt"

    m.shell(f"rm -rf {repo}")
    m.shell(f"mkdir {repo}")
    m.cd(repo)
    m.shell("git init .")

    write_to(file, "apple\n")
    m.shell(f"git add {file}")
    m.shell(f"git commit --author='{author_1}' -m 'Created'")

    write_to(file, "orange\n", mode="a")
    m.shell(f"git commit --author='{author_2}' -am 'Added line'")

    m.shell(f"git blame {file}")

    # Copy file.
    m.shell(f"git checkout -b dup")
    m.shell(f"git mv {file} {file_new}")
    m.shell(f"git add {file_new}")
    m.shell(f"git commit --author='{author_3}' -am 'Moved'")

    m.shell(f"git checkout HEAD~ -- {file}")
    m.shell(f"git commit --author='{author_3}' -am 'Restored'")

    m.shell(f"git checkout -")
    m.shell(f"git merge -m 'Merge' --no-ff dup")

    m.shell(f"git log --format=short {file}")
    m.shell(f"git blame {file_new}")

    m.shell(f"rm -rf {repo}")


assert __name__ == "__main__"
main()
