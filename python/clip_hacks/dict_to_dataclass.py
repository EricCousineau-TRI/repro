#!/usr/bin/env python3

from textwrap import indent

import pyperclip


def main():
    text = pyperclip.paste()
    print("[ Paste -> Input ]")
    print(indent(text, "  "))
    print("[ Output -> Copied]")
    text = text.replace('"]["', '.').replace('["', '.').replace('"]', '')
    print(indent(text, "  "))
    pyperclip.copy(text)


assert __name__ == "__main__"
main()
