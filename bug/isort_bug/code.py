from difflib import unified_diff
from textwrap import dedent, indent

import isort

text_in_map = {
    "single_line": dedent(r'''
        """Single-line docstring"""
        # Comment for A.
        import a
        # Comment for B - not A!
        import b
        ''').lstrip(),
    "multi_line": dedent(r'''
        """
        Multi-line docstring
        """
        # Comment for A.
        import a
        # Comment for B - not A!
        import b
        ''').lstrip(),
}

for force_sort_within_sections in [False, True]:
    for text_key, text_in in text_in_map.items():
        config = isort.Config(force_sort_within_sections=force_sort_within_sections)
        text_out = isort.code(text_in, config=config)
        print(f"force_sort_within_sections = {force_sort_within_sections}")
        print(f"text_in_map[{text_key}]")
        diff = unified_diff(
            text_in.splitlines(keepends=True),
            text_out.splitlines(keepends=True),
        )
        diff_text = "".join(diff)
        if not diff_text:
            diff_text = "<no diff>"
        print(indent(diff_text, "    "))
        print()

r'''
Output:

force_sort_within_sections = False
text_in_map[single_line]
    <no diff>

force_sort_within_sections = False
text_in_map[multi_line]
    <no diff>

force_sort_within_sections = True
text_in_map[single_line]
    <no diff>

force_sort_within_sections = True
text_in_map[multi_line]
    ---
    +++
    @@ -1,7 +1,7 @@
     """
     Multi-line docstring
     """
    -# Comment for A.
    +# Comment for B - not A!
     import a
     # Comment for B - not A!
     import b

'''
