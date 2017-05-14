#!/usr/bin/env python

import os
import subprocess
import re

def subshell(cmd):
    return subprocess.check_output(cmd, shell=True)

def get_pwd_workspace_name():
    workspace = subshell("bazel info workspace").strip()
    workspace_file = os.path.join(workspace, 'WORKSPACE')

    text = open(workspace_file).read()

    # Find workspace() statement
    m = re.search(r"workspace\s*\((.*?)\)", text, re.M)
    # Evaluate method to recover kwargs
    workspace_info = eval(m.group(0), {"workspace": dict})
    # Extract name
    workspace_name = workspace_info['name']
    return workspace_name

if __name__ == "__main__":
    print(get_pwd_workspace_name())
