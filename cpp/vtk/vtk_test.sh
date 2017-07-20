#!/bin/bash
set -e -u

glxinfo | grep -i version -C 1
glxgears
