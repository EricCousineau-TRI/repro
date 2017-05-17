#!/bin/bash

git diff --no-index test_{local,ci}.output.txt > test.output.patch
