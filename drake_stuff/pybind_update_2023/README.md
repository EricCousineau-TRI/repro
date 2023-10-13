# Updating `RobotLocomotion/pybind11` fork

Note: Merge using VSCode, way nicer than Meld for syntax highlighting + indentation
<https://stackoverflow.com/questions/69619604/visual-studio-code-review-merge-changes-side-by-side-rather-than-top-down>

```sh

cd pybind11
git checkout robotlocomotion/drake
# We squash merged, oops. `git merge-base` is garbage. Well:
git checkout -b feature-bump-merge
git reset --hard 70a58c5
git checkout --no-overaly robotlocomotion/drake
# commit

# Now start merging.
git merge 93716147  # merge of v2.9.0
# Resolve using vscode. Commit.
git merge v2.9.1
# - ref: e7881078fbd

# Resolve.
git merge v2.10.0
# - Er, painful... Try another way:


git checkout -b feature-bump-merge-fmt e7881078fbd
# We are looking to first merge in pybind11#3713, ec24786eab
git checkout ec24786eab -- :/
# Only select pre-commit changes, discard everything else.
# - ref: 5951b248

pre-commit run clang-format -a
# er... stil
git merge ec24786eab
# ... nope! painful still.


git reset --hard 5951b248
git merge ec24786eab~
# - ref: f0a4dd50
pre-commit run clang-format -a
# commit
git merge ec24786eab
# Resolve, commit.
pre-commit run clang-format -a
# commit.
# - ref: 6bc74137


git merge v2.10.0
# resolve, much easier to understand... I think?



# Tests - pass!


# Now reconnect history (oops)

# Can use `git cherry-pick -m 1 ...` to replay merges, e.g.
# git cherry-pick -m 1 f0a4dd50
# errr, nevermind! didn't work :(

# had to use 9352ff9f602 to restore merge history :/

$ git log --first-parent HEAD --oneline
cc8f36ea (HEAD -> feature-bump-merge, origin/feature-bump-merge) partial revert to proper form of 8adef2c
224d0b31 huh, missed carrying this one over - oops!
773f343f (origin/feature-bump-merge-fmt, feature-bump-merge-fmt) Merge tag 'v2.10.0' into feature-bump-merge-fmt
6bc74137 re-run pre-commit
a022f1bc Merge commit 'ec24786eab' into feature-bump-merge-fmt
dbed3e4b run re-format
f0a4dd50 Merge commit 'ec24786eab~' into feature-bump-merge-fmt Before auto-format
5951b248 import clang tidy setup
e7881078 Merge tag 'v2.9.1' into feature-bump-merge
94bc246d Merge commit '9371614764941a8f51fb75979b6bc09f94e55a43' (v2.9.0) into feature-bump-wip
986d4928 (origin/feature-bump-wip) squashed changes from Drake
70a58c57 Replace usage of deprecated Eigen class MappedSparseMatrix. (#3499)


$ python ./replay.py
```


## Merging in later versions

```
v2.9.0
v2.9.1
v2.9.2
v2.10.0
v2.10.1
v2.10.2
v2.10.3
v2.10.4
v2.11.0
v2.11.1
```

```sh
$ git merge v2.11.1
# Nope - eigen.h got split up -- need to handle that more directly...


# fab1eebe2c4c - renamed eigen.h to eigen/matrix.h, added eigen.h redirect and eigen/tensor.h
  introduced in v2.10.2
# 8e1f9d5c40f7 - created eigen/common.h - but it's small, no need to look at this


git merge fab1eebe2c4c~

# move to let conflict res simplify
( cd include/pybind11
mkdir eigen
mv eigen.h ./eigen/matrix.h
)
# commit

# erg blech, didn't seem great
git merge fab1eebe2c4c
# - if this needs to be redone, should actually make a *separate* branch, simulate *proper* git rename,
#   then do the merge, that way git can help us more during the merge.

git merge v2.11.1
```


## Some things to fix

```sh
$ cmake --build build --target check -j
...
In function ‘cast’,
    inlined from ‘operator()’ at /home/eacousineau/proj/tri/repo/externals/pybind11/include/pybind11/pybind11.h:253:33,
    inlined from ‘_FUN’ at /home/eacousineau/proj/tri/repo/externals/pybind11/include/pybind11/pybind11.h:228:21:
/home/eacousineau/proj/tri/repo/externals/pybind11/include/pybind11/detail/../cast.h:653:13: warning: ‘operator delete’ called on unallocated object ‘int_string_pair’ [-Wfree-nonheap-object]
  653 |             delete src;
      |             ^
/home/eacousineau/proj/tri/repo/externals/pybind11/include/pybind11/detail/../cast.h: In function ‘_FUN’:
/home/eacousineau/proj/tri/repo/externals/pybind11/tests/test_builtin_casters.cpp:269:40: note: declared here
  269 |     static std::pair<int, std::string> int_string_pair{2, "items"};
      |                                        ^
In function ‘cast’,
    inlined from ‘operator()’ at /home/eacousineau/proj/tri/repo/externals/pybind11/include/pybind11/pybind11.h:253:33,
    inlined from ‘_FUN’ at /home/eacousineau/proj/tri/repo/externals/pybind11/include/pybind11/pybind11.h:228:21:
/home/eacousineau/proj/tri/repo/externals/pybind11/include/pybind11/stl.h:190:5: warning: ‘operator delete’ called on unallocated object ‘lvv’ [-Wfree-nonheap-object]
  190 |     PYBIND11_TYPE_CASTER(Type, const_name("List[") + value_conv::name + const_name("]"));
      |     ^
/home/eacousineau/proj/tri/repo/externals/pybind11/include/pybind11/stl.h: In function ‘_FUN’:
/home/eacousineau/proj/tri/repo/externals/pybind11/tests/test_stl.cpp:179: note: declared here
  179 |     static std::vector<RValueCaster> lvv{2};
      |
```
