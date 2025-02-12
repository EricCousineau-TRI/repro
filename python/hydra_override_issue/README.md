# To run

```sh
uv run ./example.py --config-name=example
```

Desired output: `{'value': 'comes from /component/stuff'}`

Actual output: `{'value': 'comes from /example_template'}`

Can inspect difference in ordering with `--info=defaults` per documentation.
This is also consistent with overriding the value via command-line, e.g. `component=other`.

The following patch produces desired output:
```diff
diff --git a/python/hydra_override_issue/base/default.yaml b/python/hydra_override_issue/base/default.yaml
index cf498d62..e5735937 100644
--- a/python/hydra_override_issue/base/default.yaml
+++ b/python/hydra_override_issue/base/default.yaml
@@ -2,5 +2,5 @@

-defaults:
-  # Placeholder to make it easier to consistently use `override`.
-  - /component: null
+# defaults:
+#   # Placeholder to make it easier to consistently use `override`.
+#   - /component: null

diff --git a/python/hydra_override_issue/example.yaml b/python/hydra_override_issue/example.yaml
index 024cec85..a1677d29 100644
--- a/python/hydra_override_issue/example.yaml
+++ b/python/hydra_override_issue/example.yaml
@@ -3,3 +3,3 @@ defaults:
   - example_template
-  - override component: stuff
+  - component: stuff
   - _self_
```

## Workaround

Can inject the placeholder `component: null` before any overlays that require the component.
