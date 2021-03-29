# Single-Input Single-Output (SISO) Curve Fitting with MLP

Motivation: I tried this briefly a few days ago (2021-03-25) and couldn't make
it work, so I felt dumb.

Notebook:

* [`linear_mult.ipynb`](./linear_mult.ipynb) - Shows curve fitting for a very
  simple linear scalar multiplication. Just using SGD here.
* [`pwa_ish_mlp.ipynb`](./pwa_ish_mlp.ipynb) - Made a simple sawtoothed signal
  with a ground-truth MLP network representation. Does some training w/ `SGD`,
  `Adam`, `RMSprop`, plays with perturb network param to briefly check
  "stability" of training.
* [`mlp_sin.ipynb`](./mlp_sin.ipynb) - Fits an MLP to a simple sinusoid.
