"""Entrypoint scripts, as a regular package rather than a namespace one.

Without this file `scripts/` is an implicit namespace package: `import scripts.run_sweep`
merges candidate directories from every `sys.path` entry that contains a `scripts/`, and
path order decides which one wins. A second checkout on `PYTHONPATH` could then run its
tests against another checkout's scripts. `lmopt._import_guard` covers the same exposure
for `src/lmopt`.
"""
