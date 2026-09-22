# Manuscript build

```
make check     # sources, class file and toolchain, without compiling
make tables    # regenerate every table from the committed results
make           # full build
```

## What is generated and what is not

Every table is emitted by `tools/gen_tables.py` from a committed, provenance-stamped results file under
`results/`. No number in the manuscript is typed by hand. The generator resolves all declared sources before
writing anything, so a missing source aborts the build rather than producing a manuscript with a stale table:

```
$ make tables
gen_tables: MISSING SOURCE results/gates/extraction_convention.json
  expected at .../results/gates/extraction_convention.json
  the manuscript cannot be built without it; ...
make: *** [tables] Error 1
```

`tables/_provenance.tex` records the results commit the tables were generated from and the sha256 of every source
file; `\resultscommit` is available in the manuscript.

## Two files you must supply

`ieeeaccess.cls` and `IEEEtran.bst` are IEEE's template files and are not redistributable, so they are not
committed. Download the IEEE Access LaTeX template and drop both into `paper/`. `make check` fails with a clear
message until they are present.

## Editorial markers

`\open{...}` renders in red and marks anything still to be written or confirmed. The manuscript is not submittable
while any remain; grep for `\open{` before submission.
