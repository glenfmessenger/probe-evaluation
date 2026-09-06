# Manuscript build. Tables are regenerated from committed results before every compile, and the build FAILS if a
# declared source JSON is missing (PAPER_BRIEF framing rule 3: no untraced numbers).
#
#   make            build the PDF
#   make tables     regenerate tables/ only
#   make figures    regenerate figures/ only
#   make check      verify sources, class file and toolchain without compiling  (works without LaTeX installed)
#   make clean      remove build artifacts (generated tables are kept; `make distclean` removes them too)

REPO    := $(shell git -C .. rev-parse --show-toplevel)
PY      ?= python3
LATEXMK ?= latexmk
MAIN    := main
CLASS   := ieeeaccess.cls

.PHONY: all tables figures check clean distclean

all: check tables figures
	@command -v $(LATEXMK) >/dev/null 2>&1 || { \
	  echo "BUILD FAILED: $(LATEXMK) not found. Install a TeX distribution (TeX Live / MacTeX)."; exit 1; }
	$(LATEXMK) -pdf -halt-on-error -interaction=nonstopmode $(MAIN).tex

# Tables are a hard dependency of the manuscript: the generator exits non-zero on any missing source, which
# propagates here and stops the build.
tables:
	$(PY) tools/gen_tables.py

# Figures are generated from the same committed results as the tables and fail the same way if a source is missing,
# so framing rule 3 (no untraced numbers) covers graphics as well as text.
figures:
	$(PY) tools/gen_figures.py

# Content checks run FIRST and unconditionally, so that a missing IEEE template file -- the one thing the author
# supplies by hand -- does not mask a real problem in the manuscript or the results wiring.

check:
	$(PY) tools/gen_tables.py --check
	$(PY) tools/gen_figures.py --check
	$(PY) tools/check_refs.py
	@test -f $(CLASS) || { \
	  echo "BUILD FAILED: $(CLASS) not present in paper/."; \
	  echo "  IEEE Access does not permit redistribution of the class file, so it is not committed here."; \
	  echo "  Download the IEEE Access LaTeX template from the author portal and place $(CLASS) and IEEEtran.bst"; \
	  echo "  in paper/. Content checks above have already run and passed."; \
	  exit 1; }
	@test -f IEEEtran.bst || { echo "BUILD FAILED: IEEEtran.bst not present in paper/ (see above)."; exit 1; }
	@echo "check: ok"

clean:
	$(LATEXMK) -C >/dev/null 2>&1 || true
	rm -f $(MAIN).bbl $(MAIN).blg $(MAIN).aux $(MAIN).log $(MAIN).out

distclean: clean
	rm -f tables/tab_*.tex tables/_provenance.tex figures/fig_*.pdf
