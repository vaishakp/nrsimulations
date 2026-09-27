"""Location of the paper repo these scripts write their LaTeX and figures into.

The .tex source and figures live in their own Overleaf-synced repo,
vaishakpsu/nrcatalog-paper, checked out beside this one:

    ~/Projects/Codes/nrsimulations/     # these scripts, plus the 5 GB of data
    ~/Projects/Codes/nrcatalog-paper/   # nrcatalog.tex, figures/

They are separate because Overleaf syncs a whole repo, and the waveforms and
harvested JSON that these scripts read are far too large to carry into it.

Set NRCATALOG_PAPER to point somewhere else.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

PAPER = os.path.abspath(os.environ.get(
    "NRCATALOG_PAPER", os.path.join(HERE, "..", "..", "..", "nrcatalog-paper")))

FIGURES = os.path.join(PAPER, "figures")

if not os.path.isdir(PAPER):
    raise SystemExit(
        f"paper repo not found at {PAPER}\n"
        "Clone it beside nrsimulations, or set NRCATALOG_PAPER:\n"
        "    git clone git@psugithub:vaishakpsu/nrcatalog-paper.git")
