# Handoff — 2026-09-27

Picked up after losing track of where this work lived. Nothing was lost; two
things were just invisible. Both are open.

## 1. `paper/` is untracked — DONE 2026-09-27

The paper is now in version control, split across two repos instead of one.

**`vaishakpsu/nrcatalog-paper`** (private, new) — `~/Projects/Codes/nrcatalog-paper/`,
a sibling of this repo, matching the layout of the other Overleaf-synced paper
repos: main `.tex` and `figures/` at the root. Holds `nrcatalog.tex`,
`macros*.tex`, `table_*.tex` and all 12 figure PDFs plus their `-1.png`
previews. Ready to attach to Overleaf; remote is `git@psugithub:...`.

**This repo** — keeps `paper/scripts/` (the generators) and the data they read.
`paper/data/`, `paper/waveforms/` and `paper/horizons/` are gitignored.

Two things found along the way, both worth remembering:

- There is **no `.bib` file and none is needed** — the bibliography is an inline
  `thebibliography` with 34 hand-written `\bibitem`s at the end of
  `nrcatalog.tex`. `nrcatalog.bbl` (apsrev boilerplate, zero entries) and
  `nrcatalogNotes.bib` are build droppings.
- The paper repo's `.gitignore` names each build artifact **root-anchored**
  rather than using blanket patterns. `nrhjsurrogate-paper` learned this the
  hard way: a blanket `*.pdf` rule silently swallowed two figure PDFs, so the
  paper compiled locally but failed on Overleaf with "File not found". Every
  figure here is a `fig_*.pdf`, so that trap was one rule away.

The built PDF is **not** committed — Overleaf builds it, and a tracked PDF just
causes sync conflicts. Verified by cloning the repo clean and compiling: 15
pages, 0 errors, 0 undefined citations, 0 undefined references, 0 missing files.

Because the `.tex` moved out, the scripts no longer write to `HERE/..`. They now
resolve the paper repo through `paper/scripts/paper_dir.py`, which expects it
beside this one and honours `$NRCATALOG_PAPER`. Regenerating the tables and
macros after the move reproduced the committed files byte-for-byte. Still do not
hand-edit `macros*.tex` / `table_*.tex` (see the `nr-catalog-paper-layout`
memory for which script makes what).

Note this commit was pushed **only** to the private `psu` remote, not to
`origin` — `vaishakp/nrsimulations` is public and is the target of the paper's
own `\cite{nrsimulations}`, so putting the unpublished analysis code there is a
call to make deliberately.

## 2. The sonic → gwave copy is stopped, not finished

Last write **2026-09-23 19:03**. No rsync/scp process and no screen or tmux
session survives on the sonic login node, on sonic13, or on gwave. No transfer
script or log was saved on either side, so whatever drove it has to be
reconstructed.

Source: `/mnt/pfs/vaishak.p/sims/SpEC/gcc/bfi/` on sonic — six run families,
`EccContPrecDiff`, `EccPrecDiff`, `ICTSEccParallel`, `ICTSEccPrecX`,
`PSUTest`, `PSUTestHR`.

Landed in `~/sonic_archive` on gwave, 1.3 TB total:

- `EccContPrecDiff001/` — 1.2 TB (`Ev/`, `metadata/`, `waveforms/`)
- `ICTSEccParallel05/` — 43 GB
- `CCE/` — empty, never started
- `_probe/` — 6.4 GB, looks like a throughput test rather than real data

So one run from each of two families crossed; the bulk of the catalogue did not.

**Before restarting:** `~/sonic_archive` is on gwave's *home*, not `/scratch2`
(181 TB free). The standing rule for that cluster is to stage to scratch, never
home — worth redirecting before pushing another terabyte. Decide too whether
full `Ev/` segment trees are actually wanted on gwave or just
`waveforms/` + `metadata/`, since `Ev/` is what makes this 1.2 TB per run.

## Access

    loginson && ssh sonic13     # never build, submit, or trust output from sonic-master
    logingwave                  # vaishak.prasad@ligo01.gwave.ics.psu.edu

Read `SONIC_BUILD_AND_RESUME.md` before touching the cluster.
