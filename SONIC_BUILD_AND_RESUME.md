# Rebuilding SpEC and resuming runs on sonic

Written 2026-09-22. Everything here was verified on the machine, not inferred.

## The production revision

Every catalogue simulation was run with **one** SpEC revision:

```
17e9207c30a2f189de01429c574d55312dac1b23
InitialCommit-32087-g17e9207     authored 2023-02-22, linked 2023-03-03
```

Verified by scanning `Code Revision` in `*/Run/SpEC.out` across the whole tree:
126 catalogue segments carry this revision. The only other build in the tree
(`c7540f2c7153180cad9c609a2e7ae3dc6d48b0bc`, `InitialCommit-32910`, linked
2025-02-27) appears **only** in `PSUTest` / `PSUTestHR`, which are not
catalogue runs.

## Getting there

```bash
ssh vaishak.p@sonic.icts.res.in      # = the `loginson` alias in ~/.bashrc
ssh sonic13                          # then hop to a compute node
```

**Do everything from a compute node, never from the login node.** `sonic-master`
is still CentOS 7.9 with **slurm 21.08.0**, while the compute nodes are Rocky
Linux 9.2 with **slurm 23.02.4**. That major version mismatch makes the login
node's client lie:

- `sinfo` from the login node reports every node `DOWN+NOT_RESPONDING`
- `srun` from the login node fails with `Invalid account or account/partition
  combination specified`
- the `gpu` partition (sonic15) is invisible from the login node entirely

From sonic13 all of it works normally. This cost an hour of misdiagnosis once;
don't repeat it.

There is a third consequence that bites even when you do everything right:
**SpEC itself submits from the login node.** At the end of an initial-data
solve, `BBH_ID_Postprocess.postprocess` runs

```
ssh sonic-master "cd .../Ev/ && bash -l -c './StartJob.sh'"
```

which fails both because sonic-master's `sbatch` is broken and because a
Rocky-9-built `Nsubdomains` cannot execute on its CentOS 7 glibc. The ID solve
itself is unaffected — only the automatic handoff to the evolution dies. Just
run `Ev/StartJob.sh` yourself from sonic13 afterwards.

One more sizing trap: `MakeSubmit` silently scales a modest core request up to
a full 48-core node, reserving 192 GB, which will not fit beside other running
jobs and leaves the job pending on `(Resources)` even when plenty of CPUs are
idle. Force the size you want:

```bash
./bin/MakeSubmit.py update --Cores 16 --ForceCores True \
    --CoresPerNode 16 --ForceCoresPerNode True -f
```

## Why a rebuild is needed at all

The original binaries still exist inside each segment's `Run/SpEC`, but they no
longer run. `ldd` looks clean, yet execution dies with

```
undefined symbol: ..._ZN7Factory15EnableCreation0I13ObserverAdderE...DumpTidalTailData...
```

Cause: the binaries' rpath points into the **live working tree**
`/mnt/pfs/vaishak.p/spec/...`, and all 136 SpEC `.so` files there have since
been rebuilt from a different commit (`2c30643 add few modules`). The old
executable now loads mismatched libraries. So: never rebuild in the live tree,
and expect archived binaries to be unusable.

## Build recipe

`git worktree` is unavailable (sonic's git is 1.8.3.1), so clone instead:

```bash
cd /mnt/pfs/vaishak.p/spec
git diff MakefileRules/Machines/Sonic-gcc.env > /tmp/sonic_env.patch   # keep local mods
git clone --no-hardlinks /mnt/pfs/vaishak.p/spec /mnt/pfs/vaishak.p/spec-17e9207
cd /mnt/pfs/vaishak.p/spec-17e9207
git checkout 17e9207c30a2f189de01429c574d55312dac1b23
git apply /tmp/sonic_env.patch
cd MakefileRules
ln -sfn Machines/Sonic-gcc.def this_machine.def     # required, see MakefileRules/README
ln -sfn Machines/Sonic-gcc.env this_machine.env
```

Then build **on sonic13**:

```bash
ssh sonic13
cd /mnt/pfs/vaishak.p/spec-17e9207
source MakefileRules/this_machine.env
nice -n 5 ./MakefileRules/MakeParallel -j 24 all
```

Takes ~20 min wall at `-j 24` (2227 compiles, ~9200 s of CPU). Output lands in
`Evolution/Executables/EvolveHyperbolicSystem`.

### The one patch the build needs

Stock `Sonic-gcc.def` fails at link with `cannot find -llapack -lblas`. The
libraries are in `lapack-3.11.0/lib/`, but the `.def` omits the subdirectory:

```make
# was:  LAPACK_LIB = -Wl,-rpath,$(LAPACKHOME) -L$(LAPACKHOME) -llapack -lblas
LAPACK_LIB = -Wl,-rpath,$(LAPACKHOME)/lib -L$(LAPACKHOME)/lib -llapack -lblas
```

A `.bak` of the original sits next to it.

### Two further defects, deliberately left alone

`Sonic-gcc.def` also has `-L$(MPIHOME/lib)` and `-L$(UCXHOME/lib)` — the slash
is *inside* the parentheses, so both expand to empty and emit a bare `-L`. And
`UCXHOME = /usr` points at the system UCX rather than the custom
`ucx-1.13.1`. Neither blocks the build (MPI resolves via `-Wl,-rpath`), and
both were present for the 2023 catalogue build, so they are left unchanged to
keep the rebuild faithful. Fix only if you deliberately want to diverge.

## Environment: build vs run

The **build** uses `MakefileRules/this_machine.env`, which resolves to
gcc **12.2.0** and OpenMPI 4.1.4 from `/mnt/pfs/vaishak.p/soft`.

The **run** uses the `spec-env` module:

```bash
module load use.own && module load spec-env
```

which loads gcc **11.1.0**, openmpi/4.1.4, fftw/3.3.10, gsl/2.7.1,
lapack/3.11.0, hdf5/1.14.0-serial, petsc/3.18.4-nh5, make/make-4.4,
papi/7.0.0, ucx/1.13.1, hwloc/2.9.0, xpmem/2.6.5.

**The gcc version differs between build and run, and that is fine — verified.**
The maximum symbol version required across the binary and every SpEC `.so` is
`GLIBCXX_3.4.29`, and gcc-11.1.0's libstdc++ provides exactly up to
`GLIBCXX_3.4.29` (gcc-12.2.0 provides 3.4.30 but nothing needs it). `ldd`
reports 0 missing libraries under `spec-env`, and the binary runs. If SpEC ever
starts using a C++ feature that pushes it to 3.4.30, this breaks — re-check
with:

```bash
{ objdump -p Evolution/Executables/EvolveHyperbolicSystem; \
  find . -name '*.so' -exec objdump -p {} \; ; } |
  grep -oE 'GLIBCXX_[0-9.]+' | sort -Vu | tail -1
```

Note `spec-env` is a superset of the per-segment `bin/this_machine.env`, which
omits ucx/hwloc/xpmem. Prefer `spec-env`.

## Resuming a run

A segment that stopped on `WallClock` leaves a checkpoint, and
`MakeNextSegment.pl` (in `Support/Perl/`) will already have created the next
segment directory with all `*.input` files but **no `Run/`**. That empty next
segment is the thing to launch.

Check state across the tree with:

```bash
last=$(ls -d $EV/Lev${L}_[A-Z][A-Z] | sort | tail -1)
grep -m1 -o 'Termination condition [A-Za-z]*' $last/Run/SpEC.out
ls $last/Run/Checkpoints
```

Termination conditions seen, and what they mean:

| condition | meaning | resumable |
|---|---|---|
| `CommonHorizon` | merger reached — run is complete | n/a |
| `WallClock` | hit the queue limit, checkpoint written | **yes** |
| `PBandJTime` | SpEC's own wall-clock handoff (see `PBandJ.pm`) | **yes** |
| *(none)* | job killed, or segment prepared but never started | usually |
| `ShapeMapIsNearlySingular` | horizon shape map degenerated | **no** — needs parameter changes |

Before launching, repoint the segment's binary: the prepared segment's `SpEC`
symlink points at an **old** build (e.g. `../Lev4_AE/bin/EvolveHyperbolicSystem`)
which will fail with the undefined-symbol error above.

`Submit.sh` is generated by `MakeSubmit.py update` (reading `MakeSubmit.input`,
which is already present in a prepared segment) and is a normal Slurm script
(`-n 48`, `--ntasks-per-node 48`, `-p long`, 48 h) that sources
`bin/this_machine.env` and runs `EvolutionWrapper`, which handles the
segment chain and resubmission.

### Three things that each break the resume

Worked out the hard way on 2026-09-22; each produced a distinct failure.

**1. `bin/` is a symlink to an *earlier* segment's bin.** e.g.
`Lev4_AR/bin -> ../Lev4_AE/bin`. Writing into it therefore modifies state
shared with every other segment pointing there. Replace the symlink with a
private copy before changing anything:

```bash
cd $SEG && rm -f bin && cp -a ../Lev4_AE/bin ./bin
```

**2. Six helper executables must also come from the rebuild**, not just
`SpEC`. `EvolutionWrapper` shells out to `Nsubdomains` early (via
`SpEC::TotalNSubdomains`), and the archived one dies on the same
undefined-symbol error as the main binary, giving the cryptic
`Cannot close command '.../Nsubdomains -d ./GrDomain.input'`. Copy all of:

| helper | path in the build |
|---|---|
| `ApplyObservers` | `Support/ApplyObservers/Executables/` |
| `EvolveHyperbolicSystem` | `Evolution/Executables/` |
| `JoinH5` | `Support/H5Manip/` |
| `Nsubdomains` | `Support/DomainDataManip/` |
| `SurfaceToSpatialCoordMapFiles` | `Support/DomainDataManip/` |
| `TrajectoryToSpatialCoordMapFiles` | `Support/DomainDataManip/` |

Sanity check: `. bin/this_machine.env && ./bin/Nsubdomains -d ./GrDomain.input`
should print a number (80 for EccPrecDiff001, 81 for EccContPrecDiff002).

**3. `bin/this_machine.env` is not batch-safe as shipped.** It begins with
`source ~/.bashrc`, but `~/.bashrc` starts with

```bash
# If not running interactively, don't do anything
case $- in *i*) ;; *) return;; esac
```

so under `sbatch` neither `module` nor `conda` is ever defined, nothing loads,
and `EvolutionWrapper` dies with `Can't exec "mpirun"` at `Machines.pm:1583`.
Replace it with an explicit initialisation:

```bash
#!/bin/bash
. /mnt/pfs/vaishak.p/soft/modules-5.2.0/init/bash
. /mnt/pfs/vaishak.p/soft/anaconda3/etc/profile.d/conda.sh
export SOFT_ROOT=/mnt/pfs/vaishak.p/soft
export SOFTg_ROOT=/mnt/pfs/vaishak.p/soft-gcc
module purge
module load use.own
module load spec-env
```

Note this makes `mpirun` resolve to `soft-gcc/openmpi-4.1.4` while the binary
was *linked* against `soft/openmpi-4.1.4` — two distinct 4.1.4 builds. That is
fine and is what the original runs did: the module's `LD_LIBRARY_PATH`
overrides the rpath, so launcher and library both end up `soft-gcc`. Verified
identical behaviour for the archived 2023 binary.

Benign messages once running: `mca_btl_openib ... libosmcomp.so.3` and a vader
single-copy fallback. Both are transport warnings, not errors.

### Full resume sequence

```bash
SEG=<...>/Lev4_AR ; D=/mnt/pfs/vaishak.p/spec-17e9207
cd $SEG
rm -f bin && cp -a ../Lev4_AE/bin ./bin          # de-share
for f in Support/ApplyObservers/Executables/ApplyObservers \
         Evolution/Executables/EvolveHyperbolicSystem \
         Support/H5Manip/JoinH5 \
         Support/DomainDataManip/Nsubdomains \
         Support/DomainDataManip/SurfaceToSpatialCoordMapFiles \
         Support/DomainDataManip/TrajectoryToSpatialCoordMapFiles; do
  cp -f $D/$f bin/$(basename $f)
done
ln -sfn $D/Evolution/Executables/EvolveHyperbolicSystem SpEC
# ... write the batch-safe bin/this_machine.env shown above ...
ssh sonic13 "cd $SEG && ./bin/MakeSubmit.py update && sbatch Submit.sh"
```

Success looks like `#  And so it begins...` followed by
`t=<checkpoint time>, step=<n> (just starting evolution)` in `Run/SpEC.out`,
and 48 `SpEC` processes on the allocated node.

## Sizing

These runs use **48 MPI ranks on one node**. EccPrecDiff002 decomposes into 82
subdomains, so a single node is the right size and more would waste cores.
Nodes have 96 cores / 250 GB.

## Things to watch

- `/mnt/pfs` was **96% full** (7.7 TB of 175 TB free). Check before long runs.
- `slurmd` is `failed` on **sonic13** specifically, though it runs on sonic2/4.
  sonic13 is still fine for interactive builds; just don't expect jobs there.
- Other users share the cluster (`rohan.raha`, `kaushik.pa`, `prayush`,
  `soumen.bas`). Check `squeue`/`sinfo -t idle` from sonic13 first.

## Known-bad run

`ICTSEccParallel13` Lev3 is **not** resumable. Three separate attempts
(2024-03-08, 2024-12-01, 2024-12-14) on three different nodes all died at
exactly `t = 18897 M` with a floating-point exception on **rank 44**, ~280 M
short of merger. It is a deterministic numerical failure, not a queue problem.
