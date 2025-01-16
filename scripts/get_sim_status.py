import numpy as np
import os
from pathlib import Path
import config
import matplotlib.pyplot as plt

config.conf_matplolib()
import yaml
from waveformtools.sxs.discover_sims import SimulationExplorer
from datetime import datetime
import subprocess
from pathlib import Path

GIT_DIR = Path(
    subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    ).stdout[:-1]
)


status_file_path = GIT_DIR / "SimulationStatus.md"


if config.print_verbosity < 3:
    from config.verbosity import levels

    vl = levels()
    vl.set_print_verbosity(2)

if os.path.exists(GIT_DIR / "scripts/simulations_to_parse.yaml"):

    with open(GIT_DIR / "scripts/simulations_to_parse.yaml", "r") as f:
        cf = yaml.safe_load(f)

else:

    cf = {
        "Cases": {
            "ICTSEccParallel": "/mnt/pfs/vaishak.p/sims/SpEC/gcc/bfi/ICTSEccParallel",
            "EccContPrecDiff": "/mnt/pfs/vaishak.p/sims/SpEC/gcc/bfi/EccContPrecDiff",
            "EccPrecDiff": "/mnt/pfs/vaishak.p/sims/SpEC/gcc/bfi/EccPrecDiff",
        },
        "BFI_HOME": "/mnt/pfs/vaishak.p/sims/SpEC/BFI/BFI",
        "WAVEFORMS_OUT_DIR": "/mnt/pfs/vaishak.p/Projects/Codes/waveforms",
    }


with open(status_file_path, "w+") as sf:
    sf.write("# Simulation live status\n\n")


for project_name, project_path in cf["Cases"].items():
    se = SimulationExplorer(
        search_dir=project_path,
        bfi_home_dir=cf["BFI_HOME"],
        bfi_project_name=project_name,
        prepared_waveforms_dir=cf["WAVEFORMS_OUT_DIR"],
    )

    se.search_simulations()
    se.discover_segments()
    se.discover_levels()
    se.delete_empty_sims()
    se.construct_simulation_status()

    df = se.sim_status

    dfm = df.to_markdown().split("\n")

    with open(status_file_path, "a") as sf:

        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        sf.write(f"## {project_name} \n\n")

        sf.write(f"### Last updated: \t {now} \n\n")

        sf.write("### Status\n\n")

        for line in dfm:

            sf.write(line)
            sf.write("\n")

        sf.write("\n\n")
