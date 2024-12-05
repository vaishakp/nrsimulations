from waveformtools.sxs.prepare_waveforms import PrepareSXSWaveform
from pathlib import Path
import config

if config.print_verbosity!=2:
    from config.verbosity import levels

    lv = levels()

    lv.set_print_verbosity(2)


sims1 = {
     "ICTSEccParallel01" : [2, 3],
#    "ICTSEccParallel03": [2, 3],
#    "ICTSEccParallel04": [2, 3],
#    "ICTSEccParallel05": [2, 3],
#    "ICTSEccParallel06": [2, 3],
#    "ICTSEccParallel07": [2, 3],
#    "ICTSEccParallel08": [2, 3],
#    "ICTSEccParallel09": [2, 3],
#    "ICTSEccParallel10": [2, 3],
#    "ICTSEccParallel11": [2, 3],
#    "ICTSEccParallel12": [2, 3],
#    "ICTSEccParallel14": [2, 3],
#    "ICTSEccParallel15": [2, 3],
#    "ICTSEccParallel16": [2, 3],
#    "ICTSEccParallel17": [2, 3],
}


prefix_dir = "/mnt/pfs/vaishak.p/sims/SpEC/gcc/"
sims1_dir = Path(prefix_dir + "/bfi/ICTSEccParallel/")
eccs = [0]
success_sims = []
flag = False

for sim in sims1.keys():
    for lev in sims1[sim]:
        print("Sim", sim, "Lev", lev)

        try:
            wfp = PrepareSXSWaveform(sim_name=sim, sim_dir=sims1_dir/sim, lev=lev)
            flag = wfp.prepare_waveform()

        except:
            flag=False
            pass
            
        if flag:
            success_sims.append([sim, lev])


print(success_sims)
