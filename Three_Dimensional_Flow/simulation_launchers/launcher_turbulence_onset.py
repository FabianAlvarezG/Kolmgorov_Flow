import os
import h5py
from simulator import simulator
from TurTLE import host_info
import getpass

from simulator import username, homefolder, hostname

cluster_name = None

def launch(
        N = 256,
        ratio_xy = 3,
        ratio_yz = 1,
        base_dir = None,
        minutes = None,
        niterations = 1024,
        famplitude=512.0,
        index = 0):
    args = []
    if type(base_dir) == type(None):
        if hostname.startswith('alan') or hostname.startswith('uran'):
            base_dir = '/scratch01.local/falvarez/Kflow/forcing_control/turbulence_onset/ryz{0}'.format(ratio_yz)
            initial_base_dir = False
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            ntpp =1
            np = 32
            N =128
    else:
        initial_base_dir = True
    if type(minutes) != type(None):
        args += ['--minutes', '{0}'.format(minutes)]
    c = simulator()
    simname = 'N{0:0>4d}_rxy{1}_ryz{2}'.format(N, ratio_xy, ratio_yz)
    print(base_dir)
    work_dir = os.path.join(base_dir, 'N{0:0>4d}_rxy{1}_ryz{2}_famplitude_{3}'.format(N, ratio_xy, ratio_yz,index))
    args += ['--wd', work_dir]
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    Nz = int(N / ratio_yz)
    dt = 0.006/N
    assert(Nz % 2 == 0)
    c.launch(args = args + [
        '--niter_todo', '{0}'.format(niterations),
        '--niter_out', '{0}'.format(niterations),
        '--niter_stat', '256',
        '--np', '32',
        '--famplitude', '{0}'.format(famplitude),
        '--nx', '{0}'.format(int(ratio_xy*N)),
        '--ny', '{0}'.format(N),
        '--nz', '{0}'.format(int(N / ratio_yz)),
        '--nu', '{0}'.format(0.5),
        '--Lx', '{0}'.format(ratio_xy*2),
        '--Lz', '{0}'.format(2 / ratio_yz),
        '--niter_DB', '256',
        '--dt', '{0}'.format(dt),
        '--simname', simname,
        '--forcing_type', 'Kolmogorov_and_drag',
        '--njobs', '100',
        '--fftw_plan_rigor', 'FFTW_MEASURE'])
        #'--src-wd', os.path.join('/scratch03.local/falvarez/Kflow/forcing_control/detailed_high_reynolds/ryz{0}'.format(ratio_yz),'N{0:0>4d}_rxy{1}_ryz{2}_famplitude_{3:.3f}'.format(N, ratio_xy, ratio_yz,famplitude)),
        #'--src-simname','N0128_rxy3_ryz4',
        #'--src-iteration','1638400',
        #'--overwrite-src-parameters'])
    return None

import numpy as np
if __name__ == '__main__':
    Re = np.logspace(8,9,num=32, base=2.0)
    For = Re**2/(2*np.pi)**3*0.25
    forcing = np.array([For[25]])
    for i,f in enumerate(forcing):
    #launch(N = 48, ratio_yz = 1, niterations = 128)
    #launch(N = 48, ratio_yz = 2, niterations = 128)
        launch(N = 128, ratio_yz = 4, niterations = 1024, famplitude=f, index=2)
