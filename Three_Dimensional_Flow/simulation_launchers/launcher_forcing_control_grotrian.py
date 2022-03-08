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
        famplitude=512.0):
    args = []
    if type(base_dir) == type(None):
        if hostname.startswith('alan') or hostname.startswith('uran'):
            base_dir = '/scratch01.local/falvarez/Kflow/forcing_control/ryz{0}'.format(ratio_yz)
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
    work_dir = os.path.join(base_dir, 'N{0:0>4d}_rxy{1}_ryz{2}_famplitude_{3:.3f}'.format(N, ratio_xy, ratio_yz,famplitude))
    args += ['--wd', work_dir]
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    Nz = int(N / ratio_yz)
    dt = 2*0.009/N
    assert(Nz % 2 == 0)
    c.launch(args = args + [
        '--niter_todo', '{0}'.format(niterations),
        '--niter_out', '{0}'.format(niterations),
        '--niter_stat', '16',
        '--np', '64',
        '--famplitude', '{0}'.format(famplitude),
        '--nx', '{0}'.format(int(ratio_xy*N)),
        '--ny', '{0}'.format(N),
        '--nz', '{0}'.format(int(N / ratio_yz)),
        '--nu', '{0}'.format(0.5),
        '--Lx', '{0}'.format(ratio_xy*2),
        '--Lz', '{0}'.format(2 / ratio_yz),
        '--niter_DB', '32',
        '--dt', '{0}'.format(dt),
        '--simname', simname,
        '--forcing_type', 'Kolmogorov_and_drag',
        '--njobs', '400',
        '--fftw_plan_rigor', 'FFTW_MEASURE'])
        #'--src-wd', '/home/btph/bt307732/simulations/kolmogorov_flow/N0064_rxy3_ryz16_famplitude_0.5',
        #'--src-simname','N0064_rxy3_ryz16',
        #'--src-iteration','32768',
        #'--overwrite-src-parameters'])
    return None
import numpy as np
if __name__ == '__main__':
    Re = np.linspace(10,26,16)
    forcing = Re**2/(2*np.pi)**3*0.25
    for f in forcing:
    #launch(N = 48, ratio_yz = 1, niterations = 128)
    #launch(N = 48, ratio_yz = 2, niterations = 128)
        launch(N = 128, ratio_yz = 1, niterations = 1024, famplitude=f)

