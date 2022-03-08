import os
import h5py
from simulator_average_xyz import simulator
from TurTLE import host_info
import getpass

from simulator_average_xyz import username, homefolder, hostname

cluster_name = None

def launch(
        N = 128,
        ratio_xy = 3,
        ratio_yz = 1,
        base_dir = None,
        minutes = None,
        niterations = 1024):
    args = []
    if type(base_dir) == type(None):
        if hostname.startswith('alan') or hostname.startswith('uran'):
            base_dir = '/scratch02.local/falvarez/Kflow'
            initial_base_dir = False
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            ntpp = 1
            np = 256
            N = 256
        else:
            base_dir = os.path.join(homefolder, 'lscratch/KFlow_2D3D')
            initial_base_dir = False
            N = 48
            np = 8
            ntpp = 1
    else:
        initial_base_dir = True
    if type(minutes) != type(None):
        args += ['--minutes', '{0}'.format(minutes)]
    dt = 0.1/N
    c = simulator()
    simname = 'N{0:0>4d}_rxy{1}_ryz{2}'.format(N, ratio_xy, ratio_yz)
    print(base_dir)
    work_dir = os.path.join(base_dir, 'N{0:0>4d}_average_xyz'.format(N))
    args += ['--wd', work_dir]
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    Nz = int(N / ratio_yz)
    assert(Nz % 2 == 0)
    c.launch(args = args + [
        '--niter_todo', '{0}'.format(niterations),
        '--niter_out', '{0}'.format(niterations),
        '--niter_stat', '16',
        '--nx', '{0}'.format(int(ratio_xy*N)),
        '--ny', '{0}'.format(N),
        '--nz', '{0}'.format(int(N / ratio_yz)),
        '--Lx', '{0}'.format(ratio_xy*2),
        '--Lz', '{0}'.format(2 / ratio_yz),
        '--niter_DB', '128',
        '--simname', simname,
        '--forcing_type', 'Kolmogorov',
        '--njobs', '12',
        '--fftw_plan_rigor', 'FFTW_MEASURE',
        '--np','{0}'.format(np),
        '--ntpp','{0}'.format(ntpp),
        '--dt','{0}'.format(dt)
        #'--src-simname','N0256_rxy3_ryz1',
        #'--src-wd','/home/falvarez/ttf/KFlow/mu_0.3/N0256_rxy3_ryz1',
        #'--src-iteration','43008'
        ])
    return None

if __name__ == '__main__':
    launch(ratio_yz = 1,niterations = '1024')
    #launch(N = 48, ratio_yz = 2, niterations = 128)
    #launch(N = 48, ratio_yz = 4, niterations = 128)
