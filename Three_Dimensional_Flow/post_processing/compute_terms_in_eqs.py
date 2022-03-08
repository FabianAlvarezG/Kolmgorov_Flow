import sys
import numpy as np
from reader import DB2D_reader
from TurTLE_addons import NSReader
import matplotlib.pyplot as plt
import h5py
import os

'''
plt.rcParams.update({
    'text.usetex' : False,
    'font.family' : 'sans-serif',
    'font.sans-serif' : ['Arial'],
    'mathtext.fallback_to_cm' : True,
    })
    '''
plt.rcParams.update({'font.size': 12})


N = 128
rxy = 3
ryz = 4
scratch = 4
path = '/scratch0{0}.local/falvarez/Kflow/forcing_control/low_reynolds/ryz{1}'.format(scratch,ryz)

a = 1/rxy
### Function to project ###
dx = 2*np.pi/N
y = np.arange(0,N)*dx
x = np.arange(0,rxy*N)*dx
X,Y = np.meshgrid(x,y)
mu = 0.0
nx = N*rxy
ny = N
nz = int(N/ryz)


sigma=0
Reynolds =np.linspace(10,26,32)
forcing_amplitudes= Reynolds**2/(2*np.pi)**3*0.25

for i,f in enumerate(forcing_amplitudes):
    famplitude = f
    #path = '/localdisk/bt307732/simulations/kolmogorov_flow/ryz{0}'.format(ryz)
    work_dir = os.path.join(path,'N{0:0>4d}_rxy{1}_ryz{2}_famplitude_{3:.3f}'.format(N,rxy,ryz,famplitude))
    simname = 'N{0:0>4d}_rxy{1}_ryz{2}'.format(N,rxy,ryz)
    #try:
    cc = NSReader(simname=simname, work_dir=work_dir, read_only_cache=False)
    cc2 = DB2D_reader(simname=simname, work_dir=work_dir, read_only_cache=False)
    #except:
        #cc2 = DB2D_reader(simname=simname, work_dir=work_dir, read_only_cache=True)
    iteration =cc2.get_data_file()['iteration'][()]
    file = h5py.File(os.path.join(work_dir,"reynolds_stresses_{}.h5".format(simname)), 'a')
    path_h5_file="sigma/{0}_iteration/{1}".format(sigma,iteration)
    if path_h5_file in file:
        print('esta')
        vor3Dk = cc.get_cvorticity(iteration=iteration)
        vel3Dk = cc.get_cvelocity(iteration=iteration)
        kx, ky = cc2D.get_kx_ky()
        Kx,Ky = np.meshgrid(kx,ky)
        txx = file.get(os.path.join(path_h5_file,'/tau_xx'))
        tyy = file.get(os.path.join(path_h5_file,'/tau_yy'))
        tzz = file.get(os.path.join(path_h5_file,'/tau_zz'))
        tyx = file.get(os.path.join(path_h5_file,'/tau_yx'))
        tzx = file.get(os.path.join(path_h5_file,'/tau_zx'))
        tyz = file.get(os.path.join(path_h5_file,'/tau_yz'))
    else:
        print('no esta')
    file.close()
