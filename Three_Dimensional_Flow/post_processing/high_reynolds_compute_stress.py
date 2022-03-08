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
scratch = 1
path = '/scratch0{0}.local/falvarez/Kflow/forcing_control/high_reynolds/ryz{1}'.format(scratch,ryz)

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
#Reynolds =np.linspace(10,26,32)
Reynolds=np.logspace(5,8,num=32, base=2.0)
forcing_amplitudes= Reynolds**2/(2*np.pi)**3*0.25

for i,f in enumerate(forcing_amplitudes):
    famplitude = f
    #path = '/localdisk/bt307732/simulations/kolmogorov_flow/ryz{0}'.format(ryz)
    work_dir = os.path.join(path,'N{0:0>4d}_rxy{1}_ryz{2}_famplitude_{3:.3f}'.format(N,rxy,ryz,famplitude))
    simname = 'N{0:0>4d}_rxy{1}_ryz{2}'.format(N,rxy,ryz)
    #try:
    cc = NSReader(simname=simname, work_dir=work_dir, kspace_on=True)
    cc2 = DB2D_reader(simname=simname, work_dir=work_dir, read_only_cache=False)
    #except:
        #cc2 = DB2D_reader(simname=simname, work_dir=work_dir, read_only_cache=True)
    iteration =cc2.get_data_file()['iteration'][()]
    file = h5py.File(os.path.join(work_dir,"reynolds_stresses_{}.h5".format(simname)), 'w')
    path_h5_file="sigma/{0}_iteration/{1}".format(sigma,iteration)
    if path_h5_file in file:
        print('esta')
    else:
        print('no esta')
        vor3Dk = cc.get_cvorticity(iteration=iteration)
        vel3Dk = cc.get_cvelocity(iteration=iteration)

        if sigma>0:
            kx, ky = cc2.get_kx_ky()
            Kx,Ky = np.meshgrid(kx,ky)
            sigma= sigma*dx
            Kernel = np.exp(-sigma**2*(Kx**2+Ky**2)/2)
            Kernel2 = np.ones(Kernel.shape) - Kernel
            vor3Dk[:,0,...] = Kernel2[...,None]*vor3Dk[:,0,...]
            vel3Dk[:,0,...] = Kernel2[...,None]*vel3Dk[:,0,...]
        else:
            vor3Dk[:,0,...] = 0
            vel3Dk[:,0,...] = 0

        vor3Dk = vor3Dk.transpose((1,0,2,3))
        vel3Dk = vel3Dk.transpose((1,0,2,3))

        vor3Dr = np.fft.irfftn(vor3Dk, axes=(0,1,2))*nx*ny*nz
        vel3Dr = np.fft.irfftn(vel3Dk, axes=(0,1,2))*nx*ny*nz

        tau_xx = vel3Dr[...,0]*vel3Dr[...,0]
        tau_yy = vel3Dr[...,1]*vel3Dr[...,1]
        tau_zz = vel3Dr[...,2]*vel3Dr[...,2]
        tau_xy = vel3Dr[...,0]*vel3Dr[...,1]
        tau_zx = vel3Dr[...,2]*vel3Dr[...,0]
        tau_yz = vel3Dr[...,1]*vel3Dr[...,2]

        phi_xx = np.fft.rfftn(tau_xx, axes=(0,1,2))
        phi_yy = np.fft.rfftn(tau_yy, axes=(0,1,2))
        phi_zz = np.fft.rfftn(tau_zz, axes=(0,1,2))
        phi_xy = np.fft.rfftn(tau_xy, axes=(0,1,2))
        phi_zx = np.fft.rfftn(tau_zx, axes=(0,1,2))
        phi_yz = np.fft.rfftn(tau_yz, axes=(0,1,2))

        file.create_dataset(path_h5_file+'/complex/phi_xx', data=phi_xx, shape=phi_xx.shape, dtype=phi_xx.dtype)
        file.create_dataset(path_h5_file+'/complex/phi_yy', data=phi_yy, shape=phi_yy.shape, dtype=phi_yy.dtype)
        file.create_dataset(path_h5_file+'/complex/phi_zz', data=phi_zz, shape=phi_zz.shape, dtype=phi_zz.dtype)
        file.create_dataset(path_h5_file+'/complex/phi_xy', data=phi_xy, shape=phi_xy.shape, dtype=phi_xy.dtype)
        file.create_dataset(path_h5_file+'/complex/phi_zx', data=phi_zx, shape=phi_zx.shape, dtype=phi_zx.dtype)
        file.create_dataset(path_h5_file+'/complex/phi_yz', data=phi_yz, shape=phi_yz.shape, dtype=phi_yz.dtype)

        tau_xx =np.fft.irfftn(phi_xx[0,...], axes=(0,1))*nx*ny
        tau_yy =np.fft.irfftn(phi_yy[0,...], axes=(0,1))*nx*ny
        tau_zz =np.fft.irfftn(phi_zz[0,...], axes=(0,1))*nx*ny
        tau_xy =np.fft.irfftn(phi_xy[0,...], axes=(0,1))*nx*ny
        tau_zx =np.fft.irfftn(phi_zx[0,...], axes=(0,1))*nx*ny
        tau_yz =np.fft.irfftn(phi_yz[0,...], axes=(0,1))*nx*ny

        file.create_dataset(path_h5_file+'/real/tau_xx', data=tau_xx, shape=tau_xx.shape, dtype=tau_xx.dtype)
        file.create_dataset(path_h5_file+'/real/tau_yy', data=tau_yy, shape=tau_yy.shape, dtype=tau_yy.dtype)
        file.create_dataset(path_h5_file+'/real/tau_zz', data=tau_zz, shape=tau_zz.shape, dtype=tau_zz.dtype)
        file.create_dataset(path_h5_file+'/real/tau_xy', data=tau_xy, shape=tau_xy.shape, dtype=tau_xy.dtype)
        file.create_dataset(path_h5_file+'/real/tau_zx', data=tau_zx, shape=tau_zx.shape, dtype=tau_zx.dtype)
        file.create_dataset(path_h5_file+'/real/tau_yz', data=tau_yz, shape=tau_yz.shape, dtype=tau_yz.dtype)

    file.close()
