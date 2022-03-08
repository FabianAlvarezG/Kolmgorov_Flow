import sys
import numpy as np
from reader import DB2D_reader
from TurTLE_addons import NSReader
import matplotlib.pyplot as plt
import h5py
import os
import TurTLE

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
scratch = 3




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
Reynolds=np.logspace(8,9,num=32, base=2.0)
forcing_amplitudes= Reynolds**2/(2*np.pi)**3*0.25

for i,f in enumerate(forcing_amplitudes):
    famplitude = f
    #path = '/localdisk/bt307732/simulations/kolmogorov_flow/ryz{0}'.format(ryz)
    simname = 'N0128_rxy{0}_ryz{1}'.format(rxy,ryz)
    if TurTLE.host_info['type'] == 'pc':
        work_dir = '/home/falvarez/ttf/temp/very_high_reynolds/ryz{0}/N0128_rxy3_ryz4_{1}'.format(ryz, i+1)
    else:
        work_dir = '/scratch03.local/falvarez/Kflow/forcing_control/very_high_reynolds/ryz4/N0128_rxy3_ryz4_{0}'.format(i+1)
    checkpoint = 'N0128_rxy3_ryz4_checkpoint_0.h5'
    cc = NSReader(simname=simname, work_dir=work_dir, kspace_on=True)
    cc2 = DB2D_reader(simname=simname, work_dir=work_dir, kspace_on=True)
    cc.do_plots()

    kx, ky = cc2.get_kx_ky()
    Kx,Ky = np.meshgrid(kx,ky)

    data_file = cc.get_data_file()
    iteration = data_file['iteration'][()]

    nx = data_file['parameters/nx'][()]
    ny = data_file['parameters/ny'][()]

    vel2Dk = cc2.read_filtered_DB_iteration(iteration=iteration)
    vel2Dr = cc2.c2r(vel2Dk)
    cp_file = h5py.File(os.path.join(work_dir, checkpoint), 'r')

    vor3Dk = cc.get_cvorticity(iteration=iteration)
    vel3Dk = cc.get_cvelocity(iteration=iteration)


    full_velr = cp_file['velocity/real/{0}'.format(iteration)][()]
    velr = np.mean(full_velr, axis = 0)

    #vor3Dk = vor3Dk.transpose((1,0,2,3))
    #vel3Dk = vel3Dk.transpose((1,0,2,3))

    #vor3Dr = np.fft.irfftn(vor3Dk, axes=(0,1,2))
    vel3Dr = full_velr-velr

    tau_xx = np.mean(vel3Dr[...,0]*vel3Dr[...,0], axis=0)/(forcing_amplitudes[i]/0.25)**2
    tau_yy = np.mean(vel3Dr[...,1]*vel3Dr[...,1], axis=0)/(forcing_amplitudes[i]/0.25)**2
    tau_zz = np.mean(vel3Dr[...,2]*vel3Dr[...,2], axis=0)/(forcing_amplitudes[i]/0.25)**2
    tau_xy = np.mean(vel3Dr[...,0]*vel3Dr[...,1], axis=0)/(forcing_amplitudes[i]/0.25)**2
    tau_zx = np.mean(vel3Dr[...,2]*vel3Dr[...,0], axis=0)/(forcing_amplitudes[i]/0.25)**2
    tau_yz = np.mean(vel3Dr[...,1]*vel3Dr[...,2], axis=0)/(forcing_amplitudes[i]/0.25)**2

    phi_xx = np.fft.rfft2(tau_xx, axes=(0,1))
    phi_yy = np.fft.rfft2(tau_yy, axes=(0,1))
    phi_zz = np.fft.rfft2(tau_zz, axes=(0,1))
    phi_xy = np.fft.rfft2(tau_xy, axes=(0,1))
    phi_zx = np.fft.rfft2(tau_zx, axes=(0,1))
    phi_yz = np.fft.rfft2(tau_yz, axes=(0,1))

    file = h5py.File(os.path.join(work_dir,"reynolds_stresses_{}.h5".format(simname)), 'w')
    path_h5_file="iteration/{0}".format(iteration)

    gamma_x_k = 1j*Kx*phi_xx+1j*Ky*phi_xy
    gamma_y_k = 1j*Kx*phi_xy+1j*Ky*phi_yy
    gamma_z_k = 1j*Kx*phi_zx+1j*Ky*phi_yz

    varphi_k= 1j*Kx*gamma_y_k - 1j*Ky*gamma_x_k
    varphi = np.fft.irfft2(varphi_k, axes=(0,1))*nx*ny/(forcing_amplitudes[i]/0.25)**2

    file.create_dataset(path_h5_file+'/real/tau_xx', data=tau_xx, shape=tau_xx.shape, dtype=tau_xx.dtype)
    file.create_dataset(path_h5_file+'/real/tau_yy', data=tau_yy, shape=tau_yy.shape, dtype=tau_yy.dtype)
    file.create_dataset(path_h5_file+'/real/tau_zz', data=tau_zz, shape=tau_zz.shape, dtype=tau_zz.dtype)
    file.create_dataset(path_h5_file+'/real/tau_xy', data=tau_xy, shape=tau_xy.shape, dtype=tau_xy.dtype)
    file.create_dataset(path_h5_file+'/real/tau_zx', data=tau_zx, shape=tau_zx.shape, dtype=tau_zx.dtype)
    file.create_dataset(path_h5_file+'/real/tau_yz', data=tau_yz, shape=tau_yz.shape, dtype=tau_yz.dtype)

    file.create_dataset(path_h5_file+'/complex/phi_xx', data=phi_xx, shape=phi_xx.shape, dtype=phi_xx.dtype)
    file.create_dataset(path_h5_file+'/complex/phi_yy', data=phi_yy, shape=phi_yy.shape, dtype=phi_yy.dtype)
    file.create_dataset(path_h5_file+'/complex/phi_zz', data=phi_zz, shape=phi_zz.shape, dtype=phi_zz.dtype)
    file.create_dataset(path_h5_file+'/complex/phi_xy', data=phi_xy, shape=phi_xy.shape, dtype=phi_xy.dtype)
    file.create_dataset(path_h5_file+'/complex/phi_zx', data=phi_zx, shape=phi_zx.shape, dtype=phi_zx.dtype)
    file.create_dataset(path_h5_file+'/complex/phi_yz', data=phi_yz, shape=phi_yz.shape, dtype=phi_yz.dtype)


    file.close()
