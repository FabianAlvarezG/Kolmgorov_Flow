import numpy as np
import h5py
import matplotlib.pyplot as plt
from TurTLE_addons import NSReader
import os
from reader import DB2D_reader
import sys

rxy = 3
ryz = 4
i = int(sys.argv[1])

Reynolds=np.logspace(8,9,num=32,base=2.0)
forcing_amplitudes= Reynolds**2/(2*np.pi)**3*0.25
#f = np.array([forcing_amplitudes[16],forcing_amplitudes[23]])

simname = 'N0128_rxy{0}_ryz{1}'.format(rxy,ryz)
#work_dir = '/home/falvarez/ttf/temp/turbulence_onset/ryz4/N0128_rxy3_ryz4_famplitude_{0}'.format(i)
work_dir = '/scratch03.local/falvarez/Kflow/forcing_control/very_high_reynolds/ryz4/N0128_rxy3_ryz4_{0}'.format(i)

checkpoint = 'N0128_rxy3_ryz4_checkpoint_0.h5'

cc = NSReader(simname=simname, work_dir=work_dir, kspace_on=True)
cc2 = DB2D_reader(simname=simname, work_dir=work_dir, kspace_on=True)

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

file.close()

### Computing vortex stretching ####
velk = cc2.read_filtered_DB_iteration(iteration=iteration, dataset='velocity')
vork = cc2.read_filtered_DB_iteration(iteration=iteration, dataset='vorticity')
vorr = cc2.c2r(vork)
uzk = velk[...,2]
dx_uzk = 1j*Kx*uzk
dy_uzk = 1j*Ky*uzk
dx_uzr = cc2.c2r(dx_uzk)
dy_uzr = cc2.c2r(dy_uzk)
chi = (vorr[...,0]*dx_uzr+vorr[...,1]*dy_uzr)/(forcing_amplitudes[i]/0.25)**2

### Computing laplacian ####
N = 128
a = 1/rxy
sk =cc2.read_filtered_DB_iteration(iteration=iteration, dataset='stream_function')
nabla2_sr = np.fft.irfft2((Kx**2+Ky**2)**2*sk)*int(N*N/a)/(forcing_amplitudes[i]/0.25)/(Reynolds[i]**2/(2*np.pi)**3)

print(abs(nabla2_sr).mean(),abs(chi).mean(), abs(varphi).mean())

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
ax1.imshow(varphi, cmap='RdBu', origin='lower')
ax2.imshow(chi,cmap='RdBu', origin='lower')
ax3.imshow(nabla2_sr,cmap='RdBu', origin='lower')
ax1.set_title(r'$\phi$')
ax2.set_title(r'$\tilde{\omega}_{\bot}\cdot\nabla \tilde{u}_z $')
fig.text(0.4, 0.65,
         r'$Re = {0:.2f}$'.format(Reynolds[i]),
         style = 'italic',
         fontsize = 12)

#fig.savefig('vertical_velocities_turbulence_onset_{0}.png'.format(i), format='png')
fig.savefig('vertical_velocities_very_high_reynolds_{0}.png'.format(i), format='png')
cp_file.close()
data_file.close()
