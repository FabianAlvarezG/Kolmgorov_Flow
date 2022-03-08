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
scratch = 3
path = '/scratch0{0}.local/falvarez/Kflow/forcing_control/very_high_reynolds/ryz{1}'.format(scratch,ryz)

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

def Rc0(a):
    return np.sqrt(2)*(1+a**2)/np.sqrt(1-a**2)

def Rc(a,u):
    return (u + 2*a**2*u + a**4*u + np.sqrt((1 + a**2)**2*(-2*a**6 + u**2 + 2*a**2*u**2 + a**4*(2 + u**2))))/(a**2*(1 - a**2))

def c(R,a,u):
    return -((R + 2*a**2*R - a**2*R**2*u + np.sqrt(-(R**2*(-1 - 4*a**2 - 4*a**4 - 2*a**4*R**2 + 2*a**6*R**2 + 2*a**2*R*u + 4*a**4*R*u - a**4*R**2*u**2))))/(a*(-1 + a**2)*R**2))

def lc(a,u):
    return (2*(1 + a**2)**2*(2*a**6 - u**2 - a**4*(2 + u**2) + a**2*u*(-2*u + np.sqrt(-2*(-1 + a**2)*(1 + a**2)**2 + ((1 + a**2)**4*u**2)/a**4))))/(a**4*(-1 + a**2))

def gamma(a,u):
    return (-16*a**4*(1 + a**2)**4*(-1 - 17*a**2 - 16*a**4 + 32*a**6)*((1/a + a)**2*u + np.sqrt(2*(1 - a**2)*(1 + a**2)**2 + (1/a + a)**4*u**2)))/((1 - a**2)*(1 + 4*a**2)**2*((1 + a**2)**2*u + a**2*np.sqrt(-2*(-1 + a**2)*(1 + a**2)**2 + ((1 + a**2)**4*u**2)/a**4))**2)

def InnerProduct(vector, psi):
    integral = dx*dx*np.sum(np.conjugate(psi)*vector)/(4*np.pi**2/a)
    return integral

R = 5/3
psi_0 = (np.exp(1j*Y)+np.exp(-1j*Y))/np.sqrt(2)
psi_a0  = (np.exp(1j*Y) - np.exp(-1j*Y) + c(R,a,mu))*np.exp(1j*X*a)/np.sqrt(2+c(R,a,mu)**2)
psi_a1 = (np.exp(1j*Y) + np.exp(-1j*Y))*np.exp(1j*X*a)/np.sqrt(2)
psi_a2 = (c(R,a,mu)*(np.exp(1j*Y) - np.exp(-1j*Y))-2)*np.exp(1j*X*a)/np.sqrt(4+2*c(R,a,mu)**2)

psi_2a0  = (np.exp(1j*Y) - np.exp(-1j*Y) + c(R,2*a,mu))*np.exp(1j*X*a*2)/np.sqrt(2+c(R, 2*a,mu)**2)
psi_2a1 = (np.exp(1j*Y) + np.exp(-1j*Y))*np.exp(1j*X*a*2)/np.sqrt(2)
psi_2a2 = (c(R, 2*a,mu)*(np.exp(-1j*Y) - np.exp(-1j*Y))-2)*np.exp(1j*X*a*2)/np.sqrt(4+2*c(R, 2*a,mu)**2)

#### Dagger eigenvectors

psi_a0_dagger = (np.exp(1j*Y) - np.exp(-1j*Y) + (1-a**2)/a**3*Rc0(a)**2/Rc(a, mu))*np.exp(1j*X*a)


sigma=0
#Reynolds =np.linspace(10,26,32)
Reynolds=np.logspace(8,9,num=32,base=2.0)
forcing_amplitudes= Reynolds**2/(2*np.pi)**3*0.25

amplitudes_0 = np.copy(Reynolds)
amplitudes_a0 = np.copy(Reynolds)
amplitudes_a0_dagger = np.copy(Reynolds)

for i,f in enumerate(forcing_amplitudes):
    famplitude = f
    #path = '/localdisk/bt307732/simulations/kolmogorov_flow/ryz{0}'.format(ryz)
    simname = 'N0128_rxy{0}_ryz{1}'.format(rxy,ryz)
    work_dir = '/scratch03.local/falvarez/Kflow/forcing_control/very_high_reynolds/ryz4/N0128_rxy3_ryz4_{0}'.format(i+1)
    #try:
    cc = NSReader(simname=simname, work_dir=work_dir, kspace_on=True)
    cc2 = DB2D_reader(simname=simname, work_dir=work_dir, read_only_cache=False)
    cc.do_plots()
    #except:
        #cc2 = DB2D_reader(simname=simname, work_dir=work_dir, read_only_cache=True)
    iteration =cc2.get_data_file()['iteration'][()]
    file = h5py.File(os.path.join(work_dir,"reynolds_stresses_{}.h5".format(simname)), 'r')
    path_h5_file="iteration/{0}".format(iteration)
    if path_h5_file in file:
        print('data is already saved')

        kx, ky = cc2.get_kx_ky()
        Kx,Ky = np.meshgrid(kx,ky)

        tau_xx = file.get(os.path.join(path_h5_file,'real/tau_xx'))
        tau_yy = file.get(os.path.join(path_h5_file,'real/tau_yy'))
        tau_zz = file.get(os.path.join(path_h5_file,'real/tau_zz'))
        tau_xy = file.get(os.path.join(path_h5_file,'real/tau_xy'))
        tau_zx = file.get(os.path.join(path_h5_file,'real/tau_zx'))
        tau_yz = file.get(os.path.join(path_h5_file,'real/tau_yz'))

        phi_xx = file.get(os.path.join(path_h5_file,'complex/phi_xx'))
        phi_yy = file.get(os.path.join(path_h5_file,'complex/phi_yy'))
        phi_zz = file.get(os.path.join(path_h5_file,'complex/phi_zz'))
        phi_xy = file.get(os.path.join(path_h5_file,'complex/phi_xy'))
        phi_zx = file.get(os.path.join(path_h5_file,'complex/phi_zx'))
        phi_yz = file.get(os.path.join(path_h5_file,'complex/phi_yz'))

        gamma_x_k = 1j*Kx*phi_xx+1j*Ky*phi_xy
        gamma_y_k = 1j*Kx*phi_xy+1j*Ky*phi_yy
        gamma_z_k = 1j*Kx*phi_zx+1j*Ky*phi_yz

        varphi_k= 1j*Kx*gamma_y_k - 1j*Ky*gamma_x_k
        varphi = np.fft.irfft2(varphi_k, axes=(0,1))*nx*ny*(0.5/f)**2

        amplitudes_0[i] = abs(InnerProduct(varphi, psi_0))
        amplitudes_a0[i] = abs(InnerProduct(varphi, psi_a0))
        amplitudes_a0_dagger[i] = abs(InnerProduct(varphi, psi_a0_dagger))
        print(amplitudes_0[i],amplitudes_a0[i], amplitudes_a0_dagger[i])


    else:
        print('data is not saved yet')

    file.close()
plt.clf()
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,7))
#plt.plot(Reynolds, relative_amplitude_a0,'o', color='black')
#plt.plot(Reynolds, relative_amplitude_a0,'--', color='black')

#plt.plot(Reynolds, relative_amplitude_a2,'o', color='red')
#plt.plot(Reynolds, relative_amplitude_a2,'--', color='red')
#plt.ylim((0,1.2))
ax.plot(Reynolds, amplitudes_0,'--', color='red',marker='s',label=r'$|\langle \psi_0 | \psi \rangle |$')
ax.plot(Reynolds, amplitudes_a0,'--', color='blue', marker='s',label=r'$|\langle \psi_\alpha | \psi \rangle |$')
ax.plot(Reynolds, amplitudes_a0_dagger,'--', color='green', marker='s',label=r'$|\langle \psi_\alpha | \psi \rangle |$')

#ax.plot(ReynoldsFit, fit, color='black',label='Theoretical prediction '+r'$\mathcal{O}(\epsilon^{1/2})$',linewidth=4)
#ax.plot(ReynoldsFit, fit-epsilon,'--',color='gray',linewidth=3, alpha=.75,label='Theoretical prediction '+r'$\pm\epsilon$')
#ax.plot(ReynoldsFit, fit+epsilon,'--',color='gray',linewidth=3, alpha=.75)
#ax.plot(Reynolds, amplitudes_a0,'--', color='blue')
#ax.plot(Reynolds, amplitudes_0,'o', color='red')
#ax.set_xscale('log')
ax.set_xlabel(r'$Re$')
ax.xaxis.label.set_fontsize(15)
ax.legend()
ax.set_title(r'$N_z=$'+'{0}'.format(int(N/ryz)))
#ax.yaxis.label.set_fontsize(30)
#plt.ylim(0,1.2)


#plt.plot(Reynolds, amplitudes_a2,'o', color='red')
#plt.plot(Reynolds, amplitudes_a2,'--', color='red')
fig.savefig('RevsA_raw_stress_projection_very_high_reynolds_scratch0{0}_ryz{1}.png'.format(scratch, ryz), format='png')
