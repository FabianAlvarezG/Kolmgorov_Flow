from Kolmogorov2D import Kolmogorov2D
import matplotlib.pyplot as plt
import sys
import numpy as np
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
ratio_xy = 3

a = 1/ratio_xy
mu = 0.0
### Function to project ###
dx = 2*np.pi/N
nx = int(N/a)
ny = N

y = np.arange(0,ny)*dx
x = np.arange(0,nx)*dx
X,Y = np.meshgrid(x,y)

mu = 0.0

def c(a,u):
    return (2*(a + 2*a**3 + a**5))/(u + 2*a**2*u + a**4*u + a**2*np.sqrt(-(((1 + a**2)**2*(-2*a**4 + 2*a**6 - u**2 - 2*a**2*u**2 - a**4*u**2))/a**4)))

def lc(a,u):
    return (2*(1 + a**2)**2*(2*a**6 - u**2 - a**4*(2 + u**2) + a**2*u*(-2*u + np.sqrt(-2*(-1 + a**2)*(1 + a**2)**2 + ((1 + a**2)**4*u**2)/a**4))))/(a**4*(-1 + a**2))

def gamma(a,u):
    return (-16*a**4*(1 + a**2)**4*(-1 - 17*a**2 - 16*a**4 + 32*a**6)*((1/a + a)**2*u + np.sqrt(2*(1 - a**2)*(1 + a**2)**2 + (1/a + a)**4*u**2)))/((1 - a**2)*(1 + 4*a**2)**2*((1 + a**2)**2*u + a**2*np.sqrt(-2*(-1 + a**2)*(1 + a**2)**2 + ((1 + a**2)**4*u**2)/a**4))**2)

def InnerProduct(vector, psi):
    integral = dx*dx*np.sum(np.conjugate(psi)*vector)/(4*np.pi**2/a)
    return integral

psi_0 = (np.exp(1j*Y)+np.exp(-1j*Y))/np.sqrt(2)
psi_a0  = (np.exp(1j*Y) - np.exp(-1j*Y) + c(a,mu))*np.exp(1j*X*a)/np.sqrt(2+c(a,mu)**2)
psi_a1 = (np.exp(1j*Y) + np.exp(-1j*Y))*np.exp(1j*X*a)/np.sqrt(2)
psi_a2 = (c(a,mu)*(np.exp(1j*Y) - np.exp(-1j*Y))-2)*np.exp(1j*X*a)/np.sqrt(4+2*c(a,mu)**2)

psi_2a0  = (np.exp(1j*Y) - np.exp(-1j*Y) + c(2*a,mu))*np.exp(1j*X*a*2)/np.sqrt(2+c(2*a,mu)**2)
psi_2a1 = (np.exp(1j*Y) + np.exp(-1j*Y))*np.exp(1j*X*a*2)/np.sqrt(2)
psi_2a2 = (c(2*a,mu)*(np.exp(-1j*Y) - np.exp(-1j*Y))-2)*np.exp(1j*X*a*2)/np.sqrt(4+2*c(2*a,mu)**2)

Reynolds =np.linspace(10,26,32)
forcing_amplitudes= Reynolds**2/(2*np.pi)**3*0.25
#Reynolds =np.linspace(10,26,32)**2/(2*np.pi)**3
amplitudes_0 = np.copy(Reynolds)
amplitudes_a0 = np.copy(Reynolds)
Reynolds2D =Reynolds**2/(2*np.pi)**3
Rc =5/3
DeltaRe = np.arange(0,1.2,0.02)
#DeltaRe = DeltaRe[DeltaRe>0]
fit = np.sqrt(DeltaRe/(Rc*(Rc+DeltaRe)))*np.sqrt(lc(a,mu)/(gamma(a,mu)))*np.sqrt(2+c(a,mu)**2)
ReynoldsFit = np.sqrt((Rc+DeltaRe)*(2*np.pi)**3)
epsilon = DeltaRe/(Rc*(Rc+DeltaRe))
scratch = 4
for i,famplitude in enumerate(forcing_amplitudes):
#famplitude = forcing_amplitudes[index]
    ## Saving directories
    R = Reynolds[i]
    simname = 'N{0:0>4d}_rxy{1}_famplitude_{2:.3f}'.format(N, ratio_xy,famplitude)
    base_dir = '/scratch0{0}.local/falvarez/Kflow/2D/'.format(scratch)
    work_dir = os.path.join(base_dir, 'N{0:0>4d}_rxy{1}_famplitude_{2:.3f}'.format(N, ratio_xy,famplitude))
    wk = np.load(os.path.join(work_dir,'vorticity.npy'),allow_pickle=True)
    sim = Kolmogorov2D(a, R, N, wk,mu)
    sk = sim.compute_stream(wk)
    wr = np.fft.irfft2(wk)*nx*ny
    sr = np.fft.irfft2(sk)*nx*ny
    amplitudes_0[i] = abs(InnerProduct(sr, psi_0))*np.sqrt(2)
    amplitudes_a0[i] = abs(InnerProduct(sr, psi_a0))
    print(i, amplitudes_0[i], amplitudes_a0[i])
plt.clf()
data = np.zeros((32, 3))
data[:,0] = Reynolds
data[:,1] = amplitudes_0
data[:,2] = amplitudes_a0
np.savetxt("low_reynolds_amplitudes.csv", data, delimiter=",")

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,7))
#plt.plot(Reynolds, relative_amplitude_a0,'o', color='black')
#plt.plot(Reynolds, relative_amplitude_a0,'--', color='black')

#plt.plot(Reynolds, relative_amplitude_a2,'o', color='red')
#plt.plot(Reynolds, relative_amplitude_a2,'--', color='red')
#plt.ylim((0,1.2))
ax.plot(Reynolds, amplitudes_0,'--', color='red',marker='s',label=r'$|\langle \psi_0 | \psi \rangle |$')
ax.plot(Reynolds, amplitudes_a0,'--', color='blue', marker='s',label=r'$|\langle \psi_\alpha | \psi \rangle |$')
ax.plot(ReynoldsFit, fit, color='black',label='Theoretical prediction '+r'$\mathcal{O}(\epsilon^{1/2})$',linewidth=4)
#ax.plot(ReynoldsFit, fit-epsilon,'--',color='gray',linewidth=3, alpha=.75,label='Theoretical prediction '+r'$\pm\epsilon$')
#ax.plot(ReynoldsFit, fit+epsilon,'--',color='gray',linewidth=3, alpha=.75)
#ax.plot(Reynolds, amplitudes_a0,'--', color='blue')
#ax.plot(Reynolds, amplitudes_0,'o', color='red')
ax.set_xlabel(r'$Re$')
ax.xaxis.label.set_fontsize(15)
ax.legend()
ax.set_title(r'$N_z=1$')
#plt.ylim(0,1.2)
plt.savefig('RevsA_raw_scratch0{0}_2D.png'.format(scratch), format='png')
