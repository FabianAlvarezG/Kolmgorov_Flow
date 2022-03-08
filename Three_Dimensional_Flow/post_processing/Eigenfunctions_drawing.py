import numpy as np
from reader import DB2D_reader
import os
import matplotlib.pyplot as plt
'''
plt.rcParams.update({
    'text.usetex' : False,
    'font.family' : 'sans-serif',
    'font.sans-serif' : ['Arial'],
    'mathtext.fallback_to_cm' : True,
    })
    '''
plt.rcParams.update({'font.size': 12})




N = 256
rxy = 3
ryz = 4
scratch = 3
path = '/scratch0{0}.local/falvarez/Kflow/forcing_control/detailed_high_reynolds_extra_precision/ryz{1}'.format(scratch,ryz)

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

kx = np.fft.rfftfreq(nx, 6*np.pi/nx)
ky = np.fft.fftfreq(ny,2*np.pi/nx)

Kx,Ky = np.meshgrid(kx,ky)

def get_vel_from_stream(stream):
    sk = np.fft.rfft2(stream)
    vxk = 1j*Ky*sk
    vyk = -1j*Kx*sk
    vx = np.fft.irfft2(vxk)*nx*ny
    vy = np.fft.irfft2(vyk)*nx*ny
    return vx, vy

amplitude = 2.0
single_vortex_pair = np.real(psi_0*np.sqrt(2) + amplitude*psi_a0)
vx,vy = get_vel_from_stream(single_vortex_pair)
fig ,ax = plt.subplots(ncols=1, nrows=1)
ax.streamplot(x,y,vx,vy, density=0.8)
ax.set_aspect('equal')
fig.savefig('streamplot_test.png')
