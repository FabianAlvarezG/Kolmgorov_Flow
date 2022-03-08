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


N = 128
rxy = 3
ryz = 2
scratch = 1
path = '/scratch0{0}.local/falvarez/Kflow/forcing_control/high_reynolds/ryz{1}'.format(scratch,ryz)

a = 1/rxy
### Function to project ###
dx = 2*np.pi/N
y = np.arange(0,N)*dx
x = np.arange(0,rxy*N)*dx
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
def psi(a,mu, number=0):
    if number==0:
        psi  = (np.exp(1j*Y) - np.exp(-1j*Y) + c(a,mu))*np.exp(1j*X*a)/np.sqrt(2+c(a,mu)**2)
    elif number==1:
        psi = (np.exp(1j*Y) + np.exp(-1j*Y))*np.exp(1j*X*a)/np.sqrt(2)
    elif number==2:
        psi= (c(a,mu)*(np.exp(1j*Y) - np.exp(-1j*Y))-2)*np.exp(1j*X*a)/np.sqrt(4+2*c(a,mu)**2)
    return psi

psi_a0 = psi(a,mu,0)
psi_a1 = psi(a,mu,1)
psi_a2 = psi(a,mu,2)

psi_2a0 = psi(2*a,mu,0)
psi_2a1 = psi(2*a,mu,1)
psi_2a2 = psi(2*a,mu,2)
def main():
    Reynolds =np.logspace(5,8,num=32,base=2.0)
    forcing_amplitudes= Reynolds**2/(2*np.pi)**3*0.25
    amplitudes_0 = np.copy(Reynolds)
    amplitudes_a0 = np.copy(Reynolds)
    amplitudes_a1 = np.copy(Reynolds)
    amplitudes_a2 = np.copy(Reynolds)
    relative_amplitude_a0 = np.copy(Reynolds)
    relative_amplitude_a1 = np.copy(Reynolds)
    relative_amplitude_a2 = np.copy(Reynolds)
    amplitudes_2a0 = np.copy(Reynolds)
    amplitudes_2a1 = np.copy(Reynolds)
    amplitudes_2a2 = np.copy(Reynolds)
    Reynolds2D =Reynolds**2/(2*np.pi)**3
    Rc =5/3
    DeltaRe = np.arange(0,1.2,0.02)
    #DeltaRe = DeltaRe[DeltaRe>0]
    fit = np.sqrt(DeltaRe/(Rc*(Rc+DeltaRe)))*np.sqrt(lc(a,mu)/(gamma(a,mu)))*np.sqrt(2+c(a,mu)**2)
    ReynoldsFit = np.sqrt((Rc+DeltaRe)*(2*np.pi)**3)
    epsilon = DeltaRe/(Rc*(Rc+DeltaRe))
    for i,f in enumerate(forcing_amplitudes):
        famplitude = f
        #path = '/localdisk/bt307732/simulations/kolmogorov_flow/ryz{0}'.format(ryz)
        work_dir = os.path.join(path,'N{0:0>4d}_rxy{1}_ryz{2}_famplitude_{3:.3f}'.format(N,rxy,ryz,famplitude))
        simname = 'N{0:0>4d}_rxy{1}_ryz{2}'.format(N,rxy,ryz)
        #try:
        cc2 = DB2D_reader(simname=simname, work_dir=work_dir, read_only_cache=False)
        #except:
            #cc2 = DB2D_reader(simname=simname, work_dir=work_dir, read_only_cache=True)
        iteration =cc2.get_data_file()['iteration'][()]
        sk =cc2.read_filtered_DB_iteration(iteration=iteration, dataset='stream_function')
        sr = np.fft.irfft2(sk)*int(N*N/a)
        amplitudes_0[i] = abs(InnerProduct(sr, psi_0))*np.sqrt(2)*0.5/f
        amplitudes_a0[i] = abs(InnerProduct(sr, psi_a0)/f*0.5)
        amplitudes_a1[i] = abs(InnerProduct(sr, psi_a1)/f*0.5)
        amplitudes_a2[i] = abs(InnerProduct(sr, psi_a2)/f*0.5)
        amplitudes_2a0[i] = abs(InnerProduct(sr, psi_2a0)/f*0.5)
        relative_amplitude_a0[i] = amplitudes_a0[i]/np.sqrt(amplitudes_a0[i]**2+amplitudes_a1[i]**2+amplitudes_a2[i]**2)
        relative_amplitude_a1[i] = amplitudes_a1[i]/np.sqrt(amplitudes_a0[i]**2+amplitudes_a1[i]**2+amplitudes_a2[i]**2)
        relative_amplitude_a2[i] = amplitudes_a2[i]/np.sqrt(amplitudes_a0[i]**2+amplitudes_a1[i]**2+amplitudes_a2[i]**2)
        #print(amplitudes_a0[i])
    plt.clf()

#psi_2a0 = psi(3*a,mu,0)
#psi_2a1 = psi(3*a,mu,1)
#psi_2a2 = psi(3*a,mu,2)
