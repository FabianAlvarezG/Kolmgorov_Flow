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
ryz = 4
scratch = 3
path = '/scratch0{0}.local/falvarez/Kflow/forcing_control/detailed_high_reynolds/ryz{1}'.format(scratch,ryz)

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

def main():
    Reynolds =np.linspace(250,265,32)
    forcing_amplitudes= Reynolds**2/(2*np.pi)**3*0.25
    amplitudes_0 = np.copy(Reynolds)
    amplitudes_a0 = np.copy(Reynolds)
    amplitudes_a1 = np.copy(Reynolds)
    amplitudes_a2 = np.copy(Reynolds)

    amplitudes_2a0 = np.copy(Reynolds)
    amplitudes_2a1 = np.copy(Reynolds)
    amplitudes_2a2 = np.copy(Reynolds)


    E2D_array = np.copy(Reynolds)
    Emeanz_array = np.copy(Reynolds)
    Efluctuations_array= np.copy(Reynolds)
    Reynolds2D =Reynolds**2/(2*np.pi)**3
    Rc =5/3
    DeltaRe = np.arange(0,1.2,0.02)
    #DeltaRe = DeltaRe[DeltaRe>0]
    fit = np.sqrt(DeltaRe/(Rc*(Rc+DeltaRe)))*np.sqrt(lc(a,mu)/(gamma(a,mu)))*np.sqrt(2+c(R,a,mu)**2)
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
        #data = cc2.compute_Kflow_energetics_single_iteration_from_checkpoint(iteration=iteration)
        data = cc2.compute_Kflow_energetics_single_iteration(iteration=iteration)
        Etotal = data['Etotal']
        E2D = data['E2D']/Etotal
        Emeanz = data['Emeanz']/Etotal
        Efluctuations = data['Efluctuations']/Etotal
        print('Distribution of energy (Stream, uz_avg, fluctuations) {0}, {1}, {2}'.format(E2D,Emeanz,Efluctuations))
        E2D_array[i] = E2D
        Emeanz_array[i] = Emeanz
        Efluctuations_array[i] = Efluctuations
        sk =cc2.read_filtered_DB_iteration(iteration=iteration, dataset='stream_function')
        sr = np.fft.irfft2(sk)*int(N*N/a)
        amplitudes_0[i] = abs(InnerProduct(sr, psi_0))*np.sqrt(2)*0.5/f
        amplitudes_a0[i] = abs(InnerProduct(sr, psi_a0)/f*0.5)
        amplitudes_a1[i] = abs(InnerProduct(sr, psi_a1)/f*0.5)
        amplitudes_a2[i] = abs(InnerProduct(sr, psi_a2)/f*0.5)
        amplitudes_2a0[i] = abs(InnerProduct(sr, psi_2a0)/f*0.5)

    plt.clf()
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,7))

    ax.plot(Reynolds, amplitudes_0,'--', color='red',marker='s',label=r'$|\langle \psi_0 | \psi \rangle |$')
    ax.plot(Reynolds, amplitudes_a0,'--', color='blue', marker='s',label=r'$|\langle \psi_\alpha | \psi \rangle |$')
    #ax.plot(Reynolds, amplitudes_2a0,'--', color='green', marker='s',label=r'$|\langle \psi_{2\alpha} | \psi \rangle |$')
    #ax.plot(Reynolds, amplitudes_3a0,'--', color='purple', marker='s',label=r'$|\langle \psi_{3\alpha} | \psi \rangle |$')

    #ax.plot(ReynoldsFit, fit, color='black',label='Theoretical prediction '+r'$\mathcal{O}(\epsilon^{1/2})$',linewidth=4)
    #ax.plot(ReynoldsFit, fit-epsilon,'--',color='gray',linewidth=3, alpha=.75,label='Theoretical prediction '+r'$\pm\epsilon$')
    #ax.plot(ReynoldsFit, fit+epsilon,'--',color='gray',linewidth=3, alpha=.75)
    #ax.plot(Reynolds, amplitudes_a0,'--', color='blue')
    #ax.plot(Reynolds, amplitudes_0,'o', color='red')
    #ax.set_yscale('log')
    ax.set_xlabel(r'$Re$')
    ax.xaxis.label.set_fontsize(15)
    ax.legend()
    ax.set_title(r'$N_z=$'+'{0}'.format(int(N/ryz)))
    ax.set_ylim(0,0.6)
    print('hola')
    #ax.yaxis.label.set_fontsize(30)
    #plt.ylim(0,1.2)


    #plt.plot(Reynolds, amplitudes_a2,'o', color='red')
    #plt.plot(Reynolds, amplitudes_a2,'--', color='red')
    fig.savefig('RevsA_raw_detailed_high_reynolds_scratch0{0}_ryz{1}.png'.format(scratch, ryz), format='png')
    plt.clf()
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,7))
    ax.plot(Reynolds, E2D_array,'--', color='red',marker='s',label=r'$E_{2D}$')
    ax.plot(Reynolds, Emeanz_array,'--', color='blue', marker='s',label=r'$E_{z}$')
    ax.plot(Reynolds, Efluctuations_array,'--', color='green', marker='s',label=r'$E_{fluct}$')
    ax.set_xlabel(r'$Re$')
    ax.set_yscale('log')
    ax.xaxis.label.set_fontsize(15)
    ax.legend()

    ax.set_title(r'$N_z=$'+'{0}'.format(int(N/ryz)))
    fig.savefig('RevsA_raw_energy_detailed_high_reynolds_scratch0{0}_ryz{1}.png'.format(scratch, ryz), format='png')
    return 0
if __name__ == '__main__':
    #print('hola')
    main()
