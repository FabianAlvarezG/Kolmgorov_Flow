import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim


class Kolmogorov2D:
    ### a is the aspect ratio Ly/Lx
    ### R in the Reynolds number
    ### N is the characteritic number of points in the y direction, the simulation
    ### will have a total of 1/a * N**2 points in total
    def __init__(self, a, R, N, wi=None, mu=0):
        self.a = a
        self.R = R
        self.nx = int(1/a*N)
        self.ny = N
        self.mu=mu
        if wi is None:
            self.vorticity_intial = 0.1*(np.fft.rfft2(np.random.rand(self.ny, self.nx)/(self.nx*self.ny))-.5)
        else:
            self.vorticity_intial = wi
        self.vorticity = self.vorticity_intial
        self.vorticity[0,0]=0
        self.stream = np.zeros((self.ny, int(self.nx/2)+1))
        self.ux = np.zeros((self.ny, int(self.nx/2)+1))
        self.uy = np.zeros((self.ny, int(self.nx/2)+1))
        self.Lx = 2*np.pi/a
        self.Ly = 2*np.pi
        self.ky = 2*np.pi*np.fft.fftfreq(self.ny, self.Ly/self.ny)
        self.kx = 2*np.pi*np.fft.rfftfreq(self.nx, self.Lx/self.nx)
        self.Kx,self.Ky = np.meshgrid(self.kx, self.ky)
        self.K2 = self.Kx*self.Kx + self.Ky*self.Ky
        self.forcing = np.zeros((self.ny, int(self.nx/2)+1))
        dx = 2*np.pi/N
        self.dt = dx**2*R/50
        x = 2*np.pi*np.arange(self.nx)/N
        y = 2*np.pi*np.arange(self.ny)/N
        X,Y = np.meshgrid(x,y)
        f = -np.cos(Y)
        self.forcing = np.fft.rfft2(f/(self.nx*self.ny))/R
        self.large_scale_damping = np.zeros((self.ny, int(self.nx/2)+1))
        if a<0.5:
            self.large_scale_damping[0,1]=1
            self.large_scale_damping[0,2]=1
        else:
            self.large_scale_damping[0,1]=1
    def set_Re(self, R):
        N = self.ny
        self.R = R
        dx = 2*np.pi/N
        self.dt = dx**2*R/50
        x = 2*np.pi*np.arange(self.nx)/N
        y = 2*np.pi*np.arange(self.ny)/N
        X,Y = np.meshgrid(x,y)
        f = -np.cos(Y)
        self.forcing = np.fft.rfft2(f/(self.nx*self.ny))/R

    def compute_vorticity(self):
        self.vorticity = self.K2*self.stream
    def compute_stream(self, vorticity=None):
        if vorticity is None:
            self.K2[0,0] = -1
            self.stream = self.vorticity/self.K2
            self.K2[0,0] = 0
            self.stream[0,0] = 0
        else:
            self.K2[0,0] = -1
            stream = self.vorticity/self.K2
            self.K2[0,0] = 0
            stream[0,0] = 0
            return stream

    def compute_velocity(self, vorticity=None):
        if vorticity is None:
            self.ux = 1j*self.Ky*self.stream
            self.uy = -1j*self.Kx*self.stream
        else:
            stream = self.compute_stream(vorticity)
            ux = 1j*self.Ky*stream
            uy = -1j*self.Kx*stream
            return ux,uy

    def get_real_velocity(self, vorticity=None):
        if vorticity is None:
            ux = np.fft.irfft2(self.ux)
            uy = np.fft.irfft2(self.uy)
            return ux,uy
        else:
            uxk,uyk = self.compute_velocity(vorticity)
            ux = np.fft.irfft2(uxk)
            uy = np.fft.irfft2(uyk)
            return ux,uy

    def get_real_vorticity(self, vorticity=None):
        if vorticity is None:
            w = np.fft.irfft2(self.vorticity)
            return w
        else:
            w = np.fft.irfft2(vorticity)
            return w

    def compute_advection(self, vorticity=None):
        ux,uy = self.get_real_velocity(vorticity)
        if vorticity is None:
            dwxk = 1j*self.Kx*self.vorticity
            dwyk = 1j*self.Ky*self.vorticity
            dwx = np.fft.irfft2(dwxk)
            dwy = np.fft.irfft2(dwyk)
            return np.fft.rfft2(ux*dwx + dwy*uy)
        else:
            dwxk = 1j*self.Kx*vorticity
            dwyk = 1j*self.Ky*vorticity
            dwx = np.fft.irfft2(dwxk)
            dwy = np.fft.irfft2(dwyk)
            return np.fft.rfft2(ux*dwx + dwy*uy)
    def equation(self, vorticity):
        advection_term = self.compute_advection(vorticity)*(self.nx*self.ny)
        #print(np.max(abs(advection_term)),abs(vorticity).max())
        return -advection_term - 1/self.R*self.K2*vorticity + self.forcing - self.mu*self.large_scale_damping*vorticity
    def time_step(self):
        dt =self.dt
        k1 = self.equation(self.vorticity)
        k2 = self.equation(self.vorticity+0.5*dt*k1)
        k3 = self.equation(self.vorticity+0.5*dt*k2)
        k4 = self.equation(self.vorticity+dt*k3)
        self.vorticity = self.vorticity + dt/6*(k1+2*k2+2*k3+k4)


'''
##### Making movie ####
if __name__ == "__main__":
    a = 1/3
    N = 32
    mu = 142/585/np.sqrt(3)
    Rc = 65/9/np.sqrt(3)
    R = 8
    wi = np.load('wi.npy')
    iter = 1000000

    x = 2*np.pi*np.arange(N/a)/N
    y = 2*np.pi*np.arange(N)/N
    nx = int(N/a)
    ny = N
    X,Y = np.meshgrid(x,y)
    epsilon = 0.001
    perturbation = np.random.rand(ny,nx)-.5
    wl = -np.cos(Y)
    wlk = np.fft.rfft2(wl+epsilon*perturbation)/(nx*ny)


    sim = Kolmogorov2D(a, R, N, wlk,mu)
    #fig, ax = plt.subplots(ncols=1,nrows=1, figsize=(7/a,7))


    #x = 2*np.pi*np.arange(N/a)/N
    #y = 2*np.pi*np.arange(N)/N
    #X,Y = np.meshgrid(x,y)
    #wl = -np.cos(Y)
    #perturbation = 0.001*np.fft.rfft2(np.random.rand(N, int(N/a))/(N**2/a))
    #perturbation[0,0]=0
    #wlk = np.fft.rfft2(wl/(sim.nx*sim.ny))


    for i in range(iter):
        sim.time_step()
        print(np.max(sim.get_real_vorticity()*sim.nx*sim.ny))

    wf = sim.get_real_vorticity()*sim.nx*sim.ny

im = ax.imshow(wf-wl)
fig.savefig('test.png')

codec = 'x265'
fps = 60,
crf = 21,
fname='test'

movie_writer = anim.writers['ffmpeg'](
                fps = fps,
                metadata = {'title': fname},
                codec = 'lib' + codec,
                extra_args = ['-pix_fmt', 'yuv420p',
                              '-crf', '{0}'.format(crf),
                              '-preset', 'veryslow'])

'''
