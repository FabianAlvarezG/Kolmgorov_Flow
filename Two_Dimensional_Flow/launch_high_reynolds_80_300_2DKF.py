from Kolmogorov2D import Kolmogorov2D
import sys
import numpy as np
import os

index = int(sys.argv[1])-1

Reynolds =np.linspace(80, 300, 32)
forcing_amplitudes= Reynolds**2/(2*np.pi)**3*0.25
famplitude = forcing_amplitudes[index]
N = 128
rxy = 3

a = 1/rxy
mu = 0.0

## Saving directories
scratch = 3
simname = 'N{0:0>4d}_rxy{1}_famplitude_{2}'.format(N, rxy,index+1)
base_dir = '/scratch0{0}.local/falvarez/Kflow/2D/high_reynolds'.format(scratch)
work_dir = os.path.join(base_dir, 'N{0:0>4d}_rxy{1}_famplitude_{2}'.format(N, rxy,index+1))

if not os.path.exists(work_dir):
        os.makedirs(work_dir)

### Function to project ###
dx = 2*np.pi/N
nx = int(N/a)
ny = N

y = np.arange(0,ny)*dx
x = np.arange(0,nx)*dx
X,Y = np.meshgrid(x,y)


### Initial condition ###
epsilon = 0.001
perturbation = np.random.rand(ny,nx)-.5
#wl = -np.cos(Y)
#wlk = np.fft.rfft2(wl+epsilon*perturbation)/(nx*ny)
#wk =np.load(os.path.join(work_dir,'vorticity.npy'))
iterations = 409600*20
### Create object
R = Reynolds[index]**2/(2*np.pi)**3
sim = Kolmogorov2D(a, R, N, mu=mu)
sim.dt = sim.dt/20
## Simulate
for i in range(iterations):
    sim.time_step()
## save
wk = sim.vorticity
sk = sim.compute_stream()

np.save(os.path.join(work_dir,'vorticity.npy'), wk)
np.save(os.path.join(work_dir,'stream_function.npy'), sk)
