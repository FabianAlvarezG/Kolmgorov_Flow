import numpy as np
import scipy.linalg as spl
from numpy import linalg as LA
import matplotlib.pyplot as plt
import numpy.ma as ma
import cmocean
plt.rcParams.update({
    'text.usetex' : False,
    'font.family' : 'sans-serif',
    'font.sans-serif' : ['Arial'],
    'mathtext.fallback_to_cm' : True,
    })
plt.rcParams.update({'font.size': 20})

rxy = 3
nx = rxy *256
ny = 256

dx = 2*np.pi/256

x = np.arange(nx)*dx
y = np.arange(ny)*dx

X,Y = np.meshgrid(x,y)

class KolmogorovFlow2D:
	def __init__(self, N, alpha, R):
		self.N = N
		self.alpha = alpha
		self.R = R
		self.matrix = np.zeros((2*N+1,2*N+1))
		self.eigvalues=None
		self.eigvectors=None
		for i in np.arange(0,2*N+1,1):
			n = -N+i
			c0 = -1/R*(alpha**2 +n**2)
			c_1 = -alpha/2 *(alpha**2 +(n-1)**2-1)/(alpha**2 +n**2)
			c1 = alpha/2 *(alpha**2 +(n+1)**2-1)/(alpha**2 +n**2)
			self.matrix[i][i] = c0
			if i==0:
				self.matrix[i][i+1] = c1
			elif i ==2*N:
				self.matrix[i][i-1] = c_1
			else:
				self.matrix[i][i+1] = c1
				self.matrix[i][i-1] = c_1

	def compute_eigenvalues_eigenvectors(self):
		w,v = LA.eig(self.matrix)
		self.eigvalues = w
		self.eigvectors = v


	def stream_function(self, eig=0, phase=0):
		self.compute_eigenvalues_eigenvectors()
		index= np.argsort(-self.eigvalues)
		eig = index[eig]
		print(eig, test.eigvalues)
		#eig = np.argmax(self.eigvalues)
		stream_eig = 0
		for i in range(2*self.N+1):
			n = -self.N+i
			stream_eig = stream_eig + self.eigvectors[i,eig]*np.exp(1j*self.alpha*(X-phase) + 1j*n*Y)
		return stream_eig
	def vorticity(self, eig=0, phase=0):
		self.compute_eigenvalues_eigenvectors()
		index= np.argsort(-self.eigvalues)
		eig = index[eig]
		print(eig, test.eigvalues)
		#eig = np.argmax(self.eigvalues)
		vorticity_eig = 0
		for i in range(2*self.N+1):
			n = -self.N+i
			vorticity_eig = vorticity_eig -(self.alpha**2+n**2)*self.eigvectors[i,eig]*np.exp(1j*self.alpha*(X-phase) + 1j*n*Y)
		return vorticity_eig

	def do_plots(self):
		fig, axes = plt.subplots(nrows = 1, ncols=2, figsize=(14*rxy, 7))
		sf = self.stream_function()
		axes[0].contour(sf.real)
		axes[1].contour(sf.imag)
		plt.savefig("sf_eigv_0_N_{0}_alpha_{1}_R_{2}.png".format(self.N, self.alpha, self.R))
	def do_final_plot(self, A, eig=0):
		plt.clf()
		fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=(7*rxy, 7))
		sf = 2*A*self.stream_function(eig=eig,phase=np.pi).real - np.cos(Y)
		wf = 2*A*self.vorticity(eig=eig).real+np.cos(Y)
		ax.contour(sf, cmap=cmocean.cm.balance)
		ax.set_xlabel(r'$x$')
		ax.set_ylabel(r'$y$')
		ax.xaxis.label.set_fontsize(30)
		ax.yaxis.label.set_fontsize(30)
		#ax.set_title(r'$\psi$')
		plt.savefig("sf_final_0_N_{0}_alpha_{1:.2}_R_{2:.2}.png".format(self.N, self.alpha, self.R), format='png')
	def do_final_plot_vorticity(self, A, eig=0):
		plt.clf()
		fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=(7*rxy, 7))
		#sf = 2*A*self.stream_function(eig=eig).real - np.cos(Y)
		wf = 2*A*self.vorticity(eig=eig).real-np.cos(Y)
		v=0.4
		ax.imshow(wf, cmap=cmocean.cm.balance, vmax=1+v, vmin=1-v)
		plt.savefig("wf_final_0_N_{0}_alpha_{1:.2}_R_{2:.2}.png".format(self.N, self.alpha, self.R), format='png')


def pred(alpha):
	return np.sqrt(2)*(1+alpha**2)/np.sqrt(1-alpha**2)

def pred_vector(alpha):
	return np.array([-1, np.sqrt(2)*(1+alpha**2)*np.sqrt(1-alpha**2)/(alpha*(1-alpha**2)),1])

alpha = 1/rxy
eig=0
m = 2
test = KolmogorovFlow2D(20,m*alpha, pred(m*alpha))
test.do_final_plot(2, eig)
#test.do_final_plot_vorticity(2, eig)
#sf1 = test.stream_function()

#test2 = KolmogorovFlow2D(1,-alpha, pred(alpha))
#sf2 = test2.stream_function()

#sf = (sf1-sf2).real
#test.do_plots(eig=0)
#sf1,sf2 = test.stream_function(0)
#vf1,vf2 = test.vorticity_eig(0)
#vort=test.vorticity_eig(0)
#plt.clf()
#plt.contour(2*(sf1-sf2).real-test.f/test.nu*np.cos(Y))
#plt.savefig('cosa')
