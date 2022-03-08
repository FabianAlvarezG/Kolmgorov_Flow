from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.usetex' : False,
    'font.family' : 'sans-serif',
    'font.sans-serif' : ['Arial'],
    'mathtext.fallback_to_cm' : True,
    })
plt.rcParams.update({'font.size': 15})

def equation(x, alpha, N, mu=0):
	#alpha = args[0]
	#N = args[1]
	matrix = np.zeros((2*N+1,2*N+1))
	for i in range(2*N+1):
		n = -N+i
		c0 = -1/x*(alpha**2 +n**2)**2 #+ (alpha**2 +n**2)*x
		c_1 = -alpha/2 *(alpha**2 +(n-1)**2-1)
		c1 = alpha/2 *(alpha**2 +(n+1)**2-1)
		matrix[i][i] = c0
		if i==0:
			matrix[i][i+1] = c1
		elif i==N:
			matrix[i][i+1] = c1
			matrix[i][i-1] = c_1
			matrix[i][i] = c0 - mu*alpha**2
		elif i ==2*N:
			matrix[i][i-1] = c_1
		else:
			matrix[i][i+1] = c1
			matrix[i][i-1] = c_1

	return np.linalg.det(matrix)

def pred(alpha):
	return np.sqrt(2)*(1+alpha**2)/np.sqrt(1-alpha**2)

def pred_2(a, u):
	return (4*u + (2*u)/a**2 + 2*a**2*u + np.sqrt(4*(1 - a**2)*(2 + 4*a**2 + 2*a**4) + (4 + 2/a**2 + 2*a**2)**2*u**2))/(2*(1-a**2))

def matrix(x, alpha, N,mu=0):
	matrix = np.zeros((2*N+1,2*N+1))
	for i in range(2*N+1):
		n = -N+i
		c0 = -1/x*(alpha**2 +n**2)**2 +mu*alpha**2 #+ (alpha**2 +n**2)*x
		c_1 = -alpha/2 *(alpha**2 +(n-1)**2-1)
		c1 = alpha/2 *(alpha**2 +(n+1)**2-1)
		matrix[i][i] = c0
		if i==0:
			matrix[i][i+1] = c1
		elif i ==2*N:
			matrix[i][i-1] = c_1
		else:
			matrix[i][i+1] = c1
			matrix[i][i-1] = c_1

	return matrix

def det(R,alpha):

   return (alpha**5*(-1+alpha**2)*(1+alpha**2)**2)/(4*R) + 1/R * (1+alpha**2)**2*(-alpha**4/4 + alpha**4/R**2 + alpha**6/4 + 2*alpha**6/R**2 + alpha**8/R**2)


m = 100
N= 10
mu=0.0
alphas = np.linspace(0.01, .99, m)
alphas2 =  np.linspace(0.01, .49, m)
alphas3 =  np.linspace(0.01, .5, m)
Rc = np.zeros(m)
Rc2 = np.zeros(m)
for i,a in enumerate(alphas):
	#n = m-1-i
	#if i ==0:
	#	Rc[n] = fsolve(equation, pred_2(a, mu), (a, N, mu))[0]
	#else:
	#	Rc[n] = fsolve(equation, Rc[n+1], (a, N, mu))[0]
	Rc[i] = fsolve(equation, pred_2(a, mu), (a, N, mu), maxfev=500000)[0]

#for i,a in enumerate(alphas2):
	#Rc2[i] = fsolve(equation, pred_2(2*a, mu), (2*a, N, mu))[0]



plt.clf()
fig, ax = plt.subplots(ncols=1,nrows=1, figsize=(7,7))
#ax.plot(alphas, pred_2(alphas, 0.0),'-', color='red', linewidth=4)
ax.plot(alphas, Rc,'o', color='green', label='Numerical result using {0} modes'.format(N))
ax.plot(alphas, pred_2(alphas, mu),'--', color='orange', linewidth=4, label=r'$\sqrt{2}\frac{1+\alpha^2}{\sqrt{1-\alpha^2}}$')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$Re$')
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
ax.legend()
#ax.plot(alphas2, Rc2,'*', color='green')
#ax.plot(alphas2, pred_2(2*alphas2,0),'-', color='blue', linewidth=4)
#ax.plot(alphas2, pred_2(3*alphas3,0),'-', color='green', linewidth=4)
ax.set_ylim((0,8))
ax.set_xlim((0,1))
fig.savefig('Critical_Reynolds_Nmodes_{0}_mu_{1}.png'.format(N, mu), format='png')
#mat = matrix(2, 0,1)
