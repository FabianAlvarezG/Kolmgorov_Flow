import numpy as np
import matplotlib.pyplot as plt


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

def Rc0(a):
    return np.sqrt(2)*(1+a**2)/np.sqrt(1-a**2)

def Rc(a,u):
    return (u + 2*a**2*u + a**4*u + np.sqrt((1 + a**2)**2*(-2*a**6 + u**2 + 2*a**2*u**2 + a**4*(2 + u**2))))/(a**2*(1 - a**2))

def c(a,u):
    return (2*(a + 2*a**3 + a**5))/(u + 2*a**2*u + a**4*u + a**2*np.sqrt(-(((1 + a**2)**2*(-2*a**4 + 2*a**6 - u**2 - 2*a**2*u**2 - a**4*u**2))/a**4)))

def lc(a,u):
    return (2*(1 + a**2)**2*(2*a**6 - u**2 - a**4*(2 + u**2) + a**2*u*(-2*u + np.sqrt(-2*(-1 + a**2)*(1 + a**2)**2 + ((1 + a**2)**4*u**2)/a**4))))/(a**4*(-1 + a**2))

def gamma(a,u):
    return (-16*a**4*(1 + a**2)**4*(-1 - 17*a**2 - 16*a**4 + 32*a**6)*((1/a + a)**2*u + np.sqrt(2*(1 - a**2)*(1 + a**2)**2 + (1/a + a)**4*u**2)))/((1 - a**2)*(1 + 4*a**2)**2*((1 + a**2)**2*u + a**2*np.sqrt(-2*(-1 + a**2)*(1 + a**2)**2 + ((1 + a**2)**4*u**2)/a**4))**2)

def InnerProduct(vector, psi):
    integral = dx*dx*np.sum(np.conjugate(psi)*vector)/(4*np.pi**2/a)
    return integral

def ampCoeff(a):
    return np.sqrt(2+c(a,0)**2)*np.sqrt((2 + 4*a**2 + 2*a**4 + Rc0(a)**2 - a**2*Rc0(a)**2)/((1 + 4*a**2)**2*(1 + 17*a**2 + 16*a**4 - 32*a**6)*Rc0(a)**3))

def Re2D(Re):
    return Re**2/(2*np.pi)**3

def epsilon(Re,a):
    Re = Re2D(Re)
    return (Re - Rc0(a))/(Rc0(a)*(Re))


data = np.loadtxt('merged_data.csv', delimiter=',')
Reynolds = data[:,0]
ProjectionLaminar = data[:,1]
ProjectionEigen = data[:,2]
eps = epsilon(Reynolds, a)
x = eps[eps>0]
x = np.sqrt(x)

y = ProjectionEigen[eps>0]/ampCoeff(a)/x
X = x**2
p0 =np.polyfit(X,y,deg=4)
p= np.poly1d(p0)
print(p0)
plt.plot(Reynolds[eps>0],y*x,'*')
plt.plot(Reynolds[eps>0],x*p(X))
#plt.plot(x,x)
plt.xlim(10,80)
plt.savefig('Fit_2d_kolmogorov.png')
