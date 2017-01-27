import sympy as sp
import numpy as np
import scipy.integrate as integrate
from scipy.fftpack import ifft

alpha = sp.Symbol('alpha')
lamda = sp.Symbol('lamda')
a = sp.Symbol('a')
x = sp.Symbol('x')
phi0 = sp.Symbol('phi0')
s = sp.Symbol('s')

def g(phi, alpha, aa, u, lamda):
    return lamda*phi + u*np.exp(-lamda*aa)

def chebyshev_extremal_01(n):
    return np.array([0.5*(1 + np.cos(k*np.pi/n)) for k in range(n+1)])

def clencurt_01(n):
    C = np.zeros((n+1,2))
    k = 2*(1+np.arange(np.floor(n/2)))
    C[::2,0] = 2/np.hstack((1, 1-k*k))
    C[1,1] = -n
    V = np.vstack((C,np.flipud(C[1:n,:])))
    F = np.real(ifft(V, n=None, axis=0))
    w = np.hstack((F[0,0],2*F[1:n,0],F[n,0]))
    return 0.5*w

n = 100

A = lambda t : sp.integrate(-lamda, t)
c = lambda t : phi0 + sp.integrate(sp.exp(A(s))*x*sp.exp(-lamda*a), (s, 0, t))
phi = lambda t : sp.exp(-A(t))*c(t)
f_sym = sp.Matrix([phi(a)])

f = lambda aa, t, u : np.array(f_sym.subs([(a, aa), (lamda, t), (x, u), (phi0, 1)]))
f_vec = np.vectorize(f)
nodes = chebyshev_extremal_01(n)

G = lambda t, u : f_vec(nodes, t, u).dot(clencurt_01(n))[0][0] #quadratura della f esatta

def G2(l, x, nodes):
    n = len(nodes)-1
    nodes = np.flipud(nodes) #questo per averli in ordine crescente
    vec_f = np.array([1.0])
    ini = 1.0
    for i in range(n):
        Phi, info = integrate.odeint(lambda phi, alpha : g(phi, alpha, nodes[i+1], x, l), ini, np.linspace(nodes[i], nodes[i+1], 100000), full_output = 1)
        vec_f = np.append(vec_f, Phi[-1][0])
        ini = Phi[-1][0]
    print 'approx f =', [(node, func) for node, func in zip(nodes, vec_f)]
    print 'G2 =', np.flipud(vec_f).dot(clencurt_01(n)) 
    return np.flipud(vec_f).dot(clencurt_01(n))


print 'exact f =', [(node, func) for node, func in zip(nodes, f_vec(nodes, 1, 1))] #G2 dovrebbe approssimare G, e vec_f dovrebbe approssimare f
print 'G(1, 1) =', G(1, 1)
y = G2(1, 1, nodes)
