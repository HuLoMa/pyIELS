from math import factorial, exp, sqrt
from qutip import basis
from qutip import displace, squeeze
import numpy as np
from decimal import *

## ============ Calculate the electron state amplitudes $F_l^n$ according to https://doi.org/10.1364/OPTICA.404598 (see equation 3)
## ------ Inputs:
# n: Natural number indexing the photon Fock state
# l: Relative integer indexing the sidebands
# g0: Electron-photon coupling constant
# option='decimal': Version using the decimal module of Python in order to store large numbers in memory - to the cost of longer computation time

def Fnl(n,l,g0,option=None):
    if option=='decimal':
        s=Decimal(0)
        for k in range(max(-l,0),n+1):
            num=Decimal((-abs(g0)**2))**Decimal(k)
            deno=Decimal(factorial(k))*Decimal(factorial(l+k))*Decimal(factorial(n-k))
            s=s+num/deno
        p=(Decimal(factorial(n+l))*Decimal(factorial(n))).sqrt()*(Decimal(-(abs(g0)**2)/2).exp())*Decimal((-g0)**l)*s
        return float(p)
    else:
        s=0
        for k in range(max(-l,0),n+1):
            num=(-abs(g0)**2)**k
            deno=factorial(k)*factorial(l+k)*factorial(n-k)
            s=s+num/deno
        p=sqrt(factorial(n+l)*factorial(n))*exp(-(abs(g0)**2)/2)*((-g0)**l)*s
        return p

## ============ Calculate different analytical photonic amplitudes $\alpha$ according to https://doi.org/10.1364/OPTICA.404598 (see equation 3)
### Fock state
## ------ Inputs:
# n: Natural number indexing the photon Fock state
# nbar: Mean number of photons in the field
def alpha_Fock_analytical(n,nbar):
    if n==nbar:
        alpha=1
    else:
        alpha=0
    return sqrt(alpha)

### Coherent state
## ------ Inputs:
# n: Natural number indexing the photon Fock state
# nbar: Mean number of photons in the field
def alpha_coherent_analytical(n,nbar):
    alpha=exp(-nbar)*(nbar**n)/(factorial(n))
    return sqrt(abs(alpha))

## ============ Calculate different photonic amplitudes $\alpha$ using Qutip
### Fock state
## ------ Inputs:
# ind_table: Table of natural number indexing the photon Fock states
# nbar: Mean number of photons in the field
# N_space: Dimension of the Hilbert space
def alpha_Fock_Qutip(ind_table,nbar,N_space):
    alpha_table=ind_table*0.0*1j
    ket_ref=basis(N_space,nbar)
    for i in range(np.size(ind_table)):
        ket=basis(N_space,i)
        alpha_table[i]=ket.overlap(ket_ref)
    return alpha_table

### Coherent state
## ------ Inputs:
# ind_table: Table of natural number indexing the photon Fock states
# alpha: Coherent state complex parameter
# N_space: Dimension of the Hilbert space
def alpha_coherent_Qutip(ind_table,alpha,N_space):
    alpha_table=ind_table*0.0*1j
    d = displace(N_space, alpha)
    ket_ref=d*basis(N_space,0)
    for i in range(np.size(ind_table)):
        ket=basis(N_space,i)
        alpha_table[i]=ket.overlap(ket_ref)
    return alpha_table

### Squeezed coherent state
## ------ Inputs:
# ind_table: Table of natural number indexing the photon Fock states
# alpha: Coherent state complex parameter
# N_space: Dimension of the Hilbert space
# zeta: Complex squeezing parameter
def alpha_coherent_squeezed(ind_table,alpha,zeta,N_space):
    alpha_table=ind_table*0.0*1j
    d = displace(N_space, alpha)
    s = squeeze(N_space, zeta)
    ket_ref=s*(d*basis(N_space,0))
    for i in range(np.size(ind_table)):
        ket=basis(N_space,i)
        alpha_table[i]=ket.overlap(ket_ref)
    return alpha_table