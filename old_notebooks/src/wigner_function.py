import numpy as np
from math import sqrt, pi
from scipy.special import jv

from src.state_amplitude import alpha_Fock_analytical, alpha_coherent_analytical, alpha_Fock_Qutip, alpha_coherent_Qutip,Fnl
from src.generic import progressbar
from src.units_contants import eV, v
from src.simulation_parameters import number_of_periods, nmax, Nz, maxit, N_space

## ============ Partial Wigner function calculated for a Fock state nFock with amplitude alpha
## ------ Inputs:
# ene_photon: energy of the photon in eV
# g0: Electron-photon coupling constant
# nFock: Natural number indexing the Fock states
# alpha_table: table containing the photonic amplitude, the index of the table spans the Fock basis 
# option='decimal': Version using the decimal module of Python in order to store large numbers in memory - to the cost of longer computation time

def partialwigner(ene_photon,g0,nFock,alpha_table,option=None):
    
    # Momentum of the virtual photon  
    qz = ene_photon*eV/v 

    # Table of indices of the sidebands
    sidebands_axis=np.linspace(-nmax, nmax, 2*nmax+1, dtype=int)

    # Sampling of the z-axis for the number of periods we have chosen
    z = number_of_periods*pi*np.linspace(-1,1,Nz)/qz

    # Initialize the wigner functions
    wigner_exact=np.zeros((2*nmax+1,Nz))*1j # Exact Wigner function

    # Compute the amplitude of each sidebands for a Fock state of average number nbar
    amplitude=np.zeros(2*nmax+1)*1j
    for n_l in range(2*nmax+1):
        lsideband=sidebands_axis[n_l]
        if max(-lsideband,0)>nFock:
            amplitude[n_l]=0
        else:    
            amplitude[n_l]=alpha_table[nFock+lsideband]*Fnl(nFock,lsideband,g0,option)

    # Compute the exact Wigner function
    for j in range(Nz):
        Psi = amplitude*np.exp(1j*sidebands_axis*qz*z[j])
        wigner_exact[:,j] = np.convolve(Psi,np.conj(Psi),'same')

    return wigner_exact, sidebands_axis, z

## ============ Wigner function calculated for a quantized EM field with arbitrary photon amplitudes
## ------ Inputs:
# ene_photon: energy of the photon in eV
# g0: Electron-light coupling constant
# alpha_table: table containing the photonic amplitude, the index of the table spans the Fock basis 
# maxit: Cutoff index for the Fock state summation
# option='decimal': Version using the decimal module of Python in order to store large numbers in memory - to the cost of longer computation time

def wignerQ(ene_photon,g0,alpha_table,maxit,option=None):    
    wigner_init, sidebands_axis,z=partialwigner(ene_photon,g0,0,alpha_table,option)
    wigner_full_array=np.zeros((np.size(wigner_init[:,0]),np.size(wigner_init[0,:]),maxit))*1j

    for k in progressbar(range(maxit),"loop over Fock states",40):
        wigner_loop, sidebands_axis,z=partialwigner(ene_photon,g0,k,alpha_table,option)
        wigner_full_array[:,:,k]=wigner_loop

    return np.real(np.sum(wigner_full_array,axis=2)), sidebands_axis, z  

## ============ Wigner function calculated for a Fock state 
## ------ Inputs:
# ene_photon: energy of the photon in eV
# g0: Electron-light coupling constant
# nbar: Number of photons in the cavity
# methods: precise the methode to use 
#       - "analytical": Using analytically calculated alpha coefficients
#       - "numerical": using QuTip to calculate alpha coefficients
# option='decimal': Version using the decimal module of Python in order to store large numbers in memory - to the cost of longer computation time

def wigner_Fock(ene_photon,g0,nbar,method,option=None):
    # Initialize the alpha_table
    ind_table=np.linspace(0,maxit+nmax+1,maxit+nmax+2)
    alpha_table=ind_table*0.0*1j
    if method=="analytical":
        for i in range(np.size(ind_table)):
            alpha_table[i]=alpha_Fock_analytical(i,nbar)
    elif method=="numerical":
        alpha_table=alpha_Fock_Qutip(ind_table,nbar,N_space)
    else:
        print('Invalid method - switch to analytical by default')
        for i in range(np.size(ind_table)):
            alpha_table[i]=alpha_Fock_analytical(i,nbar)

    return wignerQ(ene_photon,g0,alpha_table,maxit,option)

## ============ Wigner function calculated for a coherent state 
## ------ Inputs:
# ene_photon: energy of the photon in eV
# g0: Electron-light coupling constant
# alpha: Coherent state complex parameter
# methods: precise the methode to use 
#       - "analytical": Using analytically calculated alpha coefficients
#       - "numerical": using QuTip to calculate alpha coefficients``
# option='decimal': Version using the decimal module of Python in order to store large numbers in memory - to the cost of longer computation time

def wigner_coherent(ene_photon,g0,alpha,method,option=None):
    nbar=sqrt(abs(alpha))
    # Initialize the alpha_table
    ind_table=np.linspace(0,maxit+nmax+1,maxit+nmax+2)
    alpha_table=ind_table*0.0*1j
    if method=="analytical":
        for i in range(np.size(ind_table)):
            alpha_table[i]=alpha_coherent_analytical(i,nbar)
    elif method=="numerical":
        alpha_table=alpha_coherent_Qutip(ind_table,alpha,N_space)
    else:
        print('Invalid method - switch to analytical by default')
        for i in range(np.size(ind_table)):
            alpha_table[i]=alpha_coherent_analytical(i,nbar)

    return wignerQ(ene_photon,g0,alpha_table,maxit,option)

## ============ Wigner function calculated for a classical EM field 
## ------ Inputs:
# ene_photon: energy of the photon in eV
# g: Electron-light coupling constant
def wigner_classical(ene_photon,g):
    
    # Momentum of the virtual photon  
    qz = ene_photon*eV/v 

    # Table of indices of the sidebands
    sidebands_axis=np.linspace(-nmax, nmax, 2*nmax+1, dtype=int)

    # Sampling of the z-axis for the number of periods we have chosen
    z = number_of_periods*pi*np.linspace(-1,1,Nz)/qz

    # Initialize the wigner functions
    wigner_exact=np.zeros((2*nmax+1,Nz))*1j # Exact Wigner function

    # Compute the exact Wigner function
    amplitude=jv(sidebands_axis,2*g)
    for j in range(Nz):
        Psi = amplitude*np.exp(1j*sidebands_axis*qz*z[j])
        wigner_exact[:,j] = np.convolve(Psi,np.conj(Psi),'same')
    
    return wigner_exact, sidebands_axis, z