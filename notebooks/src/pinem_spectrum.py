import numpy as np
from math import sqrt
from scipy.special import jv
from src.state_amplitude import Fnl, alpha_Fock_analytical, alpha_Fock_Qutip, alpha_coherent_analytical, alpha_coherent_Qutip, alpha_coherent_squeezed
from src.generic import g_distribution_temporal_averaging, g_distribution_spatial_averaging, progressbar, gaussian


## ============ PINEM Spectrum calculated for a quantized EM field with arbitrary photon amplitudes
## ------ Inputs:
# E: energy axis
# ene_photon: energy of the photon in eV
# g0: Electron-photon coupling constant
# ZLP_width: Energy standard deviation in eV
# alpha_table: table containing the photonic amplitude, the index of the table spans the Fock basis 
# maxit: Cutoff index for the Fock state summation
# nmax: Maximum number of sidebands to calculate
def pinemQ(E,ene_photon,g0,ZLP_width,alpha_table,maxit,nmax,option=None):
    ene_dim=np.size(E)
    spectrum=np.zeros(ene_dim)
    l_set=np.linspace(-nmax,nmax,2*nmax+1,dtype=int)
    E_sidebands=l_set*ene_photon
    
    for l_it in range(2*nmax+1):
        l=l_set[l_it]
        s=0
        for k in range(max(-l,0),maxit):
            s=s+abs(alpha_table[k+l]*Fnl(k,l,g0,option))**2
        spectrum=spectrum+s*gaussian(E,E_sidebands[l_it],ZLP_width)

    return spectrum

## ============ PINEM spectrum calculated for a Fock state 
## ------ Inputs:
# E: energy axis
# ene_photon: energy of the photon in eV
# g0: Electron-photon coupling constant
# ZLP_width: Energy standard deviation in eV
# nbar: Number of photons in the cavity
# methods: precise the methode to use 
#       - "analytical": Using analytically calculated alpha coefficients from https://doi.org/10.1364/OPTICA.404598
#       - "numerical": using QuTip to calculate alpha coefficients
def pinemQ_Fock(E,ene_photon,g0,ZLP_width,nbar,method):
    maxit=50 # Cutoff index for the Fock state summation
    nmax=30 # Maximum number of sidebands to calculate
    N_space=100 # Size of the photonic Hilbert space
    
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

    return pinemQ(E,ene_photon,g0,ZLP_width,alpha_table,maxit,nmax)

## ============ PINEM spectrum calculated for a coherent state 
## ------ Inputs:
# E: energy axis
# ene_photon: energy of the photon in eV
# g0: Electron-photon coupling constant
# ZLP_width: Energy standard deviation in eV
# alpha: Coherent state complex parameter
# methods: precise the methode to use 
#       - "analytical": Using analytically calculated alpha coefficients from https://doi.org/10.1364/OPTICA.404598
#       - "numerical": using QuTip to calculate alpha coefficients
def pinemQ_coherent(E,ene_photon,g0,ZLP_width,alpha,method,option=None):
    maxit=80 # Cutoff index for the Fock state summation
    nmax=50 # Maximum number of sidebands to calculate
    nbar=sqrt(abs(alpha))
    # Initialize the alpha_table
    ind_table=np.linspace(0,maxit+nmax+1,maxit+nmax+2)
    alpha_table=ind_table*0.0*1j
    if method=="analytical":
        for i in range(np.size(ind_table)):
            alpha_table[i]=alpha_coherent_analytical(i,nbar)
    elif method=="numerical":
        N_space=maxit*2+nmax # Size of the photonic Hilbert space
        alpha_table=alpha_coherent_Qutip(ind_table,alpha,N_space)
    else:
        print('Invalid method - switch to analytical by default')
        for i in range(np.size(ind_table)):
            alpha_table[i]=alpha_coherent_analytical(i,nbar)
    
    return pinemQ(E,ene_photon,g0,ZLP_width,alpha_table,maxit,nmax,option)


## ============ PINEM spectrum calculated for a squeezed coherent state 
## ------ Inputs:
# E: energy axis
# ene_photon: energy of the photon in eV
# g0: Electron-photon coupling constant
# ZLP_width: Energy standard deviation in eV
# alpha: Coherent state complex parameter
# zeta: Complex squeezing parameter
def pinemQ_coherent_squeezed(E,ene_photon,g0,ZLP_width,alpha,zeta):
    maxit=50 # Cutoff index for the Fock state summation
    nmax=30 # Maximum number of sidebands to calculate
    N_space=100 # Size of the photonic Hilbert space
    nbar=sqrt(abs(alpha))
    # Initialize the alpha_table
    ind_table=np.linspace(0,maxit+nmax+1,maxit+nmax+2)
    alpha_table=ind_table*0.0*1j

    alpha_table=alpha_coherent_squeezed(ind_table,alpha,zeta,N_space)

    return pinemQ(E,ene_photon,g0,ZLP_width,alpha_table,maxit,nmax)

## ============ Classical PINEM spectrum including spatial OR temporal averaging
## ------ Inputs:
# E: energy axis
# g: Electron-light coupling constant
# amplitude: global scaling factor of the spectrum
# ZLP_offset: ZLP energy offset in eV
# ZLP_width: ZLP standard deviation in eV
# background: parasitic overall background
# ene_photon: energy of the photon in eV
# methods: precise the methode to use 
#       - "temporal": then ratio between the light pulse duration (tau_p) and the electron pulse duration (tau_e) - Rt=tau_p/tau_e 
#       - "spatial":  then ratio between the light beam waist (sigma_p) and the electron beam waist (sigma_e) - Rs=sigma_p/sigma_e
# ratio: temporal or spatial ratio
def pinem_classical(E,g,amplitude,ZLP_offset,ZLP_width,background,ene_photon,averaging,ratio):
    N_g = 501
    g_set = np.linspace(0.001,g-0.001,N_g)
    if averaging=='temporal':
        g_distribution=g_distribution_temporal_averaging(g_set,g, ratio)
    elif averaging=='spatial':
        g_distribution=g_distribution_spatial_averaging(g_set,g, ratio)
    else:
        print('Invalid command - assume temporal-averaging by default')
        g_distribution=g_distribution_temporal_averaging(g_set,g, ratio)
    spectrum = np.zeros_like(E)+background
    for n in range(-32,32):
        spectrum += (g_distribution*(jv(n,2*g_set)**2)*gaussian(E[:,np.newaxis]-ene_photon*n-ZLP_offset,0.,ZLP_width)).sum(axis=-1)
    return amplitude*spectrum
