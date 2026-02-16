from qutip import basis, tensor
import numpy as np
from math import log2
from src.generic import gaussian

## ============ Convert the PINEM wavefunction in the cQED basis (Ising-like) into the PINEM wavefunction in the floquet basis. We thus reduce the dimension of the Hilbert space from 2^N to N.
## ------ Inputs:
# psi_ising: PINEM wavefunction in the cQED basis (Ising-like)
## ------ Outputs:
# psi_floquet: PINEM wavefunction in the floquet basis
# cn: Wavefunction complex coefficients in the floquet basis
# basis_PINEM: PINEM basis
def ising2floquet(psi_ising):
    # Number of site from the dimension of the wavefunction
    N=int(log2(np.shape(psi_ising)[0]))
    # Initialise the Floquet basis
    basis_PINEM=[]
    # Coefficient of the PINEM wavefunction in the floquet basis
    cn=np.zeros((N))*1j
    # Initialize PINEM wavefunction in the floquet basis
    psi_floquet=0

    # For each site
    for n in range(N):
        # We initialize a buffer to store the state of each site
        buffer_Ising=[]
        # For each site, we initialize an empty site
        for k in range(N):
                buffer_Ising.append(basis(2,0))
        # Except for the nth site, which we set as excited
        buffer_Ising[n]=basis(2,1)
        # We convert this list of individual site states into a tensor wavefunction
        test=tensor(buffer_Ising)
        # We then compute the overlap between this elementary state with the input wavefunction
        # to get the occupation amplitude of state n
        cn[n]=test.dag()*psi_ising
        # This correspond to the amplitude of the nth component in the Floquet basis
        basis_PINEM.append(basis(N,n))
        # We add the corresponding component to the output Floquet wavefunction
        psi_floquet+=cn[n]*basis_PINEM[n]

    return psi_floquet, cn, basis_PINEM

## ============ Convert the PINEM wavefunction in the cQED basis (Ising-like) into the PINEM wavefunction in the floquet basis. We thus reduce the dimension of the Hilbert space from 2^N to N. 
## ============ this implementation includes the photon degress of freedom
## ------ Inputs:
# psi_ising: PINEM wavefunction in the cQED basis (Ising-like)
## ------ Outputs:
# psi_floquet: PINEM wavefunction in the floquet basis
# cn: Wavefunction complex coefficients in the floquet basis
# basis_PINEM: PINEM basis
def ising2floquet_phot(psi_ising,M,N):
    # Initialise the Floquet basis
    ket0=tensor(basis(M,0),basis(N,0)) # Define data-type
    basis_PINEM=np.full((M,N), ket0)
    # Coefficient of the PINEM wavefunction in the floquet basis
    cn=np.zeros((M,N))*1j
    # Initialize PINEM wavefunction in the floquet basis
    psi_floquet=0
    
    # For each Fock state
    for n_fock in range(M):
        # For each electron state
        for n in range(N):
            # We initialize a buffer to store the state of each site
            buffer_Ising=[]
            # The first component is the Fock number
            buffer_Ising.append(basis(M,n_fock))
            # For each site, we initialize an empty site
            for k in range(N):
                buffer_Ising.append(basis(2,0))
            # Except for the nth site, which we set as excited (the +1 is to include the fact that the first component corresponds to the photon)
            buffer_Ising[n+1]=basis(2,1)
            # We convert this list of individual site states into a tensor wavefunction
            test=tensor(buffer_Ising)
            # We then compute the overlap between this elementary state with the input wavefunction
            # to get the occupation amplitude of state n
            cn[n_fock,n]=test.dag()*psi_ising
            # This correspond to the amplitude of the nth component in the Floquet basis
            ket=tensor(basis(M,n_fock),basis(N,n))
            basis_PINEM[n_fock,n]=ket
            # We add the corresponding component to the output Floquet wavefunction
            psi_floquet+=cn[n_fock,n]*basis_PINEM[n_fock,n]

    return psi_floquet, cn, basis_PINEM


## ============ Compute a PINEM spectrum from a PINEM density matrix in the floquet basis.
## ------ Inputs:
# E: Energy range of the spectrum
# rho: PINEM density matrix in the floquet basis
# ZLP_width: zero-loss width in eV
# ene_photon: energy of the photon in eV
## ------ Outputs:
# spectrum: PINEM spectrum over E
def pinem_dm2spec(E,rho,ZLP_width,ene_photon):
    # Initialize the spectrum
    spectrum=0
    # Extract the total number of states from the dimension of the density matrix
    N_state=np.size(rho[0,:])
    # Number of sidebands
    ns=(N_state-1)/2
    # Loop over the electron states
    for n in range(N_state):
        # Extract the amplitude from the diagonal and multiply by the corresponding gaussian function
        spectrum += rho[n,n]*gaussian(E,ene_photon*(n-ns),ZLP_width)
    return spectrum