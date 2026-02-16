import numpy as np
from math import pi
from scipy import signal

from src.generic import gaussian
from src.units_contants import eV, mm, gamma, v
from src.simulation_parameters import number_of_periods, nmax, Nz, dispersion

## ============ Convolve the Wigner function with the ZLP
## ------ Inputs:
# wigner_exact: An exact Wigner function 
# ene_photon: energy of the photon in eV
# ZLP_width: ZLP standard deviation in eV
def wigner_spectral_convolve(wigner, ene_photon, ZLP_width):
    # Momentum of the virtual photon 
    qz = ene_photon*eV/v          

    # ZLP FWHM and standard deviation in sideband space
    ZLP_fwhm=eV*ZLP_width/(qz*v) 
    ZLP_sig=ZLP_fwhm/(2*np.sqrt(2*np.log(2)))        

    # Calculate the number of sampling point in the sidebands space
    Np=int((2*nmax+1)*dispersion)
    # Sampling the sidebands space
    p_axis=np.linspace(-nmax, nmax, Np)/2

    # Define the ZLP
    ZLP=gaussian(p_axis, 0.0, ZLP_sig) # Calculate the ZLP
  
    # Initialize the new Wigner function
    wigner_convolved=np.zeros((Np,Nz))*1j # Wigner function including the spectral width of the electron source
    # Convolve with ZLP
    wigner_convolved[0::dispersion,:]=wigner
    wigner_convolved=signal.fftconvolve(wigner_convolved,np.expand_dims(ZLP,axis=1),'same')

    return np.real(wigner_convolved), p_axis

## ============ Propagate the Wigner function over a given distance in mm
## ------ Inputs:
# wigner: The Wigner function to propagate
# p_axis: the momentum axis of the Wigner function
# ene_photon: energy of the photon in eV
# prop_mm: Propagation distance after interaction in mm
def wigner_propagate(wigner, p_axis, ene_photon, prop_mm):
    
    # Propagation distance after interaction in mm
    dist_prop=prop_mm*mm               
    # Momentum of the virtual photon 
    qz = ene_photon*eV/v   
    
    # Sampling of the z-axis for the number of periods we have chosen
    z = number_of_periods*pi*np.linspace(-1,1,Nz)/qz
    delta_z=z[2]-z[1]

    # Shear for each sidebands
    shear=(p_axis*qz/v)/gamma**3*dist_prop/delta_z
    # Apply shear to the Wigner function
    for k in range(np.size(p_axis)):
        wigner[k,:]=np.roll(wigner[k,:],int(shear[k]))

    return wigner