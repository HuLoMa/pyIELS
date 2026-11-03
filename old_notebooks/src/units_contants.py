from math import sqrt

## Define the units we use for the simulation
# We work in atomic units (https://en.wikipedia.org/wiki/Atomic_units) - hbar, e, m_e and 4\pi\epsilon_0 = 1
c0= 299792458 #speed of light in m.s-1
a0=5.29177e-11 #Bohr radius in m - atomic unit of length
Eh= 27.211386 # Hartree energy in eV - atomic unit of energy
fine_struc=7.2973525664e-3 # Fine structure constant
t_atom=a0/(fine_struc*c0) # a0/(alpha*c) electron velocity in 1st Bohr orbit - atomic unit of time

eV = 1/Eh # eV in a.u.
nm = 1e-9/a0  # nm in a.u.
mm = 1e6*nm
c = c0/a0*t_atom # speed of light in a.u.

fs  = 1e-15/t_atom # femtosecond in a.u.
ps = 1e3*fs

T = 200e3*eV #Electron kinetic energy in a.u.
m = 1+T/c**2
v = c*sqrt(1-1/m**2) 
gamma = 1/sqrt(1-(v/c)**2) #Lorentz contraction factor
