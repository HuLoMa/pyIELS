import warnings
from qutip import Qobj, basis, qdiags

def free_electron(K : int, k : int) -> Qobj :
    r"""
    Function to create a free electron-like qutip object. 
    The state represents an energy scale with K avalaible levels. The level at the (K-1)/2 index has energy E0.
    Levels below (K-1)/2 represent a loss of energy, Level above (K-1)/2 represent a gain of energy

    Parameters
    ----------
    K : int
        Total number of available energy level of the electron. It has to be odd.
    k : int
        Energy level to be populated. At k = 0, the (K-1)/2 index is populated.

    Returns
    -------
    qobj : Qobj
        qutip object of the electron.
    """
    if K%2 :
        qobj = basis(K,k,-(K-1)//2)
    else :
        warnings.warn(f"An odd number of energy levels is expected. Switching from K = {K} to K = {K-1}")
        qobj = basis(K-1,k,-(K-2)//2)
    return qobj

def kick(K : int, shift : int = 1, unitarity : bool = False) -> Qobj :
    """
    Kick operator for a free electron. It shifts the energy level of the electron by the shift value.

    Parameters
    ----------
    K : int
        Total number of available energy levels
    shift : int
        Amount of energy levels by which to shift. It is one in most cases.
    unitarity : bool
        Optional to force unitarity on the kick operator. Only valid if the operator does not apply on the boundaries.
    
    Returns
    -------
    qobj : Qobj
        kick operator qobject
    """
    qobj = qdiags([1]*(K-shift), shift)
    qobj._isunitary = unitarity
    return qobj
