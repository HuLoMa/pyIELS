import numpy as np
from electron_state import ElectronState
from physics_utils import PhysicalQuantities, gaussian
from scipy import signal

def ewigner_discrete(electron_state : ElectronState,
                     z_axis : np.ndarray,
                     p_values : np.ndarray,
                     energy_levels : bool =False,
                     unit_system : str = 'atomic') :
    hbar = PhysicalQuantities(unit_system=unit_system).hbar 
    assert electron_state._qobj.dims[0] == electron_state._qobj.dims[1] , "The ewigner function can only be applied on a qobj with a symmetric shape"
    qobj_array = electron_state._qobj.data.to_array()
    pks_ket =np.tile(p_values,(qobj_array.shape[0],1))
    pks_bra = np.tile(p_values[:,np.newaxis],(1,qobj_array.shape[0]))
    pks_ket_z = np.tile(pks_ket, z_axis.shape + (1,1))
    pks_bra_z = np.tile(pks_bra, z_axis.shape + (1,1))
    if energy_levels :
        els = electron_state.energy_levels()
        els_ket = np.tile(els,(qobj_array.shape[0],1))
        els_bra = np.tile(els[:,np.newaxis],(1,qobj_array.shape[0]))
        authorized_pks = (els_ket + els_bra)/2
    else :
        authorized_pks = (pks_ket + pks_bra)/2
    # np.unique does not seem to work because of numerical precision
    authorized_pks_1d = np.concatenate((authorized_pks[0],authorized_pks[1:,-1]))
    wigner_elements = (1/(2*np.pi*hbar.magnitude)*np.exp(1.0j/hbar.magnitude*(z_axis[:,np.newaxis,np.newaxis]*(pks_ket_z - pks_bra_z)))*qobj_array)[:,::-1,:]
    lowest_offset = -wigner_elements.shape[1] +1
    highest_offset = wigner_elements.shape[1]
    wigner_pk_discrete_list = [np.trace(wigner_elements,offset,axis1 = 1, axis2 = 2) for offset in range(lowest_offset,highest_offset)]
    wigner_pk_discrete_array = np.array(wigner_pk_discrete_list)

    return wigner_pk_discrete_array, authorized_pks_1d

def ewigner(electron_state : ElectronState,
            z_axis : np.ndarray = None,
            p_values : np.ndarray = None,
            p_res : int = None,
            p_fwhm : float = None,
            phq : PhysicalQuantities = None,
            energy_levels : bool = True,
            radian_z : bool = True,
            unit_system : str = 'atomic') :
    if phq is None :
        assert all((v is not None for v in (z_axis,p_values,p_res,p_fwhm))), "If you are not using the PhysicalQuantities object you need to define all the relevant parameters z_axis, p_values, p_res, p_fwhm"
        hbar = PhysicalQuantities(unit_system=unit_system).hbar
    else :
        phq.unit_system = unit_system
        hbar = phq.hbar
    if z_axis is None :
        assert phq is not None, "You have to either use the PhysicalQuantities helper object or set a z_axis yourself."
        if radian_z : 
            rep_z_axis = phq.to_radian_z_axis()
        else :
            rep_z_axis = z_axis
        z_axis = phq.z_axis
    else :
        if radian_z : 
            assert p_values is not None, "If you want a radian z scale, you need to define a momentum scale (p_values parameter)"
            mom_diff = p_values[1] - p_values[0]
            rep_z_axis = mom_diff*z_axis/hbar.magnitude
        else :
            rep_z_axis = z_axis
    if p_values is None :
        assert phq is not None, "You have to either use the PhysicalQuantities helper object or set a p_axis yourself." 
        p_values = phq.momentums(electron_state.energy_levels())
    if p_res is None :
        assert phq is not None, "You have to either use the PhysicalQuantities helper object or set a p_res yourself." 
        p_res = phq.momentum_resolution
    wigner_discrete, pks = ewigner_discrete(electron_state,z_axis,p_values, energy_levels,unit_system)
    num_z = z_axis.shape[0]
    p_axis = interpolate_p_scale(pks,p_res,wigner_discrete.shape[0])
    # TODO : add support for any shape of the ZLP
    num_p = p_axis.shape[0]

    if p_fwhm is None :
        assert phq is not None, "You have to either use the PhysicalQuantities helper object or set a p_fwhm yourself."
        if energy_levels :
            fwhm = (phq.zlp_qz_fwhm/phq.momentum_difference).magnitude/(2*np.sqrt(2*np.log(2)))
            zlp = gaussian(p_axis,0.0, fwhm)
        else :
            zlp = gaussian(p_axis,0.0, phq.zlp_qz_fwhm.magnitude/(2*np.sqrt(2*np.log(2))))
    else :
        if energy_levels :
            fwhm = (p_fwhm/(p_values[1]-p_values[0]))/(2*np.sqrt(2*np.log(2)))
            zlp = gaussian(p_axis,0.0, fwhm)
        else :
            zlp = gaussian(p_axis, 0.0, p_fwhm/(2*np.sqrt(2*np.log(2))))
    wigner_convolved=np.zeros((num_p,num_z),dtype=complex)
    # Wigner function including the spectral width of the electron source
    # Convolve with ZLP
    wigner_convolved[::p_res,:]=wigner_discrete
    wigner_convolved=signal.fftconvolve(wigner_convolved,np.expand_dims(zlp,axis=1),'same')

    return np.real(wigner_convolved), p_axis, rep_z_axis

def interpolate_p_scale(pks, p_res, wig_d_shape) :
    step = (pks[1]-pks[0])/p_res
    try :
        p_axis = np.arange(pks[0],pks[-1],step)
        assert p_axis.shape[0] == wig_d_shape
    except AssertionError : 
        p_axis = np.arange(pks[0],pks[-1]+step,step)
    return p_axis