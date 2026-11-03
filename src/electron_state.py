r"""
Core class and functions of pyiels. 
"""

from qutip import Qobj, tensor
import numpy as np
import math
import qutip.core.data as _data
from qutip import settings
import warnings

__all__ = ['ElectronState']

class ElectronState :
    def __init__(self,electron_state : Qobj,
                 data : Qobj = None,
                 dim_indices : list[list[int]] = None,
                 e0_index : int = None) :
        self.electron_state = electron_state
        if e0_index is None :
            self.e0_index = max(self.electron_state.shape)//2
        else :
            self.e0_index = e0_index
        if data is None :
            self._qobj = electron_state.copy()
        else :
            self._qobj = data.copy()
        self.es_dims = self.electron_state.dims
        if dim_indices is None :
            self.es_dims_indices = [list(range(len(self.es_dims[0]))),
                                    list(range(len(self.es_dims[1])))]
        else :
            self.es_dims_indices = dim_indices
        self.es_indices = self.get_es_indices()

    def get_es_indices(self) :
        """
        Fetch the electron state indices from the total q object.
        """
        es_indices = []
        dim_list = self._qobj.dims
        for es_ind in self.es_dims_indices[0] :
            single_es_indices = []
            shape_left = math.prod(dim_list[0][:es_ind])
            shape_right = math.prod(dim_list[0][es_ind+1:])
            single_es_indices.append([x for x in range(dim_list[0][es_ind]) for _ in range(shape_right)]*shape_left)
            es_indices.append(single_es_indices[0])

        for es_ind in self.es_dims_indices[1] :
            single_es_indices = []
            shape_left = math.prod(dim_list[1][:es_ind])
            shape_right = math.prod(dim_list[1][es_ind+1:])
            single_es_indices.append([x for x in range(dim_list[1][es_ind]) for _ in range(shape_right)]*shape_left)

            es_indices.append(single_es_indices[0])

        return es_indices
    
    def check_pure_electron(self) :
        raise NotImplementedError
    
    def copy(self) -> Qobj:
        """Create identical copy"""
        return ElectronState(electron_state=self.electron_state.copy(),
                    data=self._qobj.copy(),
                    dim_indices=self.es_dims_indices,
                    e0_index = self.e0_index)
    
    # @_require_equal_type
    def __add__(self, other: Qobj | complex) :
        if other == 0:
            return self.copy()
        try :
            if isinstance(other,ElectronState) :
                # TODO : catch assertion errors and work on error messages.
                assert self.es_indices == other.es_indices
                new_electron_state = self.electron_state + other.electron_state
                return ElectronState(electron_state=new_electron_state,
                            data=Qobj(_data.add(self._data, other._data),
                                    dims=self._dims,
                                    isherm=(self._isherm and other._isherm) or None,
                                    copy=False),
                            dim_indices=self.es_dims_indices,
                            e0_index=self.e0_index)

            raise TypeError("Electron states cannot be summed with other objects.")
        except AssertionError as exc:
            raise ValueError(
                "It is not possible to sum electron states with regular qutip states."
                ) from exc

    def __radd__(self, other: Qobj | complex) :
        return self.__add__(other)

    def __sub__(self, other : Qobj | complex) : 
        if other == 0:
            return self.copy()
        try :
            if isinstance(other,ElectronState) :
                assert self.es_indices == other.es_indices
                new_electron_state = self.electron_state + other.electron_state
                return ElectronState(electron_state=new_electron_state,
                            data=Qobj(_data.sub(self._data, other._data),
                                    dims=self._dims,
                                    isherm=(self._isherm and other._isherm) or None,
                                    copy=False),
                            dim_indices=self.es_dims_indices)

            raise TypeError("Electron states cannot be subtracted with other objects.")
        except AssertionError as exc:
            raise ValueError(
                "It is not possible to subtract electron states with regular qutip states."
                ) from exc

    def __rsub__(self, other) :
        return self.__add__(other)
    
    def __getattr__(self, name):
        return getattr(self._qobj,name)
    
    # Doesn't work, makes an infinite circular call at __init__
    # def __setattr__(self, name, value):
    #     setattr(self._qobj, name, value)

    def __mul__(self, other: complex,reverse = False) :
        """
        If other is a Qobj, we dispatch to __matmul__. If not, we
        check that other is a valid complex scalar, i.e., we can do
        complex(other). Otherwise, we return NotImplemented.
        """

        if isinstance(other, Qobj):
            return self.__matmul__(other,reverse)
        if isinstance(other,ElectronState) :
            return self.__matmul__(other,reverse)

        # We send other to mul instead of complex(other) to be more flexible.
        # The dispatcher can then decide how to handle other and return
        # TypeError if it does not know what to do with the type of other.
        try:
            out = _data.mul(self._data, other)
            new_electron_state = Qobj(_data.mul(self.electron_state._data,other))
        except TypeError:
            return NotImplemented

        # Infer isherm and isunitary if possible
        try:
            multiplier = complex(other)
            isherm = (self._isherm and multiplier.imag == 0) or None
            isunitary = (abs(abs(multiplier) - 1) < settings.core['atol']
                         if self._isunitary else None)
        except TypeError:
            isherm = None
            isunitary = None

        return ElectronState(electron_state=new_electron_state,
                             data=Qobj(out,
                                       dims=self._dims,
                                       isherm=isherm,
                                       isunitary=isunitary,
                                       copy=False),
                             dim_indices=self.es_dims_indices)

    def __rmul__(self, other: complex) -> Qobj:
        # Shouldn't be here unless `other.__mul__` has already been tried, so
        # we _shouldn't_ check that `other` is `Qobj`.
        return self.__mul__(other,reverse=True)

    def dag(self) :
        """Get the Hermitian adjoint of the quantum object."""
        if self._isherm:
            return self.copy()
        new_qobj = self._qobj.dag()
        new_electron_state = self.electron_state.dag()
        new_dim_indices = [self.es_dims_indices[1],self.es_dims_indices[0]]
        return ElectronState(new_electron_state,new_qobj,new_dim_indices) 
    
    def __getitem__(self, key):
        return self._qobj.__getitem__(key)
    
    # TODO : Qobj are allowed if operators else only compatible electron state are possible.
    def __matmul__(self, other,reverse = False) :
        if isinstance(other, ElectronState):
            return self._es_matmul(other)
        elif isinstance(other, Qobj) : 
            return self._qobj_matmul(other,reverse)
        else : 
            return NotImplemented

    # For ElectronState1 * ElectronState2 multiplications, _rmatmul_ is unecessary since ElectronState1.__matmul__ is always called first.
    def _es_matmul(self,other) :
        new_dims = self._dims @ other._dims
        new_electron_state = self.electron_state@other.electron_state
        new_dim_indices = self.es_dims_indices[:-1] + other.es_dims_indices[1:]
        if self.isket and other.isbra :
            qobj = Qobj(
                _data.matmul_outer(self._data, other._data),
                dims=new_dims,
                isunitary=False,
                copy=False)
            return ElectronState(new_electron_state,
                                    qobj,
                                    new_dim_indices)
        else : 
            qobj = Qobj(
                _data.matmul(self._data, other._data),
                dims=new_dims,
                isunitary=self._isunitary and other._isunitary,
                copy=False)
            return ElectronState(new_electron_state,
                                    qobj,
                                    new_dim_indices)

    def _qobj_matmul(self,other,reverse = False) : 
        assert other.isoper, "When multiplying ElectronState with Qobj, only Qobj operators are allowed." #TODO : improve error message
        if reverse :
            new_dims = other._dims @ self._dims
            if new_dims.type == 'scalar':
                # TODO : Test this
                return _data.inner(self._data, other._data)
            qobj = Qobj(
                _data.matmul(other._data, self._data),
                dims=new_dims,
                isunitary=self._isunitary and other._isunitary,
                copy=False)
            
            try : 
                es = other@self.electron_state
            except TypeError :
                warnings.warn("The electron state and operator don't have matching dimensions. We recommend using electron state tensor operator")
                # TODO : improve error message
                es = self.electron_state
                
            return ElectronState(es,
                                 qobj,
                                 self.es_dims_indices)
        else :    
            new_dims = self._dims @ other._dims
            if new_dims.type == 'scalar':
                # TODO : Test this
                return _data.inner(self._data, other._data)
            if self.isket and other.isbra :
                                    qobj = Qobj(
                                    _data.matmul_outer(self._data, other._data),
                                    dims=new_dims,
                                    isunitary=False,
                                    copy=False)
            else :
                qobj = Qobj(
                            _data.matmul(self._data, other._data),
                            dims=new_dims,
                            isunitary=self._isunitary and other._isunitary,
                            copy=False)
            return ElectronState(self.electron_state,
                                    qobj,
                                    self.es_dims_indices)
        
    def expm(self, dtype = None) : 
        if not self._qobj._dims.issquare:
            raise TypeError("expm is only valid for square operators")
        if dtype is None and isinstance(self._qobj.data, (_data.CSR, _data.Dia)):
            dtype = _data.Dense
        new_electron_state = self.electron_state.expm()
        new_qobj = self._qobj.expm()
        return ElectronState(new_electron_state,new_qobj)

    def ptrace(self,sel : int | list[int]) :
        qutip_ptrace = self._qobj.ptrace(sel)
        orig_dims = self._qobj.dims
        # new_dims = qutip_ptrace.dims
        intersection_dim0 = list(set(sel).intersection(self.es_dims_indices[0]))
        intersection_dim1 = list(set(sel).intersection(self.es_dims_indices[1]))
        if len(intersection_dim0) >= len(intersection_dim1) :
            intersection = intersection_dim0
        else :
            intersection = intersection_dim1
        if len(intersection )  == 0:
            return qutip_ptrace
        else :
            part_new_es_indices = []
            i = 0
            for ind, _ in enumerate(orig_dims) :
                if ind in intersection :
                    part_new_es_indices.append(i)
                    i+= 1
            new_es_indices = [part_new_es_indices,part_new_es_indices]
            # TODO check what should happen to the electron state
            return ElectronState(electron_state=self.electron_state,data=qutip_ptrace,dim_indices=new_es_indices)

    def energy_levels(self) :
        # TODO : Now we are only going to use one set of energy levels as we use this function for density matrices. 
        if self._qobj.shape[0] == 1 :
            es_inds = self.es_indices[1]
        elif self._qobj.shape[1] == 1 :
            es_inds = self.es_indices[0]
        elif self._qobj.shape[0] == self._qobj.shape[1] :
            es_inds = self.es_indices[0]
        else :
            raise NotImplementedError
        
        energy_levels = np.array(es_inds) - self.e0_index
        return energy_levels

#  , momentum_points : int, zlp_fwhm : float
    # def ewigner_discrete(self,z_axis : np.array,unit : str = 'kg.m/s') :
    #     assert self._qobj.dims[0] == self._qobj.dims[1] , "The ewigner function can only be applied on a qobj with a symmetric shape"
    #     pks = self.momentum_rep(unit)
    #     qobj_array = self._qobj.data.to_array()
    #     pks_ket =np.tile(pks[0],(qobj_array.shape[0],1))
    #     pks_bra = np.tile(pks[1][:,np.newaxis],(1,qobj_array.shape[1]))
    #     pks_ket_z = np.tile(pks_ket, z_axis.shape + (1,1))
    #     pks_bra_z = np.tile(pks_bra, z_axis.shape + (1,1))
    #     authorized_pks = (pks_ket + pks_bra)/2
    #     # np.unique does not seem to work because of numerical precision
    #     authorized_pks_1d = np.concatenate((authorized_pks[0],authorized_pks[1:,-1]))
    #     wigner_elements = (1/(2*np.pi*self.phq.hbar.magnitude)*np.exp(1.0j/self.phq.hbar.magnitude*(z_axis[:,np.newaxis,np.newaxis]*(pks_ket_z - pks_bra_z)))*qobj_array)[:,::-1,:]
    #     lowest_offset = -wigner_elements.shape[1] +1
    #     highest_offset = wigner_elements.shape[1]
    #     wigner_pk_discrete_list = [np.trace(wigner_elements,offset,axis1 = 1, axis2 = 2) for offset in range(lowest_offset,highest_offset)]
    #     wigner_pk_discrete_array = np.array(wigner_pk_discrete_list)

    #     return wigner_pk_discrete_array, authorized_pks_1d
        
        
        # assert self._qobj.dims[0] == self._qobj.dims[1] , "The ewigner function can only be applied on a qobj with a symmetric shape"
        # pks = self.momentum_rep(unit)
        # qobj_array = self._qobj.data.to_array()
        # pks_ket =np.tile(pks[0],(qobj_array.shape[0],1))
        # pks_bra = np.tile(pks[1][:,np.newaxis],(1,qobj_array.shape[1]))
        # # pks_ket_z = np.tile(pks_ket, z_axis.shape + (1,1))
        # # pks_bra_z = np.tile(pks_bra, z_axis.shape + (1,1))
        # authorized_pks = (pks_ket + pks_bra)/2
        # authorized_pks_1d = np.unique(authorized_pks)
        # Nz = z_axis.shape[0]
        # Np = authorized_pks_1d.shape[0]*momentum_points
        # ps = np.linspace(np.min(pks),np.max(pks),Np)
        # zlp_width_eV = zlp_fwhm/(2*np.sqrt(2*np.log(2))) * self.phq.ureg.eV
        # zlp_width_qz = (zlp_width_eV/self.photon_energy).to_base_units()
        # wigner_convolved=np.zeros((Np,Nz),dtype=complex)
        # for i in range(pks_ket.shape[0]) :
        #     for j in range(pks_ket.shape[1]) :
        #         wigner_elements = np.expand_dims(1/(2*np.pi*self.phq.hbar.magnitude)*np.exp(1.0j/self.phq.hbar.magnitude*(z_axis*(pks_ket[i,j] - pks_bra[i,j])))*qobj_array[i,j],axis = 0)
        #         zlp = np.expand_dims(gaussian(ps,mu = (pks_ket[i,j]+pks_bra[i,j])/2,sigma = zlp_width_qz.magnitude),axis = 1)
        #         wigner_convolved += zlp * wigner_elements
        # return np.real(wigner_convolved)
        
        
        
        # wigner_convolved=np.zeros((Np,Nz) + self._qobj.shape,dtype=complex) # Wigner function including the spectral width of the electron source
        #     # Convolve with ZLP
        # wigner_convolved[::momentum_points,:,:,:]=wigner_elements
        # wigner_convolved=signal.fftconvolve(wigner_convolved,zlp[:,np.newaxis,np.newaxis,np.newaxis],'same')
        # wigner_sum = wigner_convolved.sum(axis = (2,3))
        # *qobj_array[np.newaxis,:]
        # dim = authorized_pks_1d.shape[0]
        # wigner_pk_discrete_list = [np.trace(wigner_elements,offset,axis1 = 1, axis2 = 2) for offset in range(-dim//2,dim//2)]
        # wigner_pk_discrete_array = np.array(wigner_pk_discrete_list)

        # return np.real(wigner_convolved), authorized_pks_1d

    # def ewigner(self, z_axis, momentum_points : int, zlp_shape : float, unit : str = 'kg.m/s') :
    #     wigner_discrete, pks = self.ewigner_discrete(z_axis,unit)
    #     Nz = z_axis.shape[0]
    #     Np = pks.shape[0]*momentum_points
    #     ps = np.linspace(np.min(pks),np.max(pks),Np)
    #     if isinstance(zlp_shape,float) :
    #         zlp_width_eV = zlp_shape/(2*np.sqrt(2*np.log(2))) * self.phq.ureg.eV
    #         zlp_width_qz = zlp_width_eV.to_base_units()/self.electron_speed
    #         zlp = gaussian(ps,mu = 0,sigma = zlp_width_qz.magnitude)
    #     else :
    #         assert zlp_shape.shape == ps.shape
    #         zlp = zlp_shape
    #     wigner_convolved=np.zeros((Np,Nz),dtype=complex) # Wigner function including the spectral width of the electron source
    #         # Convolve with ZLP
    #     wigner_convolved[::momentum_points,:]=wigner_discrete
    #     wigner_convolved=signal.fftconvolve(wigner_convolved,np.expand_dims(zlp,axis=1),'same')

    #     return np.real(wigner_convolved), ps
        

def etensor(*args: Qobj | ElectronState) -> Qobj | ElectronState:
    """Calculates the tensor product of input operators.

    Parameters
    ----------
    args : array_like
        ``list`` or ``array`` of quantum objects for tensor product.

    Returns
    -------
    obj : qobj
        A composite quantum object.

    Examples
    --------
    >>> tensor([sigmax(), sigmax()]) # doctest: +SKIP
    Quantum object: dims = [[2, 2], [2, 2]], \
shape = [4, 4], type = oper, isHerm = True
    Qobj data =
    [[ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]
     [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]
     [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
     [ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]]
    """
    if not args:
        raise TypeError("Requires at least one input argument")
    if len(args) == 1 and isinstance(args[0], (Qobj, ElectronState)):
        return args[0].copy()
    if len(args) == 1:
        try:
            args = tuple(args[0])
        except TypeError:
            raise TypeError("requires Qobj or ElectronState operands") from None
    if not all(isinstance(q, (Qobj, ElectronState)) for q in args):
        raise TypeError("requires Qobj or ElectronState operands")
    
    es_positions = []
    for i,arg in enumerate(args) :
        if isinstance(arg,ElectronState) :
            es_positions.append(i)
    
    if len(es_positions) == 0 :
        raise ValueError("None of the arguments are Electron states. The etensor applies on ElectronState, use qutip tensor for qutip only usage.")
    
    # if len(es_positions) > 1 : 
    #     raise NotImplementedError("We can't handle tensor of multiple electron states.")

    dim_indices = [[],[]]
    elec_state = []
    isherm = args[0]._isherm
    isunitary = args[0]._isunitary
    out_data = args[0].data
    dims_l = [args[0]._dims[0]]
    dims_r = [args[0]._dims[1]]
    if isinstance(args[0],ElectronState) :
        dim_indices[0].append(0)
        dim_indices[1].append(0)
        elec_state.append(args[0].electron_state)

    for i, arg in enumerate(args[1:]):
        out_data = _data.kron(out_data, arg.data)
        # If both _are_ Hermitian and/or unitary, then so is the output, but if
        # both _aren't_, then output still can be.
        isherm = (isherm and arg._isherm) or None
        isunitary = (isunitary and arg._isunitary) or None
        dims_l.append(arg._dims[0])
        dims_r.append(arg._dims[1])
        if isinstance(arg,ElectronState) :
            # TODO : I am not fully sure it is correct, to be checked
            dim_indices[0].append(i+1)
            dim_indices[1].append(i+1)
            elec_state.append(arg.electron_state)

    qobj = Qobj(out_data,
                dims=[dims_l,dims_r],
                isherm = isherm,
                isunitary=isunitary,
                copy = False)
    
    final_elec_state = tensor(elec_state)
    
    if qobj.dims[0] == [1] :
        dim_indices[0] = [0]
    if qobj.dims[1] == [1] :
        dim_indices[1] = [0]

    return ElectronState(final_elec_state,
                qobj,
                dim_indices=dim_indices)
