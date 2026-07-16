from qutip import Qobj, tensor
import numpy as np
import math
import functools
import numbers
import qutip.core.data as _data
from qutip import settings
from itertools import pairwise

__all__ = ['ElectronState']

class ElectronState :
    def __init__(self,electron_state : Qobj, data : Qobj = None, dim_indices : list[int] = None) :
        self.electron_state = electron_state
        if data is None :
            self._qobj = electron_state.copy()
        else :
            self._qobj = data.copy()
        self.es_dims = self.electron_state.dims
        if dim_indices is None :
            self.es_dims_indices = [0 for i in range(len(self.es_dims))]
        else :
            self.es_dims_indices = dim_indices
        self.es_indices = self.get_es_indices()

    def get_es_indices(self) :
        """
        Fetch the electron state indices from the total q object.
        """
        es_indices = []

        dim_list = self._qobj.dims
         
        es_ind = self.es_dims_indices[0]
        shape_left = math.prod(dim_list[0][:es_ind])
        shape_right = math.prod(dim_list[0][es_ind+1:])
        es_indices.append([x for x in range(dim_list[0][es_ind]) for _ in range(shape_right)]*shape_left)

        es_ind = self.es_dims_indices[1]
        shape_left = math.prod(dim_list[1][:es_ind])
        shape_right = math.prod(dim_list[1][es_ind+1:])
        es_indices.append([x for x in range(dim_list[1][es_ind]) for _ in range(shape_right)]*shape_left)
        
        return es_indices
        # es_indices = []
        
        # for mat_dim_ind,dim_list in enumerate(self._qobj.dims) :
        #     es_dim_index = self.es_dims_indices[mat_dim_ind]
        #     shape_left = math.prod(dim_list[:es_dim_index])
        #     # TODO : Check this. it is tricky.
        #     if len(dim_list) == 1 :
        #         shape_right = 1
        #     else :
        #         shape_right = math.prod(dim_list[es_dim_index:])
        #     es_indices.append([x for x in range(dim_list[es_dim_index]) for _ in range(shape_right)]*shape_left)
        # return es_indices
    
    def check_pure_electron(self) : 
        pass
    
    def copy(self) -> Qobj:
        """Create identical copy"""
        return ElectronState(electron_state=self.electron_state,
                    data=self._qobj.copy(),
                    dim_indices=self.es_dims_indices)
    
    # @_require_equal_type
    def __add__(self, other: Qobj | complex) -> Qobj:
        if other == 0:
            return self.copy()
        try :
            if isinstance(other,ElectronState) :
                assert self.es_indices == other.es_indices
                new_electron_state = self.electron_state + other.electron_state
                return ElectronState(electron_state=new_electron_state,
                            data=Qobj(_data.add(self._data, other._data),
                                    dims=self._dims,
                                    isherm=(self._isherm and other._isherm) or None,
                                    copy=False),
                            dim_indices=self.es_dims_indices)

            raise TypeError("Electron states cannot be summed with other objects.")
        except AssertionError as exc:
            raise ValueError(
                "It is not possible to sum electron states with regular qutip states."
                ) from exc

    def __radd__(self, other: Qobj | complex) :
        return self.__add__(other)
    
    def __getattr__(self, name):
        return getattr(self._qobj,name)
    
    # Doesn't work, makes an infinite circular call at __init__
    # def __setattr__(self, name, value):
    #     setattr(self._qobj, name, value)

    def __mul__(self, other: complex) :
        """
        If other is a Qobj, we dispatch to __matmul__. If not, we
        check that other is a valid complex scalar, i.e., we can do
        complex(other). Otherwise, we return NotImplemented.
        """

        if isinstance(other, Qobj):
            return self.__matmul__(other)
        if isinstance(other,ElectronState) :
            return self.__matmul__(other)

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
        return self.__mul__(other)

    def dag(self) : 
        """Get the Hermitian adjoint of the quantum object."""
        if self._isherm:
            return self.copy()
        new_qobj = self._qobj.dag()
        new_electron_state = self.electron_state.dag()
        return ElectronState(new_electron_state,new_qobj) 
    
    def __getitem__(self, key):
        return self._qobj.__getitem__(key)
    
    # TODO : Qobj are allowed if operators else only compatible electron state are possible.
    def __matmul__(self, other ) :
        new_dims = self._dims @ other._dims
        if new_dims.type == 'scalar':
            return _data.inner(self._data, other._data)
        if isinstance(other, ElectronState):
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
            qobj = Qobj(
            _data.matmul(self._data, other._data),
            dims=new_dims,
            isunitary=self._isunitary and other._isunitary,
            copy=False)
            return ElectronState(new_electron_state,
                                 qobj,
                                 new_dim_indices)
        try:
            assert other.isoper
            # new_electron_state = Qobj(
            #     _data.matmul(self._data, other._data),
            #     dims=new_dims,
            #     isunitary=self._isunitary and other._isunitary,
            #     copy=False)
            # new_dim_indices = self.es_dims_indices[:-1] + other.es_dims_indices[1:]

        except TypeError:
            return NotImplemented
        
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
    
    if len(es_positions) > 1 : 
        raise NotImplementedError("We can't handle tensor of multiple electron states.")

    isherm = args[0]._isherm
    isunitary = args[0]._isunitary
    out_data = args[0].data
    dims_l = [args[0]._dims[0]]
    dims_r = [args[0]._dims[1]]
    if isinstance(args[0],ElectronState) :
        dim_indices = [0,0]
        elec_state = args[0].electron_state

    for i, arg in enumerate(args[1:]):
        out_data = _data.kron(out_data, arg.data)
        # If both _are_ Hermitian and/or unitary, then so is the output, but if
        # both _aren't_, then output still can be.
        isherm = (isherm and arg._isherm) or None
        isunitary = (isunitary and arg._isunitary) or None
        dims_l.append(arg._dims[0])
        dims_r.append(arg._dims[1])
        if isinstance(arg,ElectronState) :
            dim_indices = [i+1,i+1]
            elec_state = arg.electron_state

    qobj = Qobj(out_data,
                dims=[dims_l,dims_r],
                isherm = isherm,
                isunitary=isunitary,
                copy = False)
    
    if qobj.dims[0] == [1] : 
        dim_indices[0] = 0
    if qobj.dims[1] == [1] : 
        dim_indices[1] = 0

    return ElectronState(elec_state,
                qobj,
                dim_indices=dim_indices)

if __name__ == '__main__' : 
    from qutip import basis
    alpha = basis(5,2)
    es = ElectronState(alpha)
    es1 = ElectronState(alpha)
    # print(es + es1)
    # idx = es.get_es_indices()
    # es._qobj[*idx]
    a = basis(5,1) * basis(2,1).dag()
    es * a

from functools import wraps

def vectorize_operation(method):
    """
    Decorator to make a method handle both single custom objects and tuples of them.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        # We assume the first argument (or self) is our wrapper object.
        # If the method is binary (e.g., self + other), we check 'other'.
        
        # Scenario 1: No extra positional arguments (unary operation, e.g., self.dagger())
        if not args:
            # If the wrapper object itself holds a tuple internally
            if isinstance(self.data, tuple):
                # Apply the method to a temporary instance created for each element
                return tuple(method(self.__class__(item), **kwargs) for item in self.data)
            return method(self, **kwargs)

        # Scenario 2: Binary operations (e.g., self + other)
        other = args[0]
        
        # Case A: 'other' is a tuple
        if isinstance(other, tuple):
            # If self is also representing a tuple internally, pair them up (zip)
            if isinstance(self.data, tuple):
                if len(self.data) != len(other):
                    raise ValueError("Tuple lengths must match for mathematical operations.")
                return tuple(
                    method(self.__class__(s), self._unwrap(o), *args[1:], **kwargs)
                    for s, o in zip(self.data, other)
                )
            # If self is single, distribute self across the tuple of 'other'
            return tuple(
                method(self, self._unwrap(o), *args[1:], **kwargs) 
                for o in other
            )
            
        # Case B: 'other' is single, but 'self' holds a tuple internally
        if isinstance(self.data, tuple):
            return tuple(
                method(self.__class__(s), self._unwrap(other), *args[1:], **kwargs)
                for s in self.data
            )

        # Case C: Both are single objects
        return method(self, *args, **kwargs)

    return wrapper