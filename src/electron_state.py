from qutip import Qobj, tensor
import numpy as np
import math
import functools
import numbers
import qutip.core.data as _data
from qutip import settings
from qutip.core.qobj import _require_equal_type

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
        for mat_dim_ind,dim_list in enumerate(self._qobj.dims) :
            es_dim_index = self.es_dims_indices[mat_dim_ind]
            shape_left = math.prod(dim_list[:es_dim_index])
            # TODO : Check this. it is tricky.
            if len(dim_list) == 1 :
                shape_right = 1
            else :
                shape_right = math.prod(dim_list[es_dim_index:])
            es_indices.append([x for x in range(dim_list[es_dim_index]) for _ in range(shape_right)]*shape_left)
        return es_indices
    
    def check_pure_electron(self) : 
        pass
    
    def copy(self) -> Qobj:
        """Create identical copy"""
        return ElectronState(electron_state=self.electron_state,
                    data=self._qobj.copy(),
                    dim_indices=self.es_dims_indices)
    
    @_require_equal_type
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

    def __radd__(self, other: Qobj | complex) -> Qobj:
        return self.__add__(other)
    
    def __getattr__(self, name):
        return getattr(self._qobj,name)
    
    # Doesn't work, makes an infinite circular call at __init__
    # def __setattr__(self, name, value):
    #     setattr(self._qobj, name, value)

    def __mul__(self, other: complex) -> Qobj:
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
            new_electron_state = _data.mul(self.electron_state._data,other)
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
    
    # TODO : Qobj are allowed if operators else only compatible electron state are possible.
    def __matmul__(self, other: Qobj) -> Qobj:
        if not isinstance(other, Qobj):
            try:
                other = Qobj(other)
            except TypeError:
                return NotImplemented
        new_dims = self._dims @ other._dims
        if new_dims.type == 'scalar':
            return _data.inner(self._data, other._data)
        if self.isket and other.isbra:
            return Qobj(
                _data.matmul_outer(self._data, other._data),
                dims=new_dims,
                isunitary=False,
                copy=False
            )

        return Qobj(
            _data.matmul(self._data, other._data),
            dims=new_dims,
            isunitary=self._isunitary and other._isunitary,
            copy=False
        )
    

if __name__ == '__main__' : 
    from qutip import basis
    alpha = basis(5,2)
    es = ElectronState(alpha)
    print(es + alpha)
    idx = es.get_es_indices()
    es._qobj[*idx]