from pint import Quantity, UnitRegistry
import numpy as np

class PhysicalQuantities : 
    def __init__(self,
                 lambda_p : float | Quantity = 800.0,
                 high_tension : float | Quantity = 200.0,
                 zlp_fwhm : float | Quantity = 0.7,
                 momentum_resolution : int = 40,
                 unit_system : str = 'atomic') : 
        self.unit_system = unit_system
        self.ureg = UnitRegistry(system = unit_system)
        self._hbar = 1.05457181e-34 * self.ureg.joule * self.ureg.second
        self._h = 6.2607015e-34 * self.ureg.joule * self.ureg.second
        self._c = 299792458 * self.ureg.meter / self.ureg.second
        self._e = 1.602176565e-19 * self.ureg.coulomb
        self._me = 511 * self.ureg.keV
        self._unit_time = 1* self.ureg.atomic_unit_of_time
        self.momentum_resolution = momentum_resolution
        if isinstance(lambda_p, Quantity) : 
            self._lambda_p = lambda_p.to('nm').magnitude * self.ureg.nanometer
        else : 
            self._lambda_p = lambda_p * self.ureg.nanometer
        if isinstance(high_tension, Quantity) : 
            self._ht = high_tension.to('kV').magnitude * self.ureg.kV
        else :
            self._ht = high_tension * self.ureg.kV
        self._photon_pulse = 2*np.pi*self.c/self.lambda_p
        self._photon_energy = self.h * self.c / self.lambda_p
        self._electron_speed = self.calculate_electron_speed()
        self._momentum_difference = self.hbar*self.photon_pulse/self.electron_speed
        if isinstance(zlp_fwhm, Quantity) :
            self._zlp_fwhm = zlp_fwhm.to('eV').magnitude * self.ureg.eV
        else :
            self._zlp_fwhm = zlp_fwhm * self.ureg.eV
        self._zlp_qz_fwhm = self.zlp_fwhm / self.electron_speed
        self._z_axis = None

    def calculate_electron_speed(self) :
        return self.c*np.sqrt(1 - (1/np.power(1 + self.ht*(self.e/self.me),2)))

    def momentums(self,energy_levels : np.ndarray) -> np.ndarray : 
        return self.momentum_difference.magnitude*energy_levels

    # def p_axis(self, momentums, momentum_resolution : int = 40) -> np.ndarray :
    #     return np.linspace(np.min(momentums) - (self.momentum_difference.magnitude*momentum_resolution)//2,
    #                        np.max(momentums) + (self.momentum_difference.magnitude*momentum_resolution)//2,
    #                        num = momentum_resolution*momentums.shape[0])

    def set_linear_z_axis(self,z_min : float | Quantity, z_max : float | Quantity,n_points : int = 1024) -> None :
        if isinstance(z_min, Quantity) :
            z_min_q = z_min.to('mm').magnitude * self.ureg.mm
        else : 
            zmin_q = z_min * self.ureg.mm
        if isinstance(z_max,Quantity) :
            z_max_q = z_max.to('mm').magnitude * self.ureg.mm
        else :
            z_max_q = z_max * self.ureg.mm
        z_axis = np.linspace(z_min_q.to_base_units().magnitude,
                               z_max_q.to_base_units().magnitude,
                               num = n_points)
        self._z_axis = z_axis

    def set_period_z_axis(self,num_periods : int = 1, n_points : int = 1024) -> None : 
        z_axis = num_periods*2*np.pi*np.linspace(-1,1,n_points)*self.hbar.magnitude/self.momentum_difference.magnitude
        self._z_axis = z_axis

    def to_radian_z_axis(self) :
        return self.z_axis*self.momentum_difference.magnitude/self.hbar.magnitude

    @property
    def hbar(self) -> Quantity :
        '''
        Planck's constant divided by 2 pi expressed in the defined unit system.

        Returns
        -------
        Quantity with value hbar 
        '''
        return self._hbar.to_base_units()

    @property
    def h(self) -> Quantity :
        '''
        Planck's constant expressed in the defined unit system.

        Returns
        -------
        Quantity with value h 
        ''' 
        return self._h.to_base_units()

    @property
    def c(self) -> Quantity :
        '''
        Light speed expressed in the defined unit system.

        Returns
        -------
        Quantity with value c
        ''' 
        return self._c.to_base_units()

    @property
    def e(self) -> Quantity :
        '''
        Electron charge expressed in the defined unit system.

        Returns
        -------
        Quantity with value e
        ''' 
        return self._e.to_base_units()

    @property
    def me(self) -> Quantity : 
        '''
        Electron rest mass expressed in the defined unit system.

        Returns
        -------
        Quantity with value me 
        ''' 
        return self._me.to_base_units()

    @property
    def lambda_p(self) -> Quantity :
        '''
        Photon wavelength expressed in the defined unit system.

        Returns
        -------
        Quantity with value lambda 
        ''' 
        return self._lambda_p.to_base_units()

    @property
    def ht(self) -> Quantity :
        '''
        Electron acceleration voltage expressed in the defined unit system.

        Returns
        -------
        Quantity with value HT 
        ''' 
        return self._ht.to_base_units()

    @property
    def unit_time(self) :
        return self._unit_time.to_base_units()

    @property
    def photon_pulse(self) : 
        return self._photon_pulse.to_base_units()

    @property
    def photon_energy(self) :
        return self._photon_energy.to_base_units()

    @property
    def electron_speed(self) :
        return self._electron_speed.to_base_units()

    @property
    def momentum_difference(self) :
        return self._momentum_difference.to_base_units()

    @property
    def zlp_fwhm(self) :
        return self._zlp_fwhm.to_base_units()

    @property
    def zlp_qz_fwhm(self) :
        return self._zlp_qz_fwhm.to_base_units()

    @property
    def z_axis(self) :
        if self._z_axis is None :
            self.set_period_z_axis()
            return self._z_axis
        else : 
            return self._z_axis

def gaussian(x : np.array,mu : float,sigma : float) :
    prefactor = 1/(sigma * np.sqrt(2*np.pi))
    variable = -0.5*np.pow(x-mu,2)/np.pow(sigma,2)
    return prefactor*np.exp(variable)


        # for dims in self.es_dims_indices :
        #     single_es_indices = []
        #     es_ind = dims[0]
        #     shape_left = math.prod(dim_list[0][:es_ind])
        #     shape_right = math.prod(dim_list[0][es_ind+1:])
        #     single_es_indices.append([x for x in range(dim_list[0][es_ind]) for _ in range(shape_right)]*shape_left)

        #     es_ind = dims[1]
        #     shape_left = math.prod(dim_list[1][:es_ind])
        #     shape_right = math.prod(dim_list[1][es_ind+1:])
        #     single_es_indices.append([x for x in range(dim_list[1][es_ind]) for _ in range(shape_right)]*shape_left)

        #     es_indices.append(single_es_indices)
        
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