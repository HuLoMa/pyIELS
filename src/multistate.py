from qutip import Qobj, tensor
import numpy as np

class MultiStateElectron(Qobj) :
    def __init__(self,electron_state : Qobj, state : Qobj = None ) :
        self.electron_state = electron_state
        self.states = state
        self.multistate = tensor(electron_state,state)
        super().__init__(self.multistate._data)
