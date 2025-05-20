import numpy as np
def operator(x):
    n = len(x)
    M = Operator((n,n),complex, buffer= np.array(x, dtype = complex))
    return M
def kraus(x):
    x = np.array(x, dtype = complex)
    M = Kraus(x.shape,complex, buffer= x)
    return M

class Operator(np.ndarray):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        return instance
    def __mul__(self,y):
        if isinstance(y, Operator):
            return np.matmul(self,y)
        else:
            return super().__mul__(y)
    def tensor(self, y):
        return np.kron(self,y)

    def adjoint(self):
        return np.swapaxes(np.conj(self),-1,-2)

    def is_unitary(self):
        return self * self.adjoint() == np.identity(len(self))

    def __eq__(self,x):
      return np.max(self - x) < 1e-10


class Kraus(Operator):
    pass

class DensityMatrix(Operator): #allow for list of DensityMatrices
    def transition(self,U : Operator):
        is_Kraus = isinstance(U,Kraus)
        if is_Kraus and len(U.shape) > 3:
            U = np.swapaxes(U,0,-3)
        output = U * self * U.adjoint()
        if isinstance(U, Kraus):
            if len(U.shape) > 3:
                output = density_matrix(np.swapaxes(output,0,-3))
            output = output.sum(axis = -3)
        return density_matrix(output)

    def partial_trace(self, i):
        #shape : s, i,j
        n = int(2 ** i)
        m = len(self) // (2*n)

        ind = np.tile(np.arange(m),n) + np.arange(n).repeat(m)*2*m
        i1 = np.tile(ind,[len(ind),1])
        j1 = i1.T
        i2 = i1 + m
        j2 = i2.T

        return self[...,i1,j1] + self[...,i2,j2]

    def measure(self, psi):
        output = np.matmul(np.matmul([psi], rho),psi)
        assert output.imag == 0
        return output.real[0]


def density_matrix(x):
    if len(np.shape(x)) == 1: # x is a state
        #make x a proper state
        x = x / np.linalg.norm(x)
        x = np.matmul(np.expand_dims(x,1),[x])
    rho = DensityMatrix(np.shape(x), complex, np.array(x, dtype = complex))
    assert rho.shape[-1] == rho.shape[-2]
    return rho

class System:
    def __init__(self,size = None, config = None):
        if config is None:
            self.rho = density_matrix(size * [[[1,0],[0,0]]])
            self.careful_mode = True
        else:
            for key, value in config.items():
                setattr(self, key, value)
    def transition_qubit(self,U, qubits, in_place = True):
        if not in_place:
            sys = System(self.config())
            sys.transition_qubit(self, U, qubits)
            return sys
        if self.careful_mode:
            assert U.is_unitary
            assert U.shape[-1] == U.shape[-2]
            assert U.shape[-1] == 2 ** len(qubits)

        if len(qubits) == 1:
            i = qubits[0]
            self.rho[i] = self.rho[i].transition(U)

        elif len(qubits) == 2:
            i,j = qubits
            combined_rho = self.rho[i].tensor(self.rho[j])
            combined_rho = combined_rho.transition(U)
            self.rho[i] = combined_rho.partial_trace(1)
            self.rho[j] = combined_rho.partial_trace(0)


    def transition(self, U, in_place = True):
        '''
            U : list of operators of length self.rho
        '''
        if not in_place:
            sys = System(self.config())
            sys.transition(U)
            return sys
        self.rho = self.rho.transition(U)


import itertools as it
pauli = {
    'I' : operator(np.identity(2)),
    'X' : operator([[0,1],[1,0]]),
    'Y' : operator([[0,complex(0,-1)],[complex(0,1),0]]),
    'Z' : operator([[1,0],[0,-1]])
}


def crand(**args):
    return complex(np.random.rand(**args),np.random.rand(**args))
def expi(theta):
    return np.cos(theta) + complex(0,1) * np.sin(theta)


ideal_gates = {
    'id' : operator(np.identity(2)),
    'x' : pauli['X'],
    'sx' : operator([[complex(1,1),complex(1,-1)],[complex(1,-1),complex(1,1)]])/2,
    'rz' : lambda phi : operator([[expi(-phi/2),0],[0,expi(phi/2)]]),
}
cz = np.identity(4)
cz[3,3] = -1
ideal_gates['cz'] = cz

