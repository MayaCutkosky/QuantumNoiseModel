import numpy as np
import jax.numpy as jnp
def operator(x):
    n = len(x)
    M = Operator([n,n], buffer= np.array(x, dtype = complex))
    return M
def kraus(x, **kwargs):
    M = Kraus(buffer= x, **kwargs)
    return M



class Operator:
    def __init__(self, shape = [2,2], dtype = complex, buffer = None, data_object = None):
        self.type = "operator"
        self._data = None
        if buffer is not None:
            if isinstance(buffer, Operator):
                buffer = buffer._data
            elif isinstance(buffer, list):
                data = []
                for x in buffer:
                    if hasattr(x, 'tolist'):
                        data.append(x.tolist())
                    else:
                        data.append(x)
                    if data_object is None:
                        data_object = self.find_data_type(x)
                    elif data_object == "numpy":
                        if self.find_data_type(x) == 'jax':
                            data_object = 'jax'
                buffer = data
        if data_object is None:
            if buffer is None:
                self._data_object = "numpy"
            else:
                self.data_object = self.find_data_type(buffer)
                if self.data_object is None:
                    self.data_object = "numpy"
        else:
            self.data_object = data_object
        if self.data_object == 'jax':
            if buffer is None:
                self._data = jnp.zeros(shape, dtype = dtype)
            else:
                self._data = jnp.array(buffer, dtype = dtype)
        elif self.data_object == 'numpy':
            self._data = np.array(buffer, dtype = dtype)
        else:
            Exception("Not understood data type:", data_object)
        
        for attr in ['shape', 'real', 'imag']:
            setattr(self, attr, getattr(self._data, attr))
        if self.data_object == 'jax':
            self.np = jnp
        else:
            self.np = np

    def find_data_type(self, data = None):
        if data is None:
            data = self._data
        data_object = str(type(data))
        if "numpy" in data_object:
            return "numpy"
        elif "jax" in data_object:
            return "jax"
        return None

    def _create_new(self, **kwargs):
        output = self.__new__(type(self))
        output.__init__(**kwargs)
        return output

    def __mul__(self,y):
        if isinstance(y, Operator):
            if y.data_object == 'jax':
                data = jnp.matmul(self._data, y._data)
            else:
                data = self.np.matmul(self._data,y._data)
        else:
            data = self._data * y

        return self._create_new(buffer = data)

    def __rmul__(self, y):
        if isinstance(y, Operator): #will never be true!
            data = self.np.matmul(y._data, self._data)
        else:
            data = y * self._data
        return self._create_new(buffer = data)

    def _replace_binary_method(self,fun_name,y):
        fun = getattr(self._data, fun_name)
        if isinstance(y, Operator):
            data = fun(y._data)
        else:
            data = fun(y)
        return self._create_new(buffer = data)

    def __sub__(self,y):
        return self._replace_binary_method("__sub__", y)


    def __add__(self, y):
        return self._replace_binary_method('__add__', y)

    def __truediv__(self, y):
        #Better not to use division for operator and operator. Only sensible interpretation is multiply by inverse which is less confusing to just code. Could use for constants though. And this is used later.
        if isinstance(y, Operator):
            return Exception('Operator-Operator division not allowed. Use Operator * Operator.adjoint()' )
        else:
            return self._create_new(buffer = self._data / y )

    def __floordiv__(self, y): #don't actually see myself using this...
        if isinstance(y, Operator):
            return Exception('Operator-Operator division not allowed. Use Operator * Operator.adjoint()' )
        else:
            return self._create_new(buffer = self._data // y )

    def tensor(self, y):
        if isinstance(y, Operator):
            y = y._data
        data = self.np.kron(self._data,y)
        return self._create_new(buffer = data)

    def adjoint(self):
        data = self.np.swapaxes(self.np.conj(self._data),-1,-2)
        return self._create_new(buffer = data)

    def is_unitary(self):
        return self * self.adjoint() == np.identity(len(self))

    def __eq__(self,x):
        if isinstance(x, list):
            for xi, yi in zip(x, self):
                if not yi == xi:
                    return False
            return True
        if isinstance(x, Operator):
            x = x._data
        return self.np.max(np.abs(self._data - x)) < 1e-10

    def __len__(self):
        return len(self._data)

    def __getitem__(self,i):
        return self._create_new(buffer = self._data[i])
    def __setitem__(self, i, value):
        if isinstance(value, Operator):
            value = value._data
        
        if self.data_object == 'jax':
            self._data = self._data.at[i].set(value)
        elif self.data_object == 'numpy' :
            self._data[i] = np.array(value)
    def __iter__(self):
        for x in self._data:
            yield self._create_new(buffer = x)

    def _replace_numpy_method(self,fun_name,*args, **kwargs):
        fun = getattr(self.np,fun_name)
        data = fun(self._data, *args, **kwargs)
        return self._create_new(buffer = data)

    def sum(self, **kwargs):
        return self._replace_numpy_method('sum', **kwargs)
    def round(self, decimals):
        return self._replace_numpy_method('round',decimals)

    def tolist(self):
        return self._data.tolist()

    def __str__(self):
        return str(self._data)
    def __repr__(self):
        return self.type + '(' + str(self._data) + ')'

    def __call__(self,*args):
        return self._data[*args]
    
    def swapaxes(self, axis1, axis2):
        self._data = self.np.swapaxes(self._data, axis1, axis2)
        self.shape = self._data.shape
        

        

class Kraus(Operator):
    def __init__(self, shape = [1,2,2], *args, **kwargs):
        super().__init__(shape, *args, **kwargs)
        self.type = 'kraus'
    def tensor(self, y):
        if isinstance(y, Operator):
            y = y._data
        if isinstance(y, Kraus):
            data = self.np.vectorize(self.np.kron)(self._data, y)
        else:
            data = self.np.kron(self._data,y)
        return self._create_new(buffer = data)
    def extend(self, y):
        if isinstance(y, Operator):
            if y.data_object == 'jax':
                self.np = jnp
            if isinstance(y, Kraus):
                y = self.np.split(y._data, len(y))
        x = self.np.split(self._data, len(self))
        self._data = self.np.vstack(x + y)
        

class DensityMatrix(Operator): #allow for list of DensityMatrices
    def transition(self,U : Operator):
        is_Kraus = isinstance(U,Kraus)
        if is_Kraus and len(U.shape) > 3:
            self._data = self.np.expand_dims(self._data, -3)
        output = self._create_new(buffer = (U * self * U.adjoint())._data)
        if is_Kraus:
            if len(U.shape) > 3:
                self._data = self.np.reshape(self._data, self.shape)
            output = output.sum(axis = -3)
        return output

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
        output = self.np.matmul(self.np.matmul([psi], self),psi)
        assert output.imag == 0
        return output.real[0]


def density_matrix(x, **kwargs):
    if len(np.shape(x)) == 1: # x is a state
        #make x a proper state
        x = x / np.linalg.norm(x)
        x = np.matmul(np.expand_dims(x,1),[x])
    rho = DensityMatrix(np.shape(x), complex, np.array(x, dtype = complex), **kwargs)
    assert rho.shape[-1] == rho.shape[-2]
    return rho

class System:
    def __init__(self,size = None, config = None, **kwargs):
        if config is None:
            self.rho = density_matrix(size * [[[1,0],[0,0]]], **kwargs)
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
ideal_gates['cz'] = operator(cz)

