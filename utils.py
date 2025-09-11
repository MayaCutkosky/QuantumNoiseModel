import numpy as np
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

def operator(x, data_object = None):
    n = len(x)
    if data_object =='jax':
        M = JaxOperator( np.array(x, dtype = complex) )
    else:
        M = Operator([n,n], buffer= np.array(x, dtype = complex), data_object=data_object)
    return M
def kraus(x, data_object = None, **kwargs):
    if data_object == 'jax':
        M = JaxKraus(buffer = x)
    else:
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
        output = type(self)(**kwargs)
        return output

    def __mul__(self,y):
        data_object = self.data_object
        if isinstance(y, Operator):
            if y.data_object == 'jax':
                data = jnp.matmul(self._data, y._data)
                data_object = 'jax'
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
            y_data = y._data
        else:
            y_data = y
        data = self.np.kron(self._data,y_data)
        if isinstance(y, Kraus):
            return y._create_new(buffer = data)
        return self._create_new(buffer = data)
    def rtensor(self, y):
        if isinstance(y, Operator):
            y_data = y._data
        else:
            y_data = y
        data = self.np.kron(y_data, self._data)
        if isinstance(y, Kraus):
            return y._create_new(buffer = data)

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
        return self.np.max(np.abs(self._data - x)) < 1e-6

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
        
    def copy(self):
        return self._create_new(buffer = self._data)
        

        

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
    def is_Kraus(self):
        return np.sum(self * self.adjoint(),axis=-3) == np.identity(self.shape[-1])


class DensityMatrix(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.type = 'density_matrix'
        self.num_qubits = np.log2(self.shape[-1])
        assert self.num_qubits % 1 == 0
        self.num_qubits = int(self.num_qubits)
    def transition(self,U : Operator):
        is_Kraus = isinstance(U,Kraus)
        if is_Kraus and len(U.shape) > 3:
            self._data = self.np.expand_dims(self._data, -3)
        output = self._create_new(buffer = (U * self * U.adjoint())._data)
        if is_Kraus:
            if len(U.shape) > 3:
                self._data = self.np.reshape(self._data, self.shape)
            output = output.sum(axis = -3)
        # trace = jnp.trace(output._data, 0,-1,-2)
        # if trace.shape == ():
        #     trace = jnp.tile(trace, [self.shape[-1], self.shape[-2]])
        # else:
        #     trace = jnp.tile(trace, [self.shape[-1], self.shape[-2],1]).T
        # output._data = output._data / trace
        return output
    
    def transition_qubit(self,U, qubits):
        qubits = list(qubits)
        qubits.sort()
        if len(qubits) == 2:
            U0, U1 = U
            mid_qubits = operator( np.identity(int(2 ** int(qubits[1] - qubits[0] - 1))) )
            op = operator([[1,0],[0,0]]).tensor(mid_qubits).tensor(U0) 
            op += operator([[0,0],[0,1]]).tensor(mid_qubits).tensor(U1)
        else:
            op = U
        op = operator(np.identity(2 ** qubits[0]) ).tensor(op).tensor(operator(np.identity(2 ** (self.num_qubits - qubits[-1]-1))))
        return self.transition(op)


    def partial_trace(self, i):
        list_of_traces = False
        if isinstance(i, list):
            l = i
            l .sort()
            l.reverse()
            list_of_traces = True
        elif hasattr(i,'shape') and i.shape != ():
            l = self.np.flip(i.sort())
            list_of_traces = True
        if list_of_traces:
            rho = self._create_new(buffer = self._data)
            for i in l:
                rho = rho.partial_trace(i)
            return rho
        
        #shape : s, i,j
        n = 2 ** i
        m = len(self) // (2*n)
        ind = self.np.tile(self.np.arange(m),n) + self.np.arange(n).repeat(m)*2*m
        i1 = self.np.tile(ind,[len(ind),1])
        j1 = i1.T
        i2 = i1 + m
        j2 = i2.T

        return self[...,j1,i1] + self[...,j2,i2]

    def measure(self, psi):
        output = self.np.matmul(self.np.matmul([psi], self),psi)
        assert output.imag == 0
        return output.real[0]
    
    def trace(self):
        return self._data.trace()

import jax
from equinox import Module
import equinox as eqx
from types import ModuleType
class JaxOperator(Operator, eqx.Module):
    type : str = eqx.field(static = True)
    _data : jax.Array
    data_object : str  = eqx.field(static = True)
    np : ModuleType = eqx.field(static = True)
    real : jax.Array
    imag : jax.Array
    shape : tuple

    def __init__(self, buffer):
        super().__init__(buffer = buffer, data_object = 'jax')

class JaxKraus(Kraus, Module):
    type : str = eqx.field(static=True)
    _data : jax.Array
    data_object : str  = eqx.field(static = True)
    np : ModuleType = eqx.field(static=True)
    real : jax.Array
    imag : jax.Array
    shape : tuple = eqx.field( static = True)
    
    def __init__(self, buffer):
        super().__init__(buffer = buffer, data_object = 'jax')

class JaxDensityMatrix(DensityMatrix, Module):
    type : str = eqx.field(static = True)
    _data : jax.Array
    num_qubits : jax.Array
    data_object : str  = eqx.field(static = True)
    np : ModuleType = eqx.field(static=True)
    real : jax.Array
    imag : jax.Array
    shape : tuple
    
    def __init__(self, buffer):
        super().__init__(buffer = buffer, data_object = 'jax')




def density_matrix(x, **kwargs):
    if len(np.shape(x)) == 1: # x is a state
        #make x a proper state
        x = x / np.linalg.norm(x)
        x = np.matmul(np.expand_dims(x,1),[x])
    if kwargs.get('data_object') == 'jax':
        rho = JaxDensityMatrix(jnp.array(np.array(x, dtype = complex)))
    else:
        rho = DensityMatrix(np.shape(x), complex, np.array(x, dtype = complex), **kwargs)
    assert rho.shape[-1] == rho.shape[-2]
    return rho

def digit_ind_to_binary_inds(n):
    return jnp.tile( jnp.arange(2**n), [1,1]).T // 2 ** (n - 1 - jnp.tile(jnp.arange(n-1,-1,-1), [1,1]) ) % 2

def binary_inds_to_digit_ind(inds):
    n = len(inds[0])
    return  jnp.sum(2 ** jnp.arange(n) * inds,axis=1) 


from jax_utils import make_inverse_map, fixed_size_where, map_qubits
import equinox as eqx




class System:
    def __init__(self,  size = None, **kwargs):
        self.rho = density_matrix([1] +[0]*(2**size-1), **kwargs)
        self.num_qubits = size

    def transition_qubit(self,U, qubits):
        def expand_U(U, q):
            op = U.tensor( np.identity(2 ** q) ).rtensor(  np.identity(2 ** (self.num_qubits - q-1)) )
            return op
    
        def expand_U2(U, qubits):
            U0, U1 = U
            mid_qubits = np.identity( 2 ** (qubits[1] - qubits[0] - 1) )
            op = U0.tensor(mid_qubits).tensor(  np.array([[1,0],[0,0]]) ) 
            op += U1.tensor(mid_qubits).tensor(  np.array([[0,0],[0,1]]) )
            op = op.tensor( np.identity(2 ** qubits[0]) ).rtensor(  np.identity(2 ** (self.num_qubits - qubits[-1]-1)) )
            return op
        
        def create_fun(q0, fun):
            return lambda U : fun(U, q0)

        
        if len(qubits) == 1:
            U = JaxOperator(U._data)
            U = expand_U(U,qubits[0])
        elif len(qubits) == 2:
            U = JaxOperator(U[0]._data), JaxOperator(U[1]._data)
            i = np.min(qubits)
            j = np.max(qubits)
            ind =i* self.num_qubits - i * (i+1) // 2  + j - i - 1
            U = jax.lax.switch(ind , [ create_fun(q, expand_U2) for q in it.combinations(range(self.num_qubits),2)], U)
        self.rho = self.rho.transition(U)
        return self

    def transition(self, U_array):
        U = U_array._create_new(buffer = [1])
        for Ui in U_array:
            U = U.tensor(Ui)
        self.rho = self.rho.transition(U)
        return self
    def calc_probabilities(self, readout_qubits = None):
        if readout_qubits is None:
            readout_qubits = jnp.arange(self.num_qubits)
            
        rho = self.rho.copy()
        def get_partial_trace_inds(i, num_qubits):
            def fun(k):
                n = 2 ** k
                m = 2 ** num_qubits // (2*n)
                ind = rho.np.tile(rho.np.arange(m),n) + rho.np.arange(n).repeat(m)*2*m
                i1 = rho.np.tile(ind,[len(ind),1])
                j1 = i1.T
                i2 = i1 + m
                j2 = i2.T
                return i1,i2,j1,j2
            def create_fun(k):
                return lambda : fun(k)
            return fun(i)
        
        extra_qubit_inds = jnp.argsort(jnp.isin(jnp.arange(self.num_qubits),readout_qubits) )[: self.num_qubits - len(readout_qubits) ]
        num_qubits = self.num_qubits
        for i in jnp.flip(extra_qubit_inds.sort()):
            i1,i2,j1,j2 = get_partial_trace_inds(i, num_qubits)
            num_qubits -= 1
            rho = rho[...,j1,i1] + rho[...,j2,i2]
        ind = readout_qubits.at[readout_qubits.argsort()].set( jnp.arange(len(readout_qubits)) )
        ind = map_qubits(readout_qubits, np.arange(len(readout_qubits)))
        
        ind = binary_inds_to_digit_ind(  digit_ind_to_binary_inds ( num_qubits)[:,ind] )
        
        prob = rho(ind, ind).real
        return prob

class JaxSystem(Module):
    rho : JaxDensityMatrix
    num_qubits : int = eqx.field(static=True)
    def __init__(self,  size = None, **kwargs):
        self.rho = density_matrix([1] +[0]*(2**size-1), data_object='jax')
        self.num_qubits = size

    #@eqx.filter_jit
    def transition_qubit(self,U, qubits):
        def expand_U(U, q):
            op = U.tensor( np.identity(2 ** q) ).rtensor(  np.identity(2 ** (self.num_qubits - q-1)) )
            return op
    
        def expand_U2(U, qubits):
        
            U0, U1 = U
            mid_qubits = np.identity( 2 ** (qubits[1] - qubits[0] - 1) )
            op = U0.tensor(mid_qubits).tensor(  np.array([[1,0],[0,0]]) ) 
            op += U1.tensor(mid_qubits).tensor(  np.array([[0,0],[0,1]]) )
            op = op.tensor( np.identity(2 ** qubits[0]) ).rtensor(  np.identity(2 ** (self.num_qubits - qubits[-1]-1)) )
            return op
        
        def create_fun(q0, fun):
            return lambda U : fun(U, q0)

        
        if len(qubits) == 1:
            U = JaxOperator(U._data)
            U = jax.lax.switch( qubits[0] , [ create_fun(q0, expand_U) for q0 in range(self.num_qubits)], U )
        elif len(qubits) == 2:
            U = JaxOperator(U[0]._data), JaxOperator(U[1]._data)
            i = jnp.min(qubits)
            j = jnp.max(qubits)
            ind =i* self.num_qubits - i * (i+1) // 2  + j - i - 1
            U = jax.lax.switch(ind , [ create_fun(q, expand_U2) for q in it.combinations(range(self.num_qubits),2)], U)
        return eqx.tree_at(lambda sys: sys.rho, self,  self.rho.transition(U) )

    def transition(self, U_array):
        U = U_array._create_new(buffer = [1])
        for Ui in U_array:
            U = U.tensor(Ui)
        return eqx.tree_at(lambda sys: sys.rho, self,  self.rho.transition(U) )
    
    def calc_probabilities(self, readout_qubits = None):
        if readout_qubits is None:
            readout_qubits = jnp.arange(self.num_qubits)
            
        rho = self.rho.copy()
        def get_partial_trace_inds(i, num_qubits):
            def fun(k):
                n = 2 ** k
                m = 2 ** num_qubits // (2*n)
                ind = rho.np.tile(rho.np.arange(m),n) + rho.np.arange(n).repeat(m)*2*m
                i1 = rho.np.tile(ind,[len(ind),1])
                j1 = i1.T
                i2 = i1 + m
                j2 = i2.T
                return i1,i2,j1,j2
            def create_fun(k):
                return lambda : fun(k)
            return jax.lax.switch(i, [create_fun(k) for k in range(num_qubits)])
        
        extra_qubit_inds = jnp.argsort(jnp.isin(jnp.arange(self.num_qubits),readout_qubits) )[: self.num_qubits - len(readout_qubits) ]
        num_qubits = self.num_qubits
        for i in jnp.flip(extra_qubit_inds.sort()):
            i1,i2,j1,j2 = get_partial_trace_inds(i, num_qubits)
            num_qubits -= 1
            rho = rho[...,j1,i1] + rho[...,j2,i2]
        ind = readout_qubits.at[readout_qubits.argsort()].set( jnp.arange(len(readout_qubits)) )
        ind = map_qubits(readout_qubits, np.arange(len(readout_qubits)))
        
        ind = binary_inds_to_digit_ind(  digit_ind_to_binary_inds ( num_qubits)[:,ind] )
        
        prob = rho(ind, ind).real
        return prob
        
            
            
            

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
    'cz' : (pauli['I'], pauli['Z']),
}

#cz = np.identity(4)
#cz[3,3] = -1
#ideal_gates['cz'] = operator(cz)

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

def softmax(x,axis = None):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis = axis)

